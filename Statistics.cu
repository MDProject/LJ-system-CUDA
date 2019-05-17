#include "Statistics.cuh"

Statistics::Statistics(MDData mddata, NonBondedNeighborList* nblist) {
	this->mddata = mddata;
	this->nblist = nblist;
}
Statistics::Statistics(MDData mddata) {
	this->mddata = mddata;
}
__global__ void computeKineticEnerg(Scalar* dev_vx, Scalar* dev_vy, Scalar* dev_vz, unsigned int natom, Scalar* hd_kE) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

}
Scalar Statistics::kineticEnergy() {
	/*Scalar* hd_kE;
	handleError(cudaMallocManaged(&hd_kE, sizeof(Scalar)));*/
	Scalar kE = 0.;
	for (int i = 0; i < mddata.hdata.Natom; i++) {
		kE += mddata.hdata.vx[i] * mddata.hdata.vx[i] + mddata.hdata.vy[i] * mddata.hdata.vy[i] + mddata.hdata.vz[i] * mddata.hdata.vz[i];
	}
	kE = kE / 2.;
	return kE;
}

Scalar Statistics::interPotential() {
	Scalar pot = 0.;
	Scalar L = mddata.hdata.boxRect.Length;
	Scalar T = mddata.hdata.boxRect.Width;
	Scalar H = mddata.hdata.boxRect.Height;
	Scalar hL = L / 2.;
	Scalar hT = T / 2.;
	Scalar hH = H / 2.;
	for (int i = 0; i < mddata.hdata.Natom; i++) {
		for (int j = 0; j < mddata.hdata.Natom; j++) {
			if (j != i) {
				Scalar dx = mddata.hdata.rx[j] - mddata.hdata.rx[i];
				Scalar dy = mddata.hdata.ry[j] - mddata.hdata.ry[i];
				Scalar dz = mddata.hdata.rz[j] - mddata.hdata.rz[i];
				// image selection
				if (dx > hL) {
					dx -= L;
				}
				if (dx < -hL) {
					dx += L;
				}
				if (dy > hT) {
					dy -= T;
				}
				if (dy < -hT) {
					dy += T;
				}
				if (dz > hH) {
					dz -= H;
				}
				if (dz < -hH) {
					dz += H;
				}
				Scalar3 dr3(dx, dy, dz);
				Scalar dr2 = dr3.norm2();
				if (dr2 < 2.5*2.5) {
					Scalar sr2i = 1. / dr2;
					Scalar sr6i = sr2i*sr2i*sr2i;
					Scalar sr12i = sr6i*sr6i;
					pot += 4. * (sr12i - sr6i);
				}
			}
		}
	}
	return pot / 2.;
}

Scalar Statistics::interPotentialNB() {
	Scalar pot = 0.;
	Scalar L = mddata.hdata.boxRect.Length;
	Scalar T = mddata.hdata.boxRect.Width;
	Scalar H = mddata.hdata.boxRect.Height;
	Scalar hL = L / 2.;
	Scalar hT = T / 2.;
	Scalar hH = H / 2.;
	unsigned int maxNeigh = nblist->getMaxNeigh();
	handleError(cudaMemcpy(nblist->host_verletlist, nblist->dev_verletlist, mddata.hdata.Natom * maxNeigh * sizeof(int), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(nblist->host_nlist, nblist->dev_nlist, mddata.hdata.Natom * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < mddata.hdata.Natom; i++) {
		unsigned int myType = mddata.hdata.atomType_int[i];
		for (int k = 1; k <= nblist->host_nlist[i]; k++) {
			unsigned int j = nblist->host_verletlist[i*maxNeigh + k];
			if (j != i) {
				unsigned int shType = mddata.hdata.atomType_int[j];
				Scalar dx = mddata.hdata.rx[j] - mddata.hdata.rx[i];
				Scalar dy = mddata.hdata.ry[j] - mddata.hdata.ry[i];
				Scalar dz = mddata.hdata.rz[j] - mddata.hdata.rz[i];
				// image selection
				if (dx > hL) {
					dx -= L;
				}
				if (dx < -hL) {
					dx += L;
				}
				if (dy > hT) {
					dy -= T;
				}
				if (dy < -hT) {
					dy += T;
				}
				if (dz > hH) {
					dz -= H;
				}
				if (dz < -hH) {
					dz += H;
				}
				Scalar3 dr3(dx, dy, dz);
				pot += mddata.hdata.nbint[myType][shType]->nbPotential(dr3);
			}
		}
	}
	return pot / 2.;
}

Scalar3 Statistics::COM_velocity() {
	Scalar comvx = 0.;
	Scalar comvy = 0.;
	Scalar comvz = 0.;
	for (int i = 0; i < mddata.hdata.Natom; i++) {
		comvx += mddata.hdata.vx[i];
		comvy += mddata.hdata.vy[i];
		comvz += mddata.hdata.vz[i];
	}
	return Scalar3(comvx / mddata.hdata.Natom, comvy / mddata.hdata.Natom, comvz / mddata.hdata.Natom);
}

Scalar3 Statistics::COM_force() {
	Scalar comfx = 0.;
	Scalar comfy = 0.;
	Scalar comfz = 0.;
	for (int i = 0; i < mddata.hdata.Natom; i++) {
		comfx += mddata.hdata.fx[i];
		comfy += mddata.hdata.fy[i];
		comfz += mddata.hdata.fz[i];
	}
	return Scalar3(comfx / mddata.hdata.Natom, comfy / mddata.hdata.Natom, comfz / mddata.hdata.Natom);
}