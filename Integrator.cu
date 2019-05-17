#include "Integrator.cuh"

__global__ void operator_LvLr_kernel(Scalar dt, unsigned int natom, Scalar* dev_rx, Scalar* dev_ry, Scalar* dev_rz, Scalar* dev_vx, Scalar* dev_vy, Scalar* dev_vz, Scalar* dev_fx, Scalar* dev_fy, Scalar* dev_fz, Scalar* dev_mass) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	while (idx < natom) {
		Scalar htFmx = dev_fx[idx] / dev_mass[idx] * dt / 2.; // F/m * dt /2
		Scalar htFmy = dev_fy[idx] / dev_mass[idx] * dt / 2.;
		Scalar htFmz = dev_fz[idx] / dev_mass[idx] * dt / 2.;
		dev_vx[idx] += htFmx;
		dev_vy[idx] += htFmy;
		dev_vz[idx] += htFmz;
		dev_rx[idx] += dev_vx[idx] * dt;
		dev_ry[idx] += dev_vy[idx] * dt;
		dev_rz[idx] += dev_vz[idx] * dt;
		idx += blockDim.x*gridDim.x;
	}
}
void VelocityVerlet::Operator_LvLr(Scalar dt) {
	dim3 grid(blocksPerGrid, 1);
	dim3 block(threadsPerBlock, 1); 
	// group the Lr and Lv operator to save the kernel launch time
	operator_LvLr_kernel << <grid, block >> > (dt, mddata->hdata.Natom, mddata->ddata.dev_rx, mddata->ddata.dev_ry, mddata->ddata.dev_rz, mddata->ddata.dev_vx, mddata->ddata.dev_vy, mddata->ddata.dev_vz, mddata->ddata.dev_fx, mddata->ddata.dev_fy, mddata->ddata.dev_fz, mddata->ddata.dev_mass);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
}

__global__ void operator_Lv_kernel(Scalar dt, unsigned int natom, Scalar* dev_vx, Scalar* dev_vy, Scalar* dev_vz, Scalar* dev_fx, Scalar* dev_fy, Scalar* dev_fz, Scalar* dev_mass) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	while (idx < natom) {
		Scalar htFmx = dev_fx[idx] / dev_mass[idx] * dt; // F/m * dt
		Scalar htFmy = dev_fy[idx] / dev_mass[idx] * dt;
		Scalar htFmz = dev_fz[idx] / dev_mass[idx] * dt;
		dev_vx[idx] += htFmx;
		dev_vy[idx] += htFmy;
		dev_vz[idx] += htFmz;
		idx += blockDim.x*gridDim.x;
	}
}
void VelocityVerlet::Operator_Lv(Scalar dt) {
	dim3 grid(blocksPerGrid, 1);
	dim3 block(threadsPerBlock, 1);
	operator_Lv_kernel << <grid, block >> > (dt, mddata->hdata.Natom, mddata->ddata.dev_vx, mddata->ddata.dev_vy, mddata->ddata.dev_vz, mddata->ddata.dev_fx, mddata->ddata.dev_fy, mddata->ddata.dev_fz, mddata->ddata.dev_mass);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
}

void VelocityVerlet::Operator_L(Scalar dt, VerletList* vlist) {
	Operator_LvLr(dt);
	UpdateForce(vlist);
	Operator_Lv(dt / 2.);
}

__global__ void updateForceKernel(NonBondedInteraction** dev_nbint, unsigned int typeNum, unsigned int maxNeigh, int* dev_vlist, int* dev_nlist,unsigned int natom, Scalar* dev_rx, Scalar* dev_ry, Scalar* dev_rz, Scalar* dev_fx, Scalar* dev_fy, Scalar* dev_fz, int* dev_type, RectangularBox* dev_rect){
	// vlist[i][j] ith atom neighbor's index j start from 1 
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	Scalar L = dev_rect->Length;
	Scalar T = dev_rect->Width;
	Scalar H = dev_rect->Height;
	Scalar hL = L / 2.;
	Scalar hT = T / 2.;
	Scalar hH = H / 2.;
	while (idx < natom) {
		Scalar facx = 0., facy = 0., facz = 0.;
		unsigned int myType = dev_type[idx];
		unsigned int myNumOfNeigh = dev_nlist[idx]; // wrong, why cannot use nlist(idx)????
		for (int i = 1; i <= myNumOfNeigh; i++ ) {
			unsigned int shIdx = dev_vlist[idx*maxNeigh + i];
			if (shIdx != idx) {
				unsigned int shType = dev_type[shIdx];
				Scalar dx = dev_rx[shIdx] - dev_rx[idx];
				Scalar dy = dev_ry[shIdx] - dev_ry[idx];
				Scalar dz = dev_rz[shIdx] - dev_rz[idx];
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
				Scalar3 dr(dx, dy, dz);
				Scalar3 f = dev_nbint[myType*typeNum + shType]->nbforce(dr);
				facx += f.x;
				facy += f.y;
				facz += f.z;
			}
		}
		/*if (idx == natom - 1) {
			printf("%f\t%d\n", facx, idx);
		}*/
		dev_fx[idx] = facx;
		dev_fy[idx] = facy;
		dev_fz[idx] = facz;
		idx += gridDim.x*blockDim.x;
	}
}
void VelocityVerlet::UpdateForce(VerletList* vlist) {
	dim3 grid(blocksPerGrid, 1);
	dim3 block(threadsPerBlock, 1); 
	updateForceKernel<<<grid,block>>>(mddata->ddata.dev_nbint, mddata->atomIndex.size(), mddata->nblist->getMaxNeigh(),mddata->nblist->dev_verletlist, mddata->nblist->dev_nlist, mddata->hdata.Natom, mddata->ddata.dev_rx, mddata->ddata.dev_ry, mddata->ddata.dev_rz, mddata->ddata.dev_fx, mddata->ddata.dev_fy, mddata->ddata.dev_fz, mddata->ddata.dev_atomType, mddata->ddata.dev_boxRect);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
}
