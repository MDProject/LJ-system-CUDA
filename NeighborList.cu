#include "NeighborList.cuh"

void VerletList::AllocateCPUMemory(unsigned int N) {
	coorRecordx = (Scalar*)malloc(N * sizeof(Scalar));
	coorRecordy = (Scalar*)malloc(N * sizeof(Scalar));
	coorRecordz = (Scalar*)malloc(N * sizeof(Scalar));
	host_verletlist = (int*)malloc(N * MaxNeighborNum * sizeof(int));
	host_nlist = (int*)malloc(N * sizeof(int));
	memset(host_nlist, 0, N * sizeof(int));
}

void VerletList::AllocateGPUMemory(unsigned int N) {
	handleError(cudaMalloc(&dev_coorRecordx, N * sizeof(Scalar)));
	handleError(cudaMalloc(&dev_coorRecordy, N * sizeof(Scalar)));
	handleError(cudaMalloc(&dev_coorRecordz, N * sizeof(Scalar)));
	handleError(cudaMalloc(&dev_verletlist, N * MaxNeighborNum * sizeof(int)));
	handleError(cudaMalloc(&dev_nlist, N * sizeof(int)));
	handleError(cudaMalloc(&dev_rect, sizeof(RectangularBox)));
	handleError(cudaMallocManaged(&hd_build, sizeof(bool)));
	// set nlist to 0
	handleError(cudaMemset(dev_nlist, 0, N * sizeof(int)));
}

void VerletList::FreeCPUMemory(unsigned int N) {
	free(coorRecordx);
	free(coorRecordy);
	free(coorRecordz);
}

void VerletList::FreeGPUMemory(unsigned int N) {
	handleError(cudaFree(hd_build));
	handleError(cudaFree(dev_nlist));
	handleError(cudaFree(dev_verletlist));
	handleError(cudaFree(dev_coorRecordx));
	handleError(cudaFree(dev_coorRecordy));
	handleError(cudaFree(dev_coorRecordz));
	handleError(cudaFree(dev_rect));
}

VerletList::VerletList(MDData mddata, Scalar d2):scaleFactor(8.), rv(d2) {
	unsigned int natom = mddata.hdata.Natom;
	unsigned int atomTypeNum = mddata.atomIndex.size();
	rc = 0.;
	for (int i = 0; i < atomTypeNum; i++) {
		for (int j = 0; j < atomTypeNum; j++) {
			Scalar rctmp = mddata.hdata.nbint[i][j]->getCutOff();
			if (rctmp > rc) {
				rc = rctmp;
			}
		}
	}
	std::cout << "Maximum Nonbonded interaction cut-off radius is " << rc << std::endl;
	if (rv <= rc) {
		std::cout << "NeighborList radius is too small, please increase" << std::endl; system("pause");
		exit(0);
	}
	Scalar volume = mddata.hdata.boxRect.Volume();
	MaxNeighborNum = scaleFactor*PI*rv*rv*rv / volume*natom; // Initial neighbor list param
	std::cout << "Scale Factor is " << scaleFactor << " and currently maximum neighbor atoms allowed is " << MaxNeighborNum << std::endl;
	AllocateCPUMemory(natom);
	AllocateGPUMemory(natom);
	*hd_build = false;
	memcpy(coorRecordx, mddata.hdata.rx, natom * sizeof(Scalar));
	memcpy(coorRecordy, mddata.hdata.ry, natom * sizeof(Scalar));
	memcpy(coorRecordz, mddata.hdata.rz, natom * sizeof(Scalar));
	handleError(cudaMemcpy(dev_coorRecordx, coorRecordx, natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(dev_coorRecordy, coorRecordy, natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(dev_coorRecordz, coorRecordz, natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(dev_rect, &mddata.hdata.boxRect, sizeof(RectangularBox), cudaMemcpyHostToDevice));
	std::cout << natom << " atoms current states has been recorded" << std::endl;
}

__global__ void buildVerletListKernel(int natom, Scalar* dev_rx, Scalar* dev_ry, Scalar* dev_rz, Scalar rc, int maxNeighbor,int* dev_vlist, int* dev_nlist, RectangularBox* dev_rect) {
	int idx = blockIdx.x* blockDim.x + threadIdx.x;
	Scalar L = dev_rect->Length;
	Scalar T = dev_rect->Width;
	Scalar H = dev_rect->Height;
	if (idx == 0) {
		for (int k = 0; k < natom - 1; k++) {
			Scalar myPosition_x = dev_rx[k];
			Scalar myPosition_y = dev_ry[k];
			Scalar myPosition_z = dev_rz[k];
			for (int i = k + 1; i < natom ; i++) { // loop for each atom; Could be further optimalized using CellList
				for (int nx = -1; nx < 2; nx++) {
					for (int ny = -1; ny < 2; ny++) {
						for (int nz = -1; nz < 2; nz++) {
							Scalar shPosition_x = dev_rx[i] + nx*L;
							Scalar shPosition_y = dev_ry[i] + ny*T;
							Scalar shPosition_z = dev_rz[i] + nz*H;
							Scalar dist2 = (myPosition_x - shPosition_x)*(myPosition_x - shPosition_x) + (myPosition_y - shPosition_y)*(myPosition_y - shPosition_y) + (myPosition_z - shPosition_z)*(myPosition_z - shPosition_z);
							if (dist2 <= rc*rc) {
								dev_nlist[i] = dev_nlist[i] + 1;
								dev_nlist[k] = dev_nlist[k] + 1;
								dev_vlist[k*maxNeighbor + dev_nlist[k]] = i;
								dev_vlist[i*maxNeighbor + dev_nlist[i]] = k;
								break;
							}
						}
					}
				}
			}
		}
	}
	else { return; }
}

void VerletList::updateList(MDData& mddata) {
	handleError(cudaMemset(dev_nlist, 0, mddata.hdata.Natom * sizeof(int)));
	dim3 block(threadsPerBlock, 1);
	dim3 grid(blocksPerGrid, 1);
	buildVerletListKernel << <grid, block >> > (mddata.hdata.Natom, mddata.ddata.dev_rx, mddata.ddata.dev_ry, mddata.ddata.dev_rz, rv, MaxNeighborNum, dev_verletlist, dev_nlist, dev_rect);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
	handleError(cudaMemcpy(dev_coorRecordx, mddata.ddata.dev_rx, mddata.hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToDevice));
	handleError(cudaMemcpy(dev_coorRecordy, mddata.ddata.dev_ry, mddata.hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToDevice));
	handleError(cudaMemcpy(dev_coorRecordz, mddata.ddata.dev_rz, mddata.hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
}

void VerletList::makeNewList(MDData& mddata) {
	dim3 block(threadsPerBlock, 1);
	dim3 grid(blocksPerGrid, 1); 
	buildVerletListKernel << <grid, block >> > (mddata.hdata.Natom, mddata.ddata.dev_rx, mddata.ddata.dev_ry, mddata.ddata.dev_rz, rv, MaxNeighborNum, dev_verletlist, dev_nlist, dev_rect);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
}

__global__ void updateRebuildKernel(unsigned int natom, Scalar maxdr, Scalar* dev_rxo, Scalar* dev_ryo, Scalar* dev_rzo, Scalar* dev_rxn, Scalar* dev_ryn, Scalar* dev_rzn, bool* hd_build, RectangularBox* dev_rect) { // o for 'old' and n for 'new', current position
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	/*if (idx == 0) {
		//printf("%d\n", *dev_build);
	}*/
	Scalar rxn = dev_rxn[idx];
	Scalar ryn = dev_ryn[idx];
	Scalar rzn = dev_rzn[idx];
	Scalar rxo = dev_rxo[idx];
	Scalar ryo = dev_ryo[idx];
	Scalar rzo = dev_rzo[idx]; // latency hidden 
	Scalar L = dev_rect->Length;
	Scalar T = dev_rect->Width;
	Scalar H = dev_rect->Height;
	Scalar hL = L / 2.;
	Scalar hT = T / 2.;
	Scalar hH = H / 2.;
	while (idx < natom) {
		if (*hd_build) {
			return;
		}
		Scalar dx = rxn - rxo;
		Scalar dy = ryn - ryo;
		Scalar dz = rzn - rzo;
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
		Scalar3 drvec(dx, dy, dz);
		Scalar dr = drvec.norm();
		if (dr> maxdr) { 
			*hd_build = true;
			return;
		}
		idx += blockDim.x*gridDim.x;
	}
}
bool VerletList::ifRebuild(MDData& mddata) {
	Scalar maxdr = rv - rc;
	dim3 block(threadsPerBlock, 1);
	dim3 grid(blocksPerGrid, 1);
	updateRebuildKernel << <grid, block >> > (mddata.hdata.Natom, maxdr, dev_coorRecordx, dev_coorRecordy, dev_coorRecordz, mddata.ddata.dev_rx, mddata.ddata.dev_ry, mddata.ddata.dev_rz, hd_build, mddata.ddata.dev_boxRect);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
	if (*hd_build) {
		*hd_build = false;
		return true;
	}
	else {
		return false;
	}
}

__device__ __host__ int VerletList::vlist(int i, int j) {
	return dev_verletlist[i*MaxNeighborNum + j]; // ith atom jth neighbor |		i: 0 - [natom-1]	j: 0 - [nlist(i)-1]
}
__device__ __host__ int VerletList::nlist(int i) {
	return dev_nlist[i];
}


