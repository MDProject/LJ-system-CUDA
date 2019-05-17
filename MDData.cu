/*
*** Verlet Neighbor List should be constructed and modified on GPU ***
*** avoid data transfer ***
*/

#include "MDData.cuh"

void MDData::FreeCPUMemory() {
	free(hdata.vx);
	free(hdata.vy);
	free(hdata.vz);
	free(hdata.fx);
	free(hdata.fy);
	free(hdata.fz);
	free(hdata.rx);
	free(hdata.ry);
	free(hdata.rz);
	free(hdata.mass);
	delete[] hdata.atomType;
}
void MDData::FreeGPUMemory() {
	
}
void MDData::AllocateCPUMemory(int N) {
	hdata.vx = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.fx = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.rx = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.vy = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.fy = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.ry = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.vz = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.fz = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.rz = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.mass = (Scalar*)malloc(N * sizeof(Scalar));
	hdata.atomType_int = (int*)malloc(N * sizeof(int));
	hdata.atomType = new std::string[N];
}
void MDData::writeGroToHost(char* path) {
	WriteGroToHost(path, hdata.vx, hdata.vy, hdata.vz, hdata.rx, hdata.ry, hdata.rz, hdata.atomType, hdata.Natom, hdata.boxRect);
}
void MDData::readGroToHost(char* path) {
	// Allocate Host Memory
	int natom = NumOfAtomGro(path);
	AllocateCPUMemory(natom);
	// read in *.gro file
	ReadGroToHost(path, hdata.vx, hdata.vy, hdata.vz, hdata.rx, hdata.ry, hdata.rz, hdata.atomType, &(hdata.Natom), hdata.boxRect);
	std::cout << "MD System Box size is " << "[ " << hdata.boxRect.Length << " , " << hdata.boxRect.Width << " , " << hdata.boxRect.Height << " ]" << std::endl;
	// Initialize force to 0 and mass to 1;
	atomIndex.push_back(hdata.atomType[0]); // at least one kind of atoms
	for (int i = 0; i < natom; i++) {
		hdata.fx[i] = 0.;
		hdata.fy[i] = 0.;
		hdata.fz[i] = 0.;
		hdata.mass[i] = 1.;
		for (int j = 0; j < atomIndex.size(); j++) {
			if (atomIndex[j].compare(hdata.atomType[i]) != 0) {
				atomIndex.push_back(hdata.atomType[i]);
			}
		}
	}
	printf("Read in Gro File to Host, total %d atoms and %d different types of atoms\n", hdata.Natom, atomIndex.size());
}

void MDData::AllocateGPUMemory(int N) {
	// current time step
	handleError(cudaMalloc(&ddata.dev_rx, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_ry, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_rz, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_vx, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_vy, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_vz, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_fx, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_fy, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_fz, N * sizeof(Scalar)));
	// Constant info
	handleError(cudaMalloc(&ddata.dev_mass, N * sizeof(Scalar)));
	handleError(cudaMalloc(&ddata.dev_atomType, N * sizeof(int)));
	handleError(cudaMalloc(&ddata.dev_boxRect, sizeof(RectangularBox)));
	handleError(cudaMalloc(&ddata.dev_natom, sizeof(int)));
}

void MDData::copyMDDataHostToDevice() { 
	std::cout << "Nonbonded Interaction on Host Registered" << std::endl;
	AllocateGPUMemory(hdata.Natom);
	// Transform atomType to int type
	int* atomTypeTmp = (int*)malloc(hdata.Natom * sizeof(int));
	for (int i = 0; i < hdata.Natom; i++) {
		for (int j = 0; j < atomIndex.size(); j++) {
			if (hdata.atomType[i].compare(atomIndex[j]) == 0) {
				atomTypeTmp[i] = j;
			}
		}
	}
	handleError(cudaMemcpy(ddata.dev_atomType, atomTypeTmp, hdata.Natom * sizeof(int), cudaMemcpyHostToDevice));
	memcpy(hdata.atomType_int, atomTypeTmp, hdata.Natom * sizeof(int));
	free(atomTypeTmp);
	// Copy hdata to ddata
	handleError(cudaMemcpy(ddata.dev_boxRect, &hdata.boxRect, sizeof(RectangularBox), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_mass, hdata.mass, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_natom, &hdata.Natom, sizeof(int), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_fx, hdata.fx, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_fy, hdata.fy, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_fz, hdata.fz, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_vx, hdata.vx, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_vy, hdata.vy, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_vz, hdata.vz, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_rx, hdata.rx, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_ry, hdata.ry, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(ddata.dev_rz, hdata.rz, hdata.Natom * sizeof(Scalar), cudaMemcpyHostToDevice));
	// copy dev_nbint_host to dev_nbint
	handleError(cudaMemcpy(ddata.dev_nbint, ddata.dev_nbint_host, atomIndex.size()*atomIndex.size()*sizeof(NonBondedInteraction*), cudaMemcpyHostToDevice));
	free(ddata.dev_nbint_host); // temporary store of GPU pointers of nonbonded interaction
	ddata.typeNum = atomIndex.size();
	std::cout << "Host Data Copied to Device" << std::endl;
}

void MDData::copyMDDataDeviceToHost() {
	handleError(cudaMemcpy(hdata.rx, ddata.dev_rx, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.ry, ddata.dev_ry, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.rz, ddata.dev_rz, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.vx, ddata.dev_vx, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.vy, ddata.dev_vy, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.vz, ddata.dev_vz, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.fx, ddata.dev_fx, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.fy, ddata.dev_fy, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
	handleError(cudaMemcpy(hdata.fz, ddata.dev_fz, hdata.Natom * sizeof(Scalar), cudaMemcpyDeviceToHost));
}

void MDData::registerNeighborListMethod(NonBondedNeighborList* nbl) {
	std::cout << "Verlet Neighbor List is Adapted" << std::endl;
	nblist = nbl;
}
int MDData::atomTypeString2Int(std::string str) {
	for (int i = 0; i < atomIndex.size(); i++) {
		if (atomIndex[i].compare(str) == 0) {
			return i;
		}
	}
}
void MDData::registerNonBondedInteraction(NonBondedInteraction* nbinter, NonBondedInteraction* dev_nbinter, std::string atom0, std::string atom1) {
	if (nbint_idx == 0) {
		std::cout << "Register Nonbonded Interaction on Host\t\t";
		hdata.nbint = (NonBondedInteraction***)malloc(atomIndex.size() * sizeof(NonBondedInteraction**));
		for (int i = 0; i < atomIndex.size(); i++) {
			hdata.nbint[i]= (NonBondedInteraction**)malloc(atomIndex.size() * sizeof(NonBondedInteraction*));
		}
		std::cout << "Interaction Map size: [" << atomIndex.size() << " * " << atomIndex.size() << "]" << std::endl;
		handleError(cudaMalloc(&ddata.dev_nbint, atomIndex.size()*atomIndex.size() * sizeof(NonBondedInteraction*)));
		ddata.dev_nbint_host = (NonBondedInteraction**)malloc(atomIndex.size()*atomIndex.size() * sizeof(NonBondedInteraction*));
	}
	unsigned int a0 = atomTypeString2Int(atom0);
	unsigned int a1 = atomTypeString2Int(atom1);
	hdata.nbint[a0][a1] = nbinter;
	hdata.nbint[a1][a0] = nbinter;
	ddata.dev_nbint_host[a0*atomIndex.size() + a1] = dev_nbinter;
	ddata.dev_nbint_host[a1*atomIndex.size() + a0] = dev_nbinter;
	std::cout << atom0 << "--" << atom1 << '\t' << hdata.nbint[a0][a1]->getInteractionName() << "parm:	";
	hdata.nbint[a0][a1]->printParam();
	nbint_idx++;
}

__global__ void shiftToBoxKernel(unsigned int natom, Scalar* dev_rx, Scalar* dev_ry, Scalar* dev_rz, RectangularBox* dev_rect) {
	Scalar L = dev_rect->Length;
	Scalar T = dev_rect->Width;
	Scalar H = dev_rect->Height;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	while (idx < natom) {
		Scalar rx = dev_rx[idx];
		Scalar ry = dev_ry[idx];
		Scalar rz = dev_rz[idx];
		if (rx > L) {
			rx -= L;
		}
		if (rx < 0) {
			rx += L;
		}
		if (ry > T) {
			ry -= T;
		}
		if (ry < 0) {
			ry += T;
		}
		if (rz > H) {
			rz -= H;
		}
		if (rz < 0) {
			rz += H;
		}
		dev_rx[idx] = rx;
		dev_ry[idx] = ry;
		dev_rz[idx] = rz;
		idx += blockDim.x*gridDim.x;
	}
}
void MDData::shiftParticlesInBox() {
	dim3 grid(blocksPerGrid, 1);
	dim3 block(threadsPerBlock, 1);
	shiftToBoxKernel<<<grid,block>>>(hdata.Natom, ddata.dev_rx, ddata.dev_ry, ddata.dev_rz, ddata.dev_boxRect);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
}

void MDData::writeXYZTraj(char* path, Scalar t) {
	if (path == NULL) {
		std::cout << "Output file path error" << std::endl; exit(0);
	}
	FILE* fp = fopen(path, "a");
	if (fp == NULL) {
		std::cout << "Traj file write fail " << std::endl;
		exit(0);
	}
	fprintf(fp, "%d\n", hdata.Natom);
	fprintf(fp, "%lf\n", t);
	for (int i = 0; i < hdata.Natom; i++) {
		fprintf(fp, "%c\t%lf\t%lf\t%lf\n", 'C', hdata.rx[i], hdata.ry[i], hdata.rz[i]);
	}
	fclose(fp);
}