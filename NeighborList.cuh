#ifndef NeighborList
#define NeighborList

#include "MDData.cuh"

class MDData;
class NonBondedNeighborList {
protected:
	RectangularBox* dev_rect;
	int MaxNeighborNum;
public:
	__device__ __host__ int getMaxNeigh() { return MaxNeighborNum; }
	int* dev_verletlist; // Saved on GPU only for saving time
	int* dev_nlist;
	int* host_verletlist; // Saved on CPU form check
	int* host_nlist;
	virtual bool ifRebuild(MDData& mddata) = 0;
	__device__ __host__ virtual int vlist(int i, int j) = 0;
	__device__ __host__ virtual int nlist(int i) = 0;
	virtual void makeNewList(MDData& mddata) = 0;
	virtual void updateList(MDData& mddata) = 0;
};

class VerletList :public NonBondedNeighborList {
private:
	Scalar rc; // max cutoff radius
	Scalar rv; // verlet radius
	Scalar scaleFactor;
	void AllocateCPUMemory(unsigned int N);
	void AllocateGPUMemory(unsigned int N);
	void FreeCPUMemory(unsigned int N);
	void FreeGPUMemory(unsigned int N);
public:
	bool* hd_build;
	Scalar* dev_coorRecordx, *dev_coorRecordy, *dev_coorRecordz; // particle's position when verlet list is made; stored on GPU
	Scalar* coorRecordx, *coorRecordy, *coorRecordz;
	VerletList(MDData mddata, Scalar d2); // record the atom positions when neighbor list is built; rc = d1 and rv = d2 so d2 > d1
	__device__ __host__ int vlist(int i, int j);
	__device__ __host__ int nlist(int i);
	void updateList(MDData& mddata);
	void makeNewList(MDData& mddata);
	bool ifRebuild(MDData& mddata);
};

#endif