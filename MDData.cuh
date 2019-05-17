#ifndef MDDATA
#define MDDATA

#include <string>
#include <stdlib.h>
#include "SystemDefines.cuh"
#include "NeighborList.cuh"
#include "NonBondedInteraction.cuh"
#include "FileManager.cuh"

class NonBondedInteraction;
class DeviceData {
public:
	// current time step value
	Scalar* dev_vx, *dev_vy, *dev_vz;
	Scalar* dev_rx, *dev_ry, *dev_rz;
	Scalar* dev_fx, *dev_fy, *dev_fz;
	// Constant info 
	Scalar* dev_mass;
	int* dev_natom;
	int* dev_atomType; // the element value corresponds to the atomIndex vector table's index ( 0 -- atomType atomIndex[0], 1 -- atomType atomIndex[1] so on )
	RectangularBox* dev_boxRect;
	unsigned int typeNum;
	NonBondedInteraction** dev_nbint_host;
	NonBondedInteraction** dev_nbint; //pointer of a specific nonbonded interaction class GPU
};

class HostData {
public:
	Scalar* vx, *vy, *vz;
	Scalar* rx, *ry, *rz;
	Scalar* fx, *fy, *fz;
	Scalar* mass;
	int Natom;
	std::string* atomType;
	int* atomType_int;
	RectangularBox boxRect;
	NonBondedInteraction*** nbint; // nbint[i][j] = nbint[j][i]:	pointer of a specific nonbonded interaction class CPU
};

class NonBondedNeighborList;
class MDData {
private:
	void AllocateCPUMemory(int N); // N is atoms number
	void AllocateGPUMemory(int N);
	void FreeCPUMemory();
	void FreeGPUMemory();
	int atomTypeString2Int(std::string str);
	unsigned int nbint_idx = 0;
public:
	DeviceData ddata;
	HostData hdata;
	std::vector<std::string> atomIndex; // atoms kinds
	NonBondedNeighborList* nblist; // pointer of specific neighbor list 
public:
	MDData() :nbint_idx(0) {}
	void registerNeighborListMethod(NonBondedNeighborList* nbl);
	void registerNonBondedInteraction(NonBondedInteraction* nbinter, NonBondedInteraction* dev_nbinter, std::string atom0, std::string atom1);
	void writeGroToHost(char* path);
	void readGroToHost(char* path); // Initial HostData and set default mass = 1 and force = 0
	void copyMDDataHostToDevice();
	void copyMDDataDeviceToHost();
	void shiftParticlesInBox();
	void writeXYZTraj(char* path, Scalar t);
};

#endif