#ifndef NonBondedInter
#define NonBondedInter

#include "MDData.cuh"

// General form:	Ulj = 4*epsilon*[(sigma/r)^12-delta*(sigma/r)6]
class NonBondedInteraction {
protected:
	std::string nbname;
	Scalar epsilon, delta, sigma, rc;
public:
	void printParam();
	std::string getInteractionName() { return nbname; }
	NonBondedInteraction(Scalar e, Scalar d, Scalar s, Scalar r);
	void setLJCutOff(Scalar r) { rc = r; }
	Scalar getCutOff() { return rc; }
	__device__ __host__ Scalar3 nbforce(Scalar3 r);
	__device__ __host__ Scalar nbPotential(Scalar3 r);
};


#endif
