#ifndef STATISTIC
#define STATISTIC

#include "MDData.cuh"

class Statistics {
private:
	MDData mddata;
	NonBondedNeighborList* nblist;
public:
	Statistics(MDData mddata);
	Statistics(MDData mddata, NonBondedNeighborList* nblist);
	Scalar kineticEnergy();
	Scalar3 COM_velocity();
	Scalar3 COM_force();
	Scalar interPotential();
	Scalar interPotentialNB();
};



#endif 