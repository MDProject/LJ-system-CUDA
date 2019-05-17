#ifndef INTEGRATOR
#define INTEGRATOR

#include "MDData.cuh"

class Integrator {
public:
	void addMDData(MDData* mdd) { mddata = mdd; }
	MDData* mddata;
	virtual std::string getName() = 0;
	virtual void UpdateForce(VerletList* vlist) = 0;
	virtual void Operator_L(Scalar dt, VerletList* vlist) = 0;
};

class VelocityVerlet :public Integrator {
private:
	void Operator_Lv(Scalar dt);
	void Operator_LvLr(Scalar dt); // dt denotes \Delta t in frenkel
public:
	void UpdateForce(VerletList* vlist);
	void Operator_L(Scalar dt, VerletList* vlist);
	std::string getName() { return "Velocity Verlet Integrator"; }
};

#endif
