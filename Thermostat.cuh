#ifndef THERMOSTAT
#define THERMOSTAT

#include "MDData.cuh"
#include "Integrator.cuh"

class MDData;
class Thermostat {
protected:
	Integrator* integrator;
public:
	virtual void thermoTranslationRemove() = 0;
	virtual void setIntegrator(Integrator* inte) = 0;
	virtual void Operator_L(Scalar dt, VerletList* vlist) = 0;
};

class NoseHooverChains2 :public Thermostat {
private:
	Scalar vxi1, vxi2, rxi1, rxi2;
	Scalar ref_T, tau_T;
	Scalar Q1, Q2;
	Scalar* hd_uk;	// GPU Memory for storing the kinetic energy sum per block
	Scalar uk; // who uses it who updates it
	int NOF;	// number of freedoms
	void Operator_LG2(Scalar dt);
	void Operator_Lvxi1(Scalar dt);
	void Operator_LG1(Scalar dt);
	void Operator_LCv(Scalar dt);
	void Operator_Lxi(Scalar dt);
	void update_uk();  // update current kinetic energy
public:
	NoseHooverChains2(Scalar refT,Scalar tauT); 
	void thermoTranslationRemove();
	void Operator_L(Scalar dt, VerletList* vlist);
	void Init(); // hd_uk
	void setIntegrator(Integrator* inte);
};

class NOThermostat :public Thermostat {
public:
	void thermoTranslationRemove();
	void init(VerletList* vlist); // prepare initial force
	void setIntegrator(Integrator* inte);
	void Operator_L(Scalar dt, VerletList* vlist);
};

#endif