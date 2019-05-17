#include "MDData.cuh"
#include "Thermostat.cuh"
#include "Integrator.cuh"
#include "Statistics.cuh"

int main() {
	char path[] = "D:\\WinSCP\\LJ_fluid\\BiMolecule_20.gro";
	char pathXYZ[]= "D:\\WinSCP\\LJ_fluid\\BiMolecule.xyz";
	char pathframe[] = "D:\\WinSCP\\LJ_fluid\\frame_test.txt";

	/* Basic MD Settings */
	Scalar dt = 0.002;
	unsigned int outFreq = 1; // outPut *.xyz file every outFreq step
	
	MDData mddata;
	mddata.readGroToHost(path);
	NonBondedInteraction lj1(1., 1., 1., 2.5), *dev_lj1;
	//NonBondedInteraction lj2(1., 1., 1., 2.5), *dev_lj2;
	//NonBondedInteraction lj3(1., 1., 1., 2.5), *dev_lj3;
	handleError(cudaMalloc(&dev_lj1, sizeof(NonBondedInteraction)));
	//handleError(cudaMalloc(&dev_lj2, sizeof(NonBondedInteraction)));
	//handleError(cudaMalloc(&dev_lj3, sizeof(NonBondedInteraction)));
	handleError(cudaMemcpy(dev_lj1, &lj1, sizeof(NonBondedInteraction), cudaMemcpyHostToDevice));
	//handleError(cudaMemcpy(dev_lj2, &lj2, sizeof(NonBondedInteraction), cudaMemcpyHostToDevice));
	//handleError(cudaMemcpy(dev_lj3, &lj3, sizeof(NonBondedInteraction), cudaMemcpyHostToDevice));
	mddata.registerNonBondedInteraction(&lj1, dev_lj1, "tp03", "tp03"); // register on host and device at same time...
	//mddata.registerNonBondedInteraction(&lj2, dev_lj2, "tp03", "tp04");
	//mddata.registerNonBondedInteraction(&lj3, dev_lj3, "tp04", "tp04");
	mddata.copyMDDataHostToDevice();
	
	VerletList vlist(mddata, 4);
	vlist.makeNewList(mddata);
	mddata.registerNeighborListMethod(&vlist);
	// set integrator solver
	
	VelocityVerlet int_vv;
	int_vv.addMDData(&mddata);

	// set thermostat (no thermostat, NVE test)

	/*NOThermostat noThermo; // NVE Ensemble, no thermostat coupling
	noThermo.setIntegrator(&int_vv);
	noThermo.init(&vlist);*/

	NoseHooverChains2 nhc2(2.8, 0.1); // ref_T and tau_T
	nhc2.setIntegrator(&int_vv);
	nhc2.Init();

	Statistics stat(mddata, &vlist);
	for (int i = 1; i < 2000; i++) {
		if (vlist.ifRebuild(mddata)) {
			std::cout << "VerletList rebuild at " << i << std::endl;
			vlist.updateList(mddata);
		}
		nhc2.Operator_L(dt, &vlist);
		mddata.shiftParticlesInBox();
		if (i % outFreq == 0) {
			mddata.copyMDDataDeviceToHost();
			//std::cout << stat.kineticEnergy() << std::endl;
			mddata.writeXYZTraj(pathXYZ, i*dt);
		}
	}
	//mddata.copyMDDataDeviceToHost();
	system("pause");
}

