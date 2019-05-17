#include "Thermostat.cuh"

__global__ void update_uk_kernel(unsigned int natom, Scalar* dev_vx, Scalar* dev_vy, Scalar* dev_vz, Scalar* sumUK) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ Scalar sumCache[threadsPerBlock];
	Scalar sumTmp = 0.;
	while (idx < natom) {
		Scalar vx = dev_vx[idx];
		Scalar vy = dev_vy[idx];
		Scalar vz = dev_vz[idx];
		sumTmp += vx*vx + vy*vy + vz*vz;
		idx += blockDim.x*gridDim.x;
	}
	sumCache[threadIdx.x] = sumTmp;
	__syncthreads();
	unsigned int i = blockDim.x / 2;
	while (i > 0) {
		if (threadIdx.x < i) {
			sumCache[threadIdx.x] += sumCache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0) {
		sumUK[blockIdx.x] = sumCache[0];
	}
}
void NoseHooverChains2::update_uk() {
	unsigned int Natom = integrator->mddata->hdata.Natom;
	dim3 grid(blocksPerGrid, 1);
	dim3 block(threadsPerBlock, 1);
	update_uk_kernel << <grid, block >> > (Natom, integrator->mddata->ddata.dev_vx, integrator->mddata->ddata.dev_vy, integrator->mddata->ddata.dev_vz, hd_uk);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
	uk = 0.;
	for (int i = 0; i < blocksPerGrid; i++) {
		uk += hd_uk[i];
	}
	uk /= 2.;
	//std::cout << uk << " ";
}

// Nose Hoover Chains: length = 2		NVT Ensemble
NoseHooverChains2::NoseHooverChains2(Scalar refT,Scalar tauT): vxi1(0.), rxi1(1.), vxi2(0.), rxi2(1.) {
	ref_T = refT;
	tau_T = tauT;
}

void NoseHooverChains2::Init() {
	unsigned int natom = integrator->mddata->hdata.Natom;
	cudaMallocManaged(&hd_uk, blocksPerGrid * sizeof(Scalar));
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
	NOF = natom * 3;
	Q1 = tau_T*tau_T*ref_T / (4.*PI*PI)*NOF;
	Q2 = Q1 / NOF;
	std::cout << "Thermosat choice:\t NoseHoover Chain with length = 2 \t\tParam:	\tQ1 " << Q1 << "\tQ2 " << Q2 << "\tReference temperature: " << ref_T << "\tRelaxation time: " << tau_T << std::endl;
	std::cout << "Integrator choice:\t" << integrator->getName() << std::endl;
}

void NoseHooverChains2::Operator_L(Scalar dt, VerletList* vlist) {
	// L_NHC
	Operator_LG2(dt * 0.25);

	Operator_Lvxi1(dt * 0.125);
	Operator_LG1(dt * 0.25);
	Operator_Lvxi1(dt * 0.125);

	Operator_LCv(dt * 0.5);
	Operator_Lxi(dt * 0.5);

	Operator_Lvxi1(dt * 0.125);
	Operator_LG1(dt * 0.25);
	Operator_Lvxi1(dt * 0.125);

	Operator_LG2(dt * 0.25);
	// L_r + L_v evolve real physical system 

	integrator->Operator_L(dt, vlist);

	// L_NHC
	Operator_LG2(dt * 0.25);

	Operator_Lvxi1(dt * 0.125);
	Operator_LG1(dt * 0.25);
	Operator_Lvxi1(dt * 0.125);

	Operator_LCv(dt * 0.5);
	Operator_Lxi(dt * 0.5);

	Operator_Lvxi1(dt * 0.125);
	Operator_LG1(dt * 0.25);
	Operator_Lvxi1(dt * 0.125);

	Operator_LG2(dt * 0.25);
}

void NoseHooverChains2::setIntegrator(Integrator* inte) {
	integrator = inte;
}

__global__ void operator_LCv_kernel(unsigned int natom, Scalar* dev_vx, Scalar* dev_vy, Scalar* dev_vz, Scalar factor) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	while (idx < natom) {
		if (idx > natom) { return; }
		dev_vx[idx] = dev_vx[idx] * factor;
		dev_vy[idx] = dev_vy[idx] * factor;
		dev_vz[idx] = dev_vz[idx] * factor;
		idx += blockDim.x*gridDim.x;
	}
}
void NoseHooverChains2::Operator_LCv(Scalar dt) {
	Scalar f = exp(-vxi1*dt);
	unsigned int Natom = integrator->mddata->hdata.Natom;
	dim3 grid(blocksPerGrid, 1);
	dim3 block(threadsPerBlock, 1);
	operator_LCv_kernel<<<grid,block>>>(Natom, integrator->mddata->ddata.dev_vx, integrator->mddata->ddata.dev_vy, integrator->mddata->ddata.dev_vz, f);
	cudaError_t result = cudaDeviceSynchronize();
	handleError(result);
}
void NoseHooverChains2::Operator_LG1(Scalar dt) {
	update_uk();
	Scalar G1 = (2 * uk - NOF*ref_T) / Q1;
	vxi1 += G1*dt;
}
void NoseHooverChains2::Operator_Lxi(Scalar dt) {
	rxi1 += vxi1 * dt;
	rxi2 += vxi2 * dt;
}
void NoseHooverChains2::Operator_LG2(Scalar dt) {
	Scalar G2 = (Q1 * vxi1 * vxi1 - ref_T) / Q2;
	vxi2 += G2 * dt;
}
void NoseHooverChains2::Operator_Lvxi1(Scalar dt) {
	vxi1 = vxi1*exp(-vxi2*dt);
}

void NoseHooverChains2::thermoTranslationRemove() {

}



// No Thermostat: NVE Ensemble
void NOThermostat::setIntegrator(Integrator* inte) {
	integrator = inte;
}

void NOThermostat::thermoTranslationRemove() {
	Scalar* dev_vx = integrator->mddata->ddata.dev_vx;
	Scalar* dev_vy = integrator->mddata->ddata.dev_vy;
	Scalar* dev_vz = integrator->mddata->ddata.dev_vz;
}

void NOThermostat::Operator_L(Scalar dt, VerletList* vlist) {
	integrator->Operator_L(dt, vlist);
}

void NOThermostat::init(VerletList* vlist) {
	integrator->UpdateForce(vlist);
}
