#include "NonBondedInteraction.cuh"

NonBondedInteraction::NonBondedInteraction(Scalar e, Scalar d, Scalar s, Scalar r) :nbname("LJ Potential"){
	epsilon = e;
	delta = d;
	sigma = s;
	rc = r;
}

void NonBondedInteraction::printParam() {
	std::cout << epsilon << ' ' << delta << ' ' << sigma << ' ' << rc << std::endl;
}

// F = 4*epsilon*[12*(sigma/r)^11*(-sigma/r^2)-6*delta*(sigma/r)^5(-sigma/r^2)]*(x/r)
//   = 24*epsilon*[-2*(sigma/r)^12+delta*(sigma/r)^6]*(x/r^2)
class NonBondedInteraction;
__device__ __host__ Scalar3 NonBondedInteraction::nbforce(Scalar3 r) {
	Scalar r2 = r.norm2();
	if (r2 < rc*rc) {
		Scalar sr2i = sigma*sigma / r2;
		Scalar sr6i = sr2i*sr2i*sr2i;
		Scalar sr12i = sr6i*sr6i;
		Scalar factor = 24.*epsilon*(-2.*sr12i + delta*sr6i);
		Scalar fx = factor*r.x / r2;
		Scalar fy = factor*r.y / r2;
		Scalar fz = factor*r.z / r2;
		return Scalar3(fx, fy, fz);
	}
	else {
		return Scalar3(0., 0., 0.);
	}
}

__device__ __host__ Scalar NonBondedInteraction::nbPotential(Scalar3 r) {
	Scalar pot = 0.;
	Scalar dr2 = r.norm2();
	if (dr2 < rc*rc) {
		Scalar sr2i = sigma*sigma / dr2;
		Scalar sr6i = sr2i*sr2i*sr2i;
		Scalar sr12i = sr6i*sr6i;
		pot += 4. * epsilon * (sr12i - sr6i);
	}
	return pot;
}
