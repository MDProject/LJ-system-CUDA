#ifndef DEFINE_HEADER
#define DEFINE_HEADER

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <vector>

#define PI 3.14159265359
#define threadsPerBlock 64 // must be power of 2
#define blocksPerGrid 16

typedef double Scalar;

void handleError(cudaError_t cu);

class Scalar3 {
public:
	Scalar x, y, z;
	__device__ __host__ Scalar3(Scalar xx, Scalar yy, Scalar zz) {
		x = xx; y = yy; z = zz;
	}
	Scalar3():x(0.),y(0.),z(0.){}
	__device__ __host__ Scalar norm();
	__device__ __host__ Scalar norm2();
};


struct RectangularBox { // All coordinates start from 0 
public:
	Scalar Length, Width, Height; // x y and z
	Scalar Volume();
};

#endif