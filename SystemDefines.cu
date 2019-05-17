#include "SystemDefines.cuh"

void handleError(cudaError_t cu) {
	if (cu != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(cu));
		system("pause");
		exit(0);
	}
}

Scalar RectangularBox::Volume() {
	return Length*Width*Height;
}

__device__ __host__ Scalar Scalar3::norm() {
	return sqrt(x*x + y*y + z*z);
}

__device__ __host__  Scalar Scalar3::norm2() {
	return x*x + y*y + z*z;
}