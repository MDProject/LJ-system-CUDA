#ifndef FILEMANAGER
#define FILEMANAGER

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "SystemDefines.cuh"

void checkFilePointer(FILE* s, std::string msg);
int NumOfAtomGro(char* path);
void ReadGroToHost(char* path, Scalar* velx, Scalar* vely, Scalar* velz, Scalar* coorx, Scalar* coory, Scalar* coorz, std::string* atomType, int* natom, RectangularBox& box);
void WriteGroToHost(char* path, Scalar* velx, Scalar* vely, Scalar* velz, Scalar* coorx, Scalar* coory, Scalar* coorz, std::string* atomType, int natom, RectangularBox box);
void WriteXYZTraj(char* path, Scalar time, Scalar*coorx, Scalar* coory, Scalar* coorz);

#endif 