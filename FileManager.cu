// stringstream: first come first out
// operator>> returns a reference to the stream, as you said. Scalarhen, in the context of the if the stream is converted to a bool through the conversion operator 

#include "FileManager.cuh"

void checkFilePointer(FILE* s, std::string msg) {
	if (s == NULL) {
		std::cout << "File pointer is NULL when " << msg << std::endl;
		system("pause"); exit(0);
	}
}

int NumOfAtomGro(char* path) {
	char strLine[255];
	FILE* fp = fopen(path, "r");
	checkFilePointer(fp, "read *.gro file");
	fgets(strLine, 255, fp);
	fgets(strLine, 255, fp);
	fclose(fp);
	return atoi(strLine);
}

// read *.gro file to RAM
void ReadGroToHost(char* path, Scalar* velx, Scalar* vely, Scalar* velz, Scalar* coorx, Scalar* coory, Scalar* coorz, std::string* atomType, int* natom, RectangularBox& box) {
	char strLine[255];
	FILE* fp = fopen(path, "r");
	checkFilePointer(fp, "read *.gro file");
	fgets(strLine, 255, fp);
	fgets(strLine, 255, fp);
	*natom = atoi(strLine);
	std::vector<std::string> vecstr;
	std::string strTmp;
	for (int i = 0; i < (*natom); i++) {
		fgets(strLine, 255, fp);
		std::string stringLine(strLine);
		std::stringstream ss(stringLine);
		while (ss >> strTmp) {
			vecstr.push_back(strTmp);
		}
		atomType[i] = vecstr[1];
		coorx[i] = (Scalar)stod(vecstr[3]);
		coory[i] = (Scalar)stod(vecstr[4]);
		coorz[i] = (Scalar)stod(vecstr[5]);
		velx[i] = (Scalar)stod(vecstr[6]);
		vely[i] = (Scalar)stod(vecstr[7]);
		velz[i] = (Scalar)stod(vecstr[8]);
		vecstr.clear();
	}
	fgets(strLine, 255, fp);
	std::string stringLine(strLine);
	std::stringstream ss(stringLine);
	ss >> strTmp;
	box.Length = (Scalar)stod(strTmp);
	ss >> strTmp;
	box.Width = (Scalar)stod(strTmp);
	ss >> strTmp;
	box.Height = (Scalar)stod(strTmp);
	fclose(fp);
}

// write a frame to RAM file *.gro
void WriteGroToHost(char* path, Scalar* velx, Scalar* vely, Scalar* velz, Scalar* coorx, Scalar* coory, Scalar* coorz, std::string* atomType, int natom, RectangularBox box) {
	FILE* fp = fopen(path, "w+");
	checkFilePointer(fp, "write *.gro file");
	fprintf(fp, "%s\n", "Write by MDCUDA");
	fprintf(fp, "%d\n", natom);
	for (int i = 0; i < natom; i++) {
		fprintf(fp, "%5d%-5s%5s%5d% 8.3f%8.3f%8.3f% 8.4f% 8.4f% 8.4f\n", i + 1, atomType[i].c_str(), atomType[i].c_str(), i + 1, coorx[i], coory[i], coorz[i], velx[i], vely[i], velz[i]);
	}
	fprintf(fp, "%f\t%f\t%f\n", box.Length, box.Width, box.Height);
	fclose(fp);
}

// write *.xyz file 
void WriteXYZTraj(char* path, Scalar time, Scalar*coorx, Scalar* coory, Scalar* coorz) {

}


