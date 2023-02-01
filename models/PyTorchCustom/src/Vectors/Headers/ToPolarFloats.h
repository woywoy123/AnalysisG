#ifndef H_VECTOR_TOPOLAR_F
#define H_VECTOR_TOPOLAR_F

#include<iostream>
#include<torch/extension.h>

namespace VectorFloats
{
	double PT(double px, double py);
	double Phi(double px, double py);	
	double Eta(double px, double py, double pz); 
	double _Eta(double pt, double pz); 
	std::vector<double> PtEtaPhi(double px, double py, double pz);
}
#endif
