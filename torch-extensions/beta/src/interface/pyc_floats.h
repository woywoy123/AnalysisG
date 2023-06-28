#ifndef H_PYC_FLOATS_
#define H_PYC_FLOATS_

#include <transform/cartesian-floats/cartesian.h>
#include <transform/polar-floats/polar.h>

namespace pyc
{
    namespace transform 
    {
        namespace separate
        {
            double Pt(double px, double py);
            double Eta(double px, double py, double pz); 
            double Phi(double px, double py); 
            std::vector<double> PtEtaPhi(double px, double py, double pz); 
            std::vector<double> PtEtaPhiE(double px, double py, double pz, double e); 

            double Px(double pt, double phi);
            double Py(double pt, double phi); 
            double Pz(double pt, double eta); 
            std::vector<double> PxPyPz(double pt, double eta, double phi); 
            std::vector<double> PxPyPzE(double pt, double eta, double phi, double e); 
        }
        namespace combined
        {
            std::vector<double> Pt(std::vector<std::vector<double>> pmc);
            std::vector<double> Eta(std::vector<std::vector<double>> pmc); 
            std::vector<double> Phi(std::vector<std::vector<double>> pmc);
            std::vector<std::vector<double>> PtEtaPhi(std::vector<std::vector<double>> pmc); 
            std::vector<std::vector<double>> PtEtaPhiE(std::vector<std::vector<double>> pmc); 

            std::vector<double> Px(std::vector<std::vector<double>> pmu);
            std::vector<double> Py(std::vector<std::vector<double>> pmu); 
            std::vector<double> Pz(std::vector<std::vector<double>> pmu); 
            std::vector<std::vector<double>> PxPyPz(std::vector<std::vector<double>> pmu); 
            std::vector<std::vector<double>> PxPyPzE(std::vector<std::vector<double>> pmu); 
        }
    }
}

#endif
