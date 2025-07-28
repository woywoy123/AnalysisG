#ifndef NUSOL_H
#define NUSOL_H

#include "linalg.h"
#include <utility>
#include <vector>

class Boundary; 

class NuSol 
{
    private:
        friend Boundary; 
        const Vec4& b_jet;
        const Vec4& lepton;
        double mT, mW;
        double Z2, Z;
        double x0p, x0, Sx, Sy, w, om2, eps2, x1, y1;
        double Bm, Bb;
        double cos_th, sin_th;

        Mat3 calculate_R_T();
        void calculate_kinematics();
        Boundary* bx = nullptr; 
    
    public:
        ~NuSol(); 
        NuSol(const Vec4& b, const Vec4& lep, double mass_top, double mass_w);
        std::pair<bool, std::pair<Mat3, Mat3>> getH_derivatives();
        std::vector<std::vector<double>> get_boundary();

        double getZ2() const { return Z2; }
        std::pair<bool, Vec2> getZ2Derivatives() const;
        std::pair<bool, Mat3> getH();
};

class Boundary 
{
    public: 
        Boundary(const NuSol* nu); 

        double Z2(double mW, double lambdaW, double mT, double lambdaT);
        void  DZ2(double mW, double lambdaW, double mT, double lambdaT);
        void D2Z2();

        void update(double mW, double lambdaW, double mT, double lambdaT); 
        void analytical_boundary(double mW, double lambdaW, double mT, double lambdaT); 

        double phi(); 
        std::pair<double, double> Center(); 
        std::pair<double, double> Axes(); 
        void bounds(std::vector<std::vector<double>>* inpt, int res = 10);
        ~Boundary(); 

    private:
        const NuSol* nu = nullptr; 
        double z2_origin = 0; 
        bool computed = false;

        double hessian[3]; 
        double jacobian[2]; 
        double boundary[3][3]; 
};


#endif
