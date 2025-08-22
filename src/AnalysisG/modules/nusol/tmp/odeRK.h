#ifndef MULTISOL_RK4_H
#define MULTISOL_RK4_H

#include <reconstruction/multisol.h>
#include <reconstruction/matrix.h>
#include <tools/tools.h>

struct ellipse_t {
    vec3   A,   B,   C;
    vec3  vA,  vB,  vC; 
    double t;
    double z = 1.0; 
    void print(); 

};


struct recon_t {
    bool is_valid = false;
    double residual = -1;
    std::vector<double> t;
    std::vector<double> z;
    std::vector<double> phi;
};


ellipse_t operator+(const ellipse_t& a, const ellipse_t& b); 
ellipse_t operator*(const ellipse_t& a, double s); 

class odeRK : public tools
{
    
    public:
        odeRK(std::vector<multisol*>* sols, vec3 met, int iter, double step);
        ~odeRK(); 
        
        void solve(); 
        void rk4(double dt);

        void   update_t(); 
        double solve_z_phi(); 
        double residual(std::vector<double> wg, std::vector<double> phx); 
        std::vector<ellipse_t> derivative(const std::vector<ellipse_t>& dS); 

        double ghost_angle(int nui); 
        std::vector<double> plane_rk4(const std::vector<double>& t_initial);
        std::vector<double> plane_align(const std::vector<ellipse_t>& current_state);

    private:
        vec3 met_; 

        int itr    = 0; 
        int nsx    = 0; 
        double dt_ = 0; 
        double ct  = 0; 

        std::vector<ellipse_t>  _state = {}; 
        std::vector<multisol*>* _data  = nullptr; 


}; 



#endif
