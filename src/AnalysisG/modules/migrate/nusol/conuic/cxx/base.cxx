#include <templates/particle_template.h>
#include <conuic/variables.h>
#include <common/matrix.h>
#include <string>

long double line_sy(long double sx, long double delta){return sx / delta;}
long double line_sx(long double sy, long double delta){return sy * delta;}

long double to_sx(int sign, long double x1, long double y1, kinematic_c* data){
    return - std::pow(1 / data -> b_mu, 2) * (data -> w.pair(sign) * y1 + (1 - std::pow(data -> b_mu, 2))*x1); 
}

long double to_sy(int sign, long double x1, long double y1, kinematic_c* data){
    long double w = data -> w.pair(sign); 
    return - std::pow(1 / data -> b_mu, 2) * ( w * x1 + (w * w - std::pow(data -> b_mu, 2))*y1); 
}

long double to_x1(int sign, long double sx, long double sy, kinematic_c* data){
    long double w = data -> w.pair(sign); 
    return sx - (sx + w * sy) / std::pow(data -> O.pair(sign), 2); 
}

long double to_y1(int sign, long double sx, long double sy, kinematic_c* data){
    long double w = data -> w.pair(sign); 
    return sx - (sx + w * sy) * w / std::pow(data -> O.pair(sign), 2); 
}

std::complex<long double> mW(long double sx, long double m_nu, kinematic_c* data){
    return std::sqrt(std::complex<long double>(m_nu * m_nu - std::pow(data -> m_mu, 2) - 2 * data -> p_mu * sx));
}

std::complex<long double> mT(long double sx, long double sy, long double m_nu, kinematic_c* data){
    long double a = std::pow(data -> m_b, 2) - std::pow(data -> m_mu, 2) + m_nu * m_nu;
    long double b = - 2 * (sx * (data -> p_mu + data -> p_b * data -> theta.cos) + data -> p_b * data -> theta.sin * sy); 
    return std::sqrt(std::complex<long double>(a + b)); 
}

std::complex<long double> mN(int sign, G2_t* data){
    long double p = data -> p_mu * data -> delta.pair(sign) * data -> O.pair(sign); 
    long double a = (data -> b2_mu - std::pow(data -> w.pair(sign), 2)) * std::pow(data -> delta.pair(sign), 2); 
    long double b = 2 * data -> w.pair(sign) * data -> delta.pair(sign) - (1 - data -> b2_mu);
    return std::sqrt(std::complex(data -> m2_mu - (p * p) / (a + b))); 
}

long double dG2(long double sx, long double sy, G2_t* data){
    return -data -> G.p * data -> G.m * (sx - data -> delta.p * sy)*(sx - data -> delta.m * sy); 
}

long double mobius(long double sx, long double sy, G2_t* data){
    return (sx - data -> delta.p * sy) * (sx - data -> delta.m * sy); 
}
