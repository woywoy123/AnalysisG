#ifndef H_CONUIX_SOLVERS
#define H_CONUIX_SOLVERS

#define PI 3.14159265358979323846
#include <complex>
#include <cmath>

template <typename g>
int linear(g a, g b, std::complex<g>* v0, double tol = 1e-14){
    if (a < tol){return 0;}
    *v0 = std::complex<g>(-b/a);
    return 1; 
}

template <typename g>
int quadratic(g a, g b, g c, std::complex<g>* v0, std::complex<g>* v1, double tol = 1e-14){
    if (std::fabs(a) < tol){return linear(b, c, v1, tol);}
    std::complex<g> dsc = std::sqrt(std::complex<g>(b*b - 4 * a * c)); 
    const g hlf = 1.0 / (2.0 * a);
    (*v0) = (-std::complex<g>(b) + dsc) * hlf; 
    (*v1) = (-std::complex<g>(b) - dsc) * hlf; 
    return 2;
}; 

template <typename g>
int cubic(
        g a, g b, g c, g d,
        std::complex<g>* v0, std::complex<g>* v1, std::complex<g>* v2,
        double tol = 1e-14
){
    if (std::fabs(a) < tol){return quadratic(b, c, d, v0, v1);}
    const g sthrd = std::sqrt(g(3.0)); 
    const g third = 1.0 / 3.0; 
    const g hlf = 0.5; 
    const g cnx = third * third * third; 
   
    const g a_ = 1.0 / a;  
    const g b_ = b * a_; 
    const g c_ = c * a_; 
    const g d_ = d * a_;

    const g p = c_ - b_ * b_ * third;
    const g q = d_ + (2.0 * b_ * b_ * b_ - 9.0 * b_ * c_) * cnx;
    
    const g delta = (q * q) * 0.25 + (p * p * p) * cnx;
    if (delta > 0){
        g disc = std::sqrt(delta);
        g u = std::cbrt(-q * hlf + disc);
        g v = std::cbrt(-q * hlf - disc);
        
        *v0 = std::complex<g>(u + v - b_ * third, 0);
        *v1 = std::complex<g>(-(u + v) * hlf - b_ * third,   (sthrd * (u - v) * hlf));
        *v2 = std::complex<g>(-(u + v) * hlf - b_ * third, - (sthrd * (u - v) * hlf));
        return 3; 
    }
    if (delta < 0){
        const g r = std::sqrt(-p * p * p * cnx);
        const g theta = acos( - hlf * q / r );
        *v0 = std::complex<g>(2.0 * std::cbrt(r) * cos(theta * third) - b_ * third, 0);
        *v1 = std::complex<g>(2.0 * std::cbrt(r) * cos((theta + 2.0 * PI) * third) - b_ * third, 0);
        *v2 = std::complex<g>(2.0 * std::cbrt(r) * cos((theta + 4.0 * PI) * third) - b_ * third, 0);
        return 3;
    }

    g u = std::cbrt(-q * hlf);
    if (std::fabs(u) < tol) {
        *v0 = -b_ * third;
        *v1 = -b_ * third;
        *v2 = -b_ * third;
        return 3; 
    }

    *v0 = 2.0 * u - b_ * third;
    *v1 = -u - b_ * third;
    *v2 = -u - b_ * third;
    return 3; 
}


#endif



