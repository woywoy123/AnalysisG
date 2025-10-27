#include <iostream>
#include <complex>
#include <iomanip>
#include <cmath>
#include <string>

#define PI 3.14159265358979323846

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
    const g half = 1.0 / (2.0 * a);
    (*v0) = (-std::complex<g>(b) + dsc) * half; 
    (*v1) = (-std::complex<g>(b) - dsc) * half; 
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
    const g half = 0.5; 
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
        g u = std::cbrt(-q * half + disc);
        g v = std::cbrt(-q * half - disc);
        
        *v0 = std::complex<g>(u + v - b_ * third, 0);
        *v1 = std::complex<g>(-(u + v) * half - b_ * third,   (sthrd * (u - v) * half));
        *v2 = std::complex<g>(-(u + v) * half - b_ * third, - (sthrd * (u - v) * half));
        return 3; 
    }
    if (delta < 0){
        const g r = std::sqrt(-p * p * p * cnx);
        const g theta = acos( - half * q / r );
        *v0 = std::complex<g>(2.0 * std::cbrt(r) * cos(theta * third) - b_ * third, 0);
        *v1 = std::complex<g>(2.0 * std::cbrt(r) * cos((theta + 2.0 * PI) * third) - b_ * third, 0);
        *v2 = std::complex<g>(2.0 * std::cbrt(r) * cos((theta + 4.0 * PI) * third) - b_ * third, 0);
        return 3;
    }

    g u = std::cbrt(-q * half);
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


template <typename g>
int quartic(
    g a, g b, g c, g d, g e, 
    std::complex<g>* v0, std::complex<g>* v1, 
    std::complex<g>* v2, std::complex<g>* v3,
    double tol = 1e-14
){
    if (std::fabs(a) < tol){return cubic(b, c, d, e, v0, v1, v2, tol);}
    
    // Normalize coefficients
    const g a_ = b / a;
    const g b_ = c / a;
    const g c_ = d / a;
    const g d_ = e / a;
    
    // Depressed quartic coefficients
    const g p = b_ - (3.0 * a_ * a_) / 8.0;
    const g q = c_ + (a_ * a_ * a_) / 8.0 - a_ * b_ / 2.0; 
    const g r = d_ - (3.0 * a_ * a_ * a_ * a_) / 256.0 + a_ * a_ * b_ / 16.0 - a_ * c_ / 4.0; 

    // Resolvent cubic coefficients (CORRECTED)
    const g r_a = 1.0;
    const g r_b = -2.0 * p;
    const g r_c = p * p - 4.0 * r;
    const g r_d = -q * q;

    std::complex<g> z0, z1, z2;
    cubic(r_a, r_b, r_c, r_d, &z0, &z1, &z2, tol);
    
    // Find the best real root for m^2 (CORRECTED selection logic)
    g m_sq;
    if (std::fabs(z0.imag()) <= tol && z0.real() >= -tol) {
        m_sq = z0.real();
    } else if (std::fabs(z1.imag()) <= tol && z1.real() >= -tol) {
        m_sq = z1.real();
    } else if (std::fabs(z2.imag()) <= tol && z2.real() >= -tol) {
        m_sq = z2.real();
    } else {
        m_sq = z0.real(); // fallback
    }
    
    if (m_sq < 0) m_sq = 0;
    
    std::complex<g> m = std::sqrt(std::complex<g>(m_sq));
    
    if (std::abs(m) < tol) {
        std::complex<g> root1, root2;
        int num_roots = quadratic(g(1.0), p, r, &root1, &root2, tol);
        
        std::complex<g> y1 = std::sqrt(root1);
        std::complex<g> y2 = -std::sqrt(root1);
        std::complex<g> y3 = std::sqrt(root2);
        std::complex<g> y4 = -std::sqrt(root2);
        
        g offset = -a_ * 0.25;
        *v0 = y1 + offset;
        *v1 = y2 + offset;
        *v2 = y3 + offset;
        *v3 = y4 + offset;
        return 4;
    }
    
    std::complex<g> n1 = (m_sq + p + q / m) * g(0.5);
    std::complex<g> n2 = (m_sq + p - q / m) * g(0.5);
    
    g offset = -a_ * 0.25;
    
    std::complex<g> temp1, temp2;
    quadratic(g(1.0), m.real(), n1.real(), &temp1, &temp2, tol);
    *v0 = temp1 + offset;
    *v1 = temp2 + offset;
    
    quadratic(g(1.0), -m.real(), n2.real(), &temp1, &temp2, tol);
    *v2 = temp1 + offset;
    *v3 = temp2 + offset;
    
    return 4;
}



template <typename g>
void print(std::complex<g>* v){
    if (!v){return;}
    std::cout << "ROOT: "; 
    if (v -> real() >= 0){ std::cout << " ";}
    std::cout << v -> real(); 
    
    std::cout << " i: "; 
    std::cout << v -> imag() << "\n";
}

template <typename g>
void printRoots(
        const std::string& title, 
        std::complex<g>* v0, std::complex<g>* v1, 
        std::complex<g>* v2, std::complex<g>* v3, 
        int r
){
    std::cout << "--- " << title << " ---\n";
    std::cout << std::fixed << std::setprecision(8);
    print(v0); 
    print(v1);
    print(v2);
    print(v3);
    std::cout << "rn: " << r << std::endl; 
    std::cout << std::endl;
}

template <typename g>
std::complex<g> testroots(
     g a, g b, g c, g d, g e, std::complex<g>* v0, int r
){
    const std::complex<g> x = (*v0); 

    if (r == 2){return a * x * x + b * x + c;} 
    if (r == 3){return a * x * x * x + b * x * x + c*x + d;} 
    if (r == 4){return a * x * x * x * x + b * x * x * x + c * x * x + d * x + e;} 
    return 0; 
}

int main() {
    const long double quad_coeffs[30][3] = {
        { 1.0, -1.0, -2.0 },
        { -2.0, 3.0, 0.0 },
        { 0.5, -45.5, -450.0 },
        { 3.0, -3.0, -2.25 },
        { -1.0, 10.0, -9.0 },
        { 0.1, -10.2, 20.0 },
        { 1.0, -10000.5, 14998.5 },
        { 10.0, 0.0, -1.0e-7 },
        { 4.0, 2.0004, 0.0002 },
        { 1.0, -3.000000000001, 2.000000000002 },
        { 1.0, -10000000002.0, 20000000000.0 },
        { 0.5, 5000000000.0, 2500000000.0 },
        { 1e12, 0.0, -0.1 },
        { 1.0, -1e-10, 0.0 },
        { 1e15, -3e15, 2e15 },
        { 1e-15, -1e-15, -2e-15 },
        { -1.0, 2.000000000001, -1.000000000001 },
        { 2.0, 0.0, -2.0e20 },
        { -1e5, 1e-5, 0.0 },
        { 1.0, -19998.0, 99980001.0 },
        { 5.1, -2.2, 0.7 },
        { -3.9, 1.0, 4.5 },
        { 0.1, -0.5, 0.02 },
        { 1.8, 3.4, -2.1 },
        { -4.0, -1.0, 5.3 },
        { 2.2, -4.0, 3.0 },
        { 0.4, 1.5, -0.6 },
        { -1.0, -2.5, 1.5 },
        { 2.8, 0.5, -1.2 },
        { -0.15, -0.8, 2.4 }
    };

    const long double cubic_coeffs[40][4] = {
        { 1.0, -1.0, -2.0, 0.0 },
        { -0.5, -4.75, 5.25, 6.75 },
        { 2.0, -202.0, -196.0, 400.0 },
        { 10.0, -10.0, -1.0e-7, 0.00000000001 },
        { 1.0, -2.5, 0.75, 1.125 },
        { -1.0, 27.0, -243.0, 729.0 },
        { 0.1, -999.0, -199.96, 199.98 },
        { 4.0, 1.0, -5.0, -2.5 },
        { 1.0, -4.5, 6.75, -3.375 },
        { 0.1, -30.0, 3000.0, -100000.0 },
        { 1.0, 9.5001, 4.5009, 0.00045 },
        { 1.0, -9992.0, -179998.0, 179982.0 },
        { 5.0, -15.0, 15.0, -5.0 },
        { 1.0, 0.0, 0.0, 0.0 },
        { 1.0, 3.0, 3.0, 1.0 },
        { 1.0, -1e10, -1.0, 1e10 },
        { 1.0, 1e10, 1.0e-11, -1.0e-30 },
        { 1.0, -10000000002.0, 20000000003.0, -10000000001.0 },
        { 1.0, 0.0, -1.0e20, 0.0 },
        { 1e15, -3e5, 3e-5, -1e-15 },
        { 1.0, -29997.0, 299940003.0, -99970002999.0 },
        { 1.0, -1e10, 0.0, 0.0 },
        { -1.0, 10000000002.0, -20000000001.0, 10000000000.0 },
        { 1.0, 10000000000.0, -1e20, -1e30 },
        { 1e-5, -3e5, 3e15, -1e25 },
        { 5.0, -5.0, -5e20, 5e20 },
        { 1.0, -3e-10, 3e-20, -1e-30 },
        { 1.0, -1e-10, -1.000000000001, 1e-10 },
        { 1.0, -10000.0, 9999.0, 0.0 },
        { -1.0, 0.0003, -3.0e-8, 1.0e-12 },
        { 0.25, 0.88, -1.7, 0.5 },
        { -1.3, -2.1, 0.9, 4.5 },
        { 3.5, 1.0, -0.2, -6.7 },
        { -0.95, -3.4, 2.1, 1.1 },
        { 1.6, 0.1, -5.3, -0.01 },
        { -2.2, -4.0, 3.0, 2.0 },
        { 0.4, 1.5, -0.6, -1.9 },
        { -1.0, -2.5, 1.5, 0.5 },
        { 2.8, 0.5, -1.2, -0.3 },
        { -0.15, 0.8, -2.4, 1.6 }
    };


    const long double quartic_coeffs[30][5] = {
        { 1.0, -2.5, -0.5, 3.0, 0.0 },
        { -0.5, 49.0, 489.0, -980.0, -9000.0 },
        { 2.0, -8.0, 12.0, -8.0, 2.0 },
        { 1.0, -1.0, -0.75, 0.0, 0.0 },
        { 1.0, -10000.5, -9986.0, 10996.0, 9999.0 },
        { 100.0, -100.0, -75.0, 0.0075, 0.000075 },
        { 1.0, -8.0, 24.0, -32.0, 16.0 },
        { 1.0, 17.0, 71.25, 137.25, 60.75 },
        { -1.0, 199.0, -9998.5, -10000.0, 5000.0 },
        { 1.0, -0.5+1e-10, -2.0-1.5e-10, 1.0+2e-10, -1e-10 },
        { 1.0, -2e10, 1.0e20+2.0, -2.0e20, 1.0e20 },
        { 1.0, 0.0, -1.0e20, 0.0, 1.0e-20 },
        { 1e15, -1e5, 1e-5, -1e-15, 0.0 },
        { 1e-10, -4.0, 6e10, -4e20, 1e30 },
        { 1.0, 0.0, -2.0e20, 0.0, 1.0e40 },
        { 1e18, -6e8, 11e-2, -6e-12, 0.0 },
        { 1.0, -4.000000000004, 6.000000000006, -4.000000000004, 1.000000000001 },
        { 1e15, 3e15, 3e15, 1e15, 0.0 }, 
        { 1e-15, -2e-5, 1e-10, 0.0, 0.0 },
        { 1.0, -29988.0, 299710003.0, -299690001000.0, 80970299901.0 },
        { 1.3, 0.8, -2.0, -1.1, 0.4 },
        { -2.5, -1.0, 0.5, 3.0, -1.2 },
        { 0.7, 1.6, -3.2, -4.1, 1.5 },
        { -0.9, -2.3, 1.4, 0.7, -0.2 },
        { 3.0, 1.5, -0.5, -0.2, 1.0 },
        { 0.01, -0.05, 0.1, -0.2, 0.5 },
        { -5.0, 2.5, -1.0, 0.5, -0.25 },
        { 1.0, -1.0, 1.0, -1.0, 1.0 },
        { -10.0, 1.0, -0.1, 0.01, -0.001 },
        { 0.2, 0.4, 0.8, 1.6, 3.2 }
    };


    //for (int i(0); i < 30; ++i){
    //    std::string cax = std::to_string(quad_coeffs[i][0]) + ", ";
    //    cax += std::to_string(quad_coeffs[i][1]) + ", ";
    //    cax += std::to_string(quad_coeffs[i][2]) + "";

    //    std::complex<long double> v0, v1, v2, v3; 
    //    int r = quadratic(quad_coeffs[i][0], quad_coeffs[i][1], quad_coeffs[i][2], &v0, &v1); 
    //    long double v1t = testroots<long double>(quad_coeffs[i][0], quad_coeffs[i][1], quad_coeffs[i][2], 0, 0, &v0, 2).real();
    //    long double v2t = testroots<long double>(quad_coeffs[i][0], quad_coeffs[i][1], quad_coeffs[i][2], 0, 0, &v1, 2).real(); 
    //    if (std::abs(v1t) < 1e-6 == std::abs(v2t) < 1e-6){continue;}
    //    printRoots<long double>("Quadratic - [" + cax + "]", &v0, &v1, nullptr, nullptr, r);
    //    std::cout << "test: " << v1t << " " << v2t << std::endl;
    //}
    
    //for (int i(0); i < 40; ++i){
    //    std::string cax = std::to_string(cubic_coeffs[i][0]) + ", ";
    //    cax += std::to_string(cubic_coeffs[i][1]) + ", ";
    //    cax += std::to_string(cubic_coeffs[i][2]) + ", ";
    //    cax += std::to_string(cubic_coeffs[i][3]) + "";

    //    std::complex<long double> v0, v1, v2, v3; 
    //    int r = cubic<long double>(cubic_coeffs[i][0], cubic_coeffs[i][1], cubic_coeffs[i][2], cubic_coeffs[i][3], &v0, &v1, &v2); 
    //    long double v1t = testroots<long double>(cubic_coeffs[i][0], cubic_coeffs[i][1], cubic_coeffs[i][2], cubic_coeffs[i][3], 0, &v0, 3).real();
    //    long double v2t = testroots<long double>(cubic_coeffs[i][0], cubic_coeffs[i][1], cubic_coeffs[i][2], cubic_coeffs[i][3], 0, &v1, 3).real(); 
    //    long double v3t = testroots<long double>(cubic_coeffs[i][0], cubic_coeffs[i][1], cubic_coeffs[i][2], cubic_coeffs[i][3], 0, &v2, 3).real();  
    //    if (std::abs(v1t) < 1e-6 && std::abs(v2t) < 1e-6){continue;}
    //    printRoots<long double>("Cubic - [" + cax + "]", &v0, &v1, &v2, nullptr, r);
    //    std::cout << "test: " << i << " | " << v1t << " " << v2t << " " << v3t << std::endl;
    //}


    for (int i(0); i < 30; ++i){
        std::string cax = std::to_string(quartic_coeffs[i][0]) + ", ";
        cax += std::to_string(quartic_coeffs[i][1]) + ", ";
        cax += std::to_string(quartic_coeffs[i][2]) + ", ";
        cax += std::to_string(quartic_coeffs[i][3]) + ", ";
        cax += std::to_string(quartic_coeffs[i][4]) + "";

        std::complex<long double> v0, v1, v2, v3; 
        int r = quartic<long double>(quartic_coeffs[i][0], quartic_coeffs[i][1], quartic_coeffs[i][2], quartic_coeffs[i][3], quartic_coeffs[i][4], &v0, &v1, &v2, &v3); 
        long double v1t = testroots<long double>(quartic_coeffs[i][0], quartic_coeffs[i][1], quartic_coeffs[i][2], quartic_coeffs[i][3], quartic_coeffs[i][4], &v0, 4).real();
        long double v2t = testroots<long double>(quartic_coeffs[i][0], quartic_coeffs[i][1], quartic_coeffs[i][2], quartic_coeffs[i][3], quartic_coeffs[i][4], &v1, 4).real(); 
        long double v3t = testroots<long double>(quartic_coeffs[i][0], quartic_coeffs[i][1], quartic_coeffs[i][2], quartic_coeffs[i][3], quartic_coeffs[i][4], &v2, 4).real();  
        long double v4t = testroots<long double>(quartic_coeffs[i][0], quartic_coeffs[i][1], quartic_coeffs[i][2], quartic_coeffs[i][3], quartic_coeffs[i][4], &v3, 4).real();  
        if (std::abs(v1t) < 1e-6 && std::abs(v2t) < 1e-6){continue;}
        printRoots<long double>("Quartic - [" + cax + "]", &v0, &v1, &v2, &v3, r);
        std::cout << "test: " << i << " | " << v1t << " " << v2t << " " << v3t << " " << v4t << std::endl;
    }








    return 0;
}

//    const g p = 3.0 * (a * c) - b2 / a2;
//    const g q = (b3 / a3) - 4.5 * b * c / a2 + 13.5 * d / a2;
//    const g o = -b / (3.0 * a); 
//









