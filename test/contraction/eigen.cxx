#include <iostream>
#include <complex>
#include <cstdio>
#include <cmath>

// determinant of 3x3 matrix (unchecked)
double det3x3(double A[3][3]){
   return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

int main(){

    double M[3][3] = {
        {-1.2783e-09, -4.0274e-09, -1.7979e-08},
        {-4.0274e-09, -1.2689e-08,  1.6132e-08},
        { 1.7979e-08, -1.6132e-08,  1.9274e-08}
    }; 
    printf("%.10g %.10g %.10g \n", M[0][0], M[0][1], M[0][2]); 
    printf("%.10g %.10g %.10g \n", M[1][0], M[1][1], M[1][2]); 
    printf("%.10g %.10g %.10g \n", M[2][0], M[2][1], M[2][2]); 

    // Set up characteristic equation:   det( A - lambda I ) = 0
    // as a cubic in lambda:  a.lambda^3 + b.lambda^2 + c.lambda + d = 0
    double a = -1.0; 
    double b = M[0][0] + M[1][1] + M[2][2]; 
    double c = M[1][2] * M[2][1] - M[1][1] * M[2][2] 
             + M[2][0] * M[0][2] - M[2][2] * M[0][0] 
             + M[0][1] * M[1][0] - M[0][0] * M[1][1];
    double d = det3x3(M); 
    printf("%.10g\n", d);
 
    // Solve cubic by Cardano's method (easier in complex numbers!)
    double p = ( b * b - 3.0 * a * c ) / ( 9.0 * a * a );
    double q = ( 9.0 * a * b * c - 27.0 * a * a * d - 2.0 * b * b * b ) / ( 54.0 * a * a * a );
    std::complex delta = std::complex<double>(q * q - p * p * p, 0);

    // warning: complex exponents and sqrt
    std::complex g1 = std::pow( q + std::sqrt( delta ), 1.0 / 3.0 );
    std::complex g2 = std::pow( q - std::sqrt( delta ), 1.0 / 3.0 );
    double offset = -b / ( 3.0 * a );

    // complex cube root of unity
    std::complex omega  = std::complex<double>( -0.5, 0.5 * std::sqrt( 3.0 ) ); 
    std::complex omega2 = omega * omega;
    std::cout << omega2 << std::endl;

    std::complex lmb1 = g1          + g2          + offset;
    std::complex lmb2 = g1 * omega  + g2 * omega2 + offset;
    std::complex lmb3 = g1 * omega2 + g2 * omega  + offset;

    printf("r: %.20g, i: %.20g, \n", lmb1.real(), lmb1.imag());
    printf("r: %.20g, i: %.20g, \n", lmb2.real(), lmb2.imag());
    printf("r: %.20g, i: %.20g, \n", lmb3.real(), lmb3.imag());
    return 0;
} 
