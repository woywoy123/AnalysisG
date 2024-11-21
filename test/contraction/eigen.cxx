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
    double M[3][3] = {{3, 2, 6}, {2, 2, 5}, {-2, -1, -4}}; 
    printf("%f %f %f \n", M[0][0], M[0][1], M[0][2]); 
    printf("%f %f %f \n", M[1][0], M[1][1], M[1][2]); 
    printf("%f %f %f \n", M[2][0], M[2][1], M[2][2]); 


    // Set up characteristic equation:   det( A - lambda I ) = 0
    // as a cubic in lambda:  a.lambda^3 + b.lambda^2 + c.lambda + d = 0
    double a = -1.0; 
    double b = M[0][0] + M[1][1] + M[2][2]; 
    double c = M[1][2] * M[2][1] 
             - M[1][1] * M[2][2] 
             + M[2][0] * M[0][2] 
             - M[2][2] * M[0][0] 
             + M[0][1] * M[1][0] 
             - M[0][0] * M[1][1];
    double d = det3x3(M); 
    printf("%f\n", d);
 
    // Solve cubic by Cardano's method (easier in complex numbers!)
    double p = ( b * b - 3.0 * a * c ) / ( 9.0 * a * a );
    double q = ( 9.0 * a * b * c - 27.0 * a * a * d - 2.0 * b * b * b ) / ( 54.0 * a * a * a );
    std::complex delta = q * q - p * p * p;

    // warning: complex exponents and sqrt
    std::complex g1 = std::pow( q + std::sqrt( delta ), 1.0 / 3.0 );
    std::complex g2 = std::pow( q - std::sqrt( delta ), 1.0 / 3.0 );
    double offset = -b / ( 3.0 * a );

    // complex cube root of unity
    std::complex omega  = std::complex<double>( -0.5, 0.5 * std::sqrt( 3.0 ) ); 
    std::complex omega2 = omega * omega;

    std::complex lmb1 = g1          + g2          + offset;
    std::complex lmb2 = g1 * omega  + g2 * omega2 + offset;
    std::complex lmb3 = g1 * omega2 + g2 * omega  + offset;

    printf("r: %f, i: %f, \n", lmb1.real(), lmb1.imag());
    printf("r: %f, i: %f, \n", lmb2.real(), lmb2.imag());
    printf("r: %f, i: %f, \n", lmb3.real(), lmb3.imag());
    return 0;
} 
