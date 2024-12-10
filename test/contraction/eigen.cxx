#include <iostream>
#include <complex>
#include <cstdio>
#include <cmath>

#define SIZE 3  // Define the size of the matrix

void multiply(double matrix[SIZE][SIZE], double vector[SIZE], double result[SIZE]) { 
    for (int i = 0; i < SIZE; i++) { 
        result[i] = 0.0; 
        for (int j = 0; j < SIZE; j++) {result[i] += matrix[i][j] * vector[j];} 
    } 
} 
 
double vector_norm(double vector[SIZE]) { 
    double norm = 0.0; 
    for (int i = 0; i < SIZE; i++) {norm += vector[i] * vector[i];} 
    return sqrt(norm); 
} 
 
void normalize(double vector[SIZE]) { 
    double norm = vector_norm(vector); 
    for (int i = 0; i < SIZE; i++) {vector[i] /= norm;} 
} 

double power_iteration(double matrix[SIZE][SIZE], double eigenvector[SIZE], int max_iterations, double tolerance) {
    double eigenvalue = 0.0;
    double temp_vector[SIZE] = {1.0, 1.0, 1.0};  // Initial guess for the eigenvector

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        double result[SIZE];
        multiply(matrix, temp_vector, result);

        // Calculate the new eigenvalue
        double new_eigenvalue = vector_norm(result);
        normalize(result);

        // Check for convergence
        if (fabs(new_eigenvalue - eigenvalue) < tolerance) {break;}
        eigenvalue = new_eigenvalue;
        for (int i = 0; i < SIZE; i++) {temp_vector[i] = result[i];}
        for (int i = 0; i < SIZE; i++) {eigenvector[i] = result[i];}
    }
    return eigenvalue;
}

int main() {
    
    double M[3][3] = {
        8.01318657e+01, 5.51435597e+00, 1.16572607e+07, 
        4.27558938e+01, 8.36588978e+00, 7.40343012e+06, 
        1.33325465e-05 -4.11361520e-05, -7.23462912e+00
    };

    double eigenvector[SIZE];
    double eigenvalue = power_iteration(M, eigenvector, 10000, 1e-12);

    printf("%.10g %.10g %.10g \n", M[0][0], M[0][1], M[0][2]); 
    printf("%.10g %.10g %.10g \n", M[1][0], M[1][1], M[1][2]); 
    printf("%.10g %.10g %.10g \n", M[2][0], M[2][1], M[2][2]); 

    printf("Dominant Eigenvalue: %f\n", eigenvalue);
    printf("Corresponding Eigenvector: [");
    for (int i = 0; i < SIZE; i++) {
        printf("%f", eigenvector[i]);
        if (i < SIZE - 1) printf(", ");
    }
    printf("]\n");

    return 0;
}



// determinant of 3x3 matrix (unchecked)
//double det3x3(double A[3][3]){
//   return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
//        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
//        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
//}
//
//int main(){
//
//    double M[3][3] = {
//        9.3871e-09 ,  1.7943e-08 , -2.9486e-08, 
//        1.7943e-08 ,  3.4296e-08 , -3.1988e-08,  
//        -2.9486e-08,  -3.1988e-08, -2.9142e-08
//    }; 
//
//    printf("%.10g %.10g %.10g \n", M[0][0], M[0][1], M[0][2]); 
//    printf("%.10g %.10g %.10g \n", M[1][0], M[1][1], M[1][2]); 
//    printf("%.10g %.10g %.10g \n", M[2][0], M[2][1], M[2][2]); 
//
//    // Set up characteristic equation:   det( A - lambda I ) = 0
//    // as a cubic in lambda:  a.lambda^3 + b.lambda^2 + c.lambda + d = 0
//    double a = -1.0; 
//    double b = M[0][0] + M[1][1] + M[2][2]; 
//    double c = M[1][2] * M[2][1] - M[1][1] * M[2][2] 
//             + M[2][0] * M[0][2] - M[2][2] * M[0][0] 
//             + M[0][1] * M[1][0] - M[0][0] * M[1][1];
//    double d = det3x3(M); 
//    printf("%.10g\n", d);
// 
//    // Solve cubic by Cardano's method (easier in complex numbers!)
//    double p = ( b * b - 3.0 * a * c ) / ( 9.0 * a * a );
//    double q = ( 9.0 * a * b * c - 27.0 * a * a * d - 2.0 * b * b * b ) / ( 54.0 * a * a * a );
//    std::complex delta = std::complex<double>(q * q - p * p * p, 0);
//
//    // warning: complex exponents and sqrt
//    std::complex g1 = std::pow( q + std::sqrt( delta ), 1.0 / 3.0 );
//    std::complex g2 = std::pow( q - std::sqrt( delta ), 1.0 / 3.0 );
//    double offset = -b / ( 3.0 * a );
//
//    // complex cube root of unity
//    std::complex omega  = std::complex<double>( -0.5, 0.5 * std::sqrt( 3.0 ) ); 
//    std::complex omega2 = omega * omega;
//    std::cout << omega2 << std::endl;
//
//    std::complex lmb1 = g1          + g2          + offset;
//    std::complex lmb2 = g1 * omega  + g2 * omega2 + offset;
//    std::complex lmb3 = g1 * omega2 + g2 * omega  + offset;
//
//    printf("r: %.20g, i: %.20g, \n", lmb1.real(), lmb1.imag());
//    printf("r: %.20g, i: %.20g, \n", lmb2.real(), lmb2.imag());
//    printf("r: %.20g, i: %.20g, \n", lmb3.real(), lmb3.imag());
//    return 0;
//} 
