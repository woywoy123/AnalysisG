#include <stdio.h>
#include <math.h>
#include <complex.h> // For complex number arithmetic
#include <iostream>

// Define constants for clarity
#define PI 3.14159265358979323846

typedef struct {double data[3][3];} Matrix3x3;
typedef struct {double data[3];} Vector3;
typedef struct {std::complex<double> data[3];} CVector3;

void print_matrix(const Matrix3x3* A);
void print_cvector(const CVector3* v, const char* title);
void print_vector(const Vector3* v, const char* title);
void normalize_vector(Vector3* v);
void solve_cubic(double a, double b, double c, double d, CVector3* roots);
void find_eigenvector(const Matrix3x3* A, std::complex<double> eigenvalue, Vector3* eigenvector);

void print_matrix(const Matrix3x3* A) {
    for (int i = 0; i < 3; ++i) {
        printf("  [ %.6e  %.6e  %.6e ]\n", A->data[i][0], A->data[i][1], A->data[i][2]);
    }
    printf("\n");
}

void print_cvector(const CVector3* v, const char* title) {
    printf("%s:\n", title);
    for (int i = 0; i < 3; ++i) {
        printf("  λ%d = %.9f + %.9fi\n", i + 1, v->data[i].real(), v->data[i].imag());
    }
    printf("\n");
}



// Fast square root approximation using inverse square root method
static inline double fast_sqrt(double x) {
    if (x <= 0.0) return 0.0; // Handle 0 and negative

    // Initial approximation using magic number
    double y = x;
    int64_t* i_ptr = reinterpret_cast<int64_t*>(&y);
    *i_ptr = 0x5fe6eb50c7b537a9 - (*i_ptr >> 1); // Magic constant for double
    
    // One Newton-Raphson iteration
    y = y * (1.5 - 0.5 * x * y * y);
    return x * y; // sqrt(x) = x * (1/sqrt(x))
}





void print_vector(const Vector3* v, const char* title) {
    printf("%s:\n", title);
    printf("  [ %.6f ]\n", v->data[0]);
    printf("  [ %.6f ]\n", v->data[1]);
    printf("  [ %.6f ]\n", v->data[2]);
    printf("\n");
}

int main() {
    Matrix3x3 A = {{
            { 1.01957185e-02,  6.97381504e-03, -8.87101492e-01},
            {-3.00361005e-02, -7.25096319e-03,  2.67903963e+00},
            { 3.46070529e-05,  3.70918000e-05, -2.94475530e-03}
    }};

    printf("Original Matrix A:\n");
    print_matrix(&A);

    double p    = -(A.data[0][0] + A.data[1][1] + A.data[2][2]);
    double m11   = A.data[1][1] * A.data[2][2] - A.data[1][2] * A.data[2][1];
    double m22   = A.data[0][0] * A.data[2][2] - A.data[0][2] * A.data[2][0];
    double m33   = A.data[0][0] * A.data[1][1] - A.data[0][1] * A.data[1][0];
    double q     = m11 + m22 + m33;
    double r     = -A.data[0][0] * m11 + A.data[0][1] * (A.data[1][0] * A.data[2][2] - A.data[1][2] * A.data[2][0]) 
                                       - A.data[0][2] * (A.data[1][0] * A.data[2][1] - A.data[1][1] * A.data[2][0]);


    printf("Characteristic Equation:\nλ³ + (%.18f)λ² + (%.18f)λ + (%.18f) = 0\n\n", p, q, r);
    CVector3 eigenvalues;
    solve_cubic(1, 8.67362e-19, 5.81954e-05, -2.11758e-22, &eigenvalues); 



    //solve_cubic(1.0, p, q, r, &eigenvalues);
    print_cvector(&eigenvalues, "Eigenvalues (λ)");

    for (int i = 0; i < 3; ++i) {
        std::cout << eigenvalues.data[i] << std::endl;
        continue;
        if (fabs(eigenvalues.data[i].imag()) < 1e-9) {
            Vector3 eigenvector;
            find_eigenvector(&A, eigenvalues.data[i].real(), &eigenvector);
            normalize_vector(&eigenvector);

            char title[50];
            sprintf(title, "Eigenvector for λ = %.18f", eigenvalues.data[i].real());
            print_vector(&eigenvector, title);
        }
    }

    return 0;
}


void normalize_vector(Vector3* v) {
    double mag = sqrt(v->data[0] * v->data[0] + v->data[1] * v->data[1] + v->data[2] * v->data[2]);
    if (mag < 1e-9){return;}
    v->data[0] /= mag; v->data[1] /= mag; v->data[2] /= mag;
}

void solve_cubic(double a, double b, double c, double d, CVector3* roots) {
    std::complex<double> p = (3.0 * a * c - b * b) / (3.0 * a * a);
    std::complex<double> q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    std::complex<double> root1, root2, root3;
    std::complex<double> delt = (q * q / 4.0) + (p * p * p / 27.0);
    double rt3 = pow(3.0, 0.5) / 2.0; 

    std::complex<double> ui = std::complex<double>(0.0, 1.0); 

    if (delt.real() >= 0) {
        std::complex<double> u_val = pow(-q / 2.0 + sqrt(delt), 1.0/3.0);
        std::complex<double> v_val = pow(-q / 2.0 - sqrt(delt), 1.0/3.0);
        root1 = u_val + v_val;
        root2 = -0.5 * (u_val + v_val) + std::complex<double>(ui * rt3) * (u_val - v_val);
        root3 = -0.5 * (u_val + v_val) - std::complex<double>(ui * rt3) * (u_val - v_val);
    } else { 
        std::complex<double> r   = pow(-(p * p * p) / 27.0, 0.5);
        std::complex<double> phi = acos(-q / (2.0 * r));
        root1 = 2.0 * pow(r, 1.0/3.0) * cos( phi / 3.0);
        root2 = 2.0 * pow(r, 1.0/3.0) * cos((phi + 2.0 * PI) / 3.0);
        root3 = 2.0 * pow(r, 1.0/3.0) * cos((phi + 4.0 * PI) / 3.0);
    }

    // Convert back to roots of original equation
    double shift = b / (3.0 * a);
    roots->data[0] = root1 - shift;
    roots->data[1] = root2 - shift;
    roots->data[2] = root3 - shift;
}

void find_eigenvector(const Matrix3x3* A, std::complex<double> eigenvalue, Vector3* eigenvector) {
    Matrix3x3 B; // B = A - λI
    for(int i = 0; i < 3; i++) {for(int j = 0; j < 3; j++) {B.data[i][j] = A->data[i][j];}}
    for(int i = 0; i < 3; i++) {B.data[i][i] -= eigenvalue.real();}

    std::cout << eigenvalue << std::endl; 



    Vector3 row1 = {B.data[0][0], B.data[0][1], B.data[0][2]};
    Vector3 row2 = {B.data[1][0], B.data[1][1], B.data[1][2]};

    eigenvector->data[0] = row1.data[1] * row2.data[2] - row1.data[2] * row2.data[1];
    eigenvector->data[1] = row1.data[2] * row2.data[0] - row1.data[0] * row2.data[2];
    eigenvector->data[2] = row1.data[0] * row2.data[1] - row1.data[1] * row2.data[0];

    double mag = sqrt(eigenvector->data[0]*eigenvector->data[0] +
                      eigenvector->data[1]*eigenvector->data[1] +
                      eigenvector->data[2]*eigenvector->data[2]);

    if (mag < 1e-9) {
        Vector3 row3 = {B.data[2][0], B.data[2][1], B.data[2][2]};
        eigenvector->data[0] = row1.data[1] * row3.data[2] - row1.data[2] * row3.data[1];
        eigenvector->data[1] = row1.data[2] * row3.data[0] - row1.data[0] * row3.data[2];
        eigenvector->data[2] = row1.data[0] * row3.data[1] - row1.data[1] * row3.data[0];
    }
}
