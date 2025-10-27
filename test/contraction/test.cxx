#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <complex.h>
using namespace std; 


#define PI 3.14159265358979323846

// Structure to hold the roots of a cubic equation
typedef struct {
    std::complex<double> roots[3];
    int num_real_roots;
} CubicRoots;

double creal(std::complex<double> g){return g.real();}
double cimag(std::complex<double> g){return g.imag();}


CubicRoots solve_cubic(double a, double b, double c, double d) {
    CubicRoots result;
    
    // Normalize coefficients
    double A = b / a;
    double B = c / a;
    double C = d / a;
    
    // Depressed cubic: t³ + pt + q = 0 where t = x + A/3
    double p = B - A * A / 3.0;
    double q = C + (2.0 * A * A * A - 9.0 * A * B) / 27.0;
    
    // Discriminant
    double discriminant = (q * q / 4.0) + (p * p * p / 27.0);
    
    if (std::fabs(discriminant) < 1e-12) {
        discriminant = 0.0; // Handle floating point errors
    }
    
    if (discriminant > 0) {
        // One real root, two complex roots
        double sqrt_disc = std::sqrt(discriminant);
        double u = std::cbrt(-q / 2.0 + sqrt_disc);
        double v = std::cbrt(-q / 2.0 - sqrt_disc);
        
        double real_root = u + v - A / 3.0;
        std::complex<double> complex_root1 = (-(u + v) / 2.0 - A / 3.0) + I * (std::sqrt(3.0) * (u - v) / 2.0);
        std::complex<double> complex_root2 = (-(u + v) / 2.0 - A / 3.0) - I * (std::sqrt(3.0) * (u - v) / 2.0);
        
        result.roots[0] = real_root;
        result.roots[1] = complex_root1;
        result.roots[2] = complex_root2;
        result.num_real_roots = 1;
    }
    else if (discriminant < 0) {
        // Three real roots
        double r = std::sqrt(-p * p * p / 27.0);
        double theta = acos(-q / (2.0 * r));
        
        result.roots[0] = 2.0 * std::cbrt(r) * cos(theta / 3.0) - A / 3.0;
        result.roots[1] = 2.0 * std::cbrt(r) * cos((theta + 2.0 * PI) / 3.0) - A / 3.0;
        result.roots[2] = 2.0 * std::cbrt(r) * cos((theta + 4.0 * PI) / 3.0) - A / 3.0;
        result.num_real_roots = 3;
    }
    else {
        // discriminant == 0: Multiple roots
        double u = std::cbrt(-q / 2.0);
        
        if (fabs(u) < 1e-12) {
            // Triple root
            result.roots[0] = -A / 3.0;
            result.roots[1] = -A / 3.0;
            result.roots[2] = -A / 3.0;
        }
        else {
            // Double root and single root
            result.roots[0] = 2.0 * u - A / 3.0;
            result.roots[1] = -u - A / 3.0;
            result.roots[2] = -u - A / 3.0;
        }
        result.num_real_roots = 3;
    }
    
    return result;
}

// Function to print roots with formatting
void print_roots(CubicRoots roots) {
    printf("Number of real roots: %d\n", roots.num_real_roots);
    printf("Roots:\n");
    
    for (int i = 0; i < 3; i++) {
        double real_part = creal(roots.roots[i]);
        double imag_part = cimag(roots.roots[i]);
        
        if (fabs(imag_part) < 1e-10) {
            // Real root
            printf("  Root %d: %.6f (real)\n", i + 1, real_part);
        } else {
            // Complex root
            printf("  Root %d: %.6f %+.6fi (complex)\n", i + 1, real_part, imag_part);
        }
    }
    printf("\n");
}

// Function to test the cubic solver with given coefficients
void test_cubic_solver(double a, double b, double c, double d, const char* description) {
    printf("Testing: %s\n", description);
    printf("Coefficients: [%.6f, %.6f, %.6f, %.6f]\n", a, b, c, d);
    
    CubicRoots roots = solve_cubic(a, b, c, d);
    print_roots(roots);
    
    // Verify the roots by plugging them back into the equation
    printf("Verification (should be close to 0):\n");
    for (int i = 0; i < 3; i++) {
        std::complex<double> x = roots.roots[i];
        std::complex<double> value = a * x * x * x + b * x * x + c * x + d;
        
        if (fabs(cimag(value)) < 1e-10) {
            printf("  f(root %d) = %.6e\n", i + 1, creal(value));
        } else {
            printf("  f(root %d) = %.6e %+.6ei\n", i + 1, creal(value), cimag(value));
        }
    }
    printf("----------------------------------------\n\n");
}

int main() {
    printf("CUBIC EQUATION SOLVER\n");
    printf("========================================\n\n");
    
    // Test case 1: x³ - x² - 2x = 0
    // Factors as: x(x-2)(x+1) = 0
    // Roots: 0, 2, -1
    test_cubic_solver(1.0, -1.0, -2.0, 0.0, "x³ - x² - 2x = 0");
    
    // Test case 2: 0.1x³ - 999x² - 199.96x + 199.98 = 0
    test_cubic_solver(0.1, -999.0, -199.96, 199.98, "0.1x³ - 999x² - 199.96x + 199.98 = 0");
    
    // Additional test cases
    printf("ADDITIONAL TEST CASES:\n");
    printf("========================================\n\n");
    
    // Test case 3: x³ - 6x² + 11x - 6 = 0
    // Factors as: (x-1)(x-2)(x-3) = 0
    // Roots: 1, 2, 3
    test_cubic_solver(1.0, -6.0, 11.0, -6.0, "x³ - 6x² + 11x - 6 = 0");
    
    // Test case 4: x³ + 1 = 0
    // Roots: -1, 0.5 ± 0.866i
    test_cubic_solver(1.0, 0.0, 0.0, 1.0, "x³ + 1 = 0");
    
    // Test case 5: x³ - 3x² + 3x - 1 = 0
    // Factors as: (x-1)³ = 0
    // Triple root: 1
    test_cubic_solver(1.0, -3.0, 3.0, -1.0, "x³ - 3x² + 3x - 1 = 0");
    
    return 0;
}
