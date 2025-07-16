#include <stdio.h>
#include <math.h>
#include <iostream>

#define MAX_SOLUTIONS 4
#define TOLERANCE 1e-8

#include <stdio.h>
#include <math.h>

#define MAX_SOLUTIONS 4
#define TOLERANCE 1e-8

int solve_quadratic(double a, double b, double c, double roots[2]) {
    double disc = b * b - 4 * a * c;
    if (disc < -TOLERANCE) return 0;
    if (fabs(disc) < TOLERANCE) {
        roots[0] = -b / (2 * a);
        return 1;
    }
    double sqrt_disc = sqrt(disc);
    roots[0] = (-b + sqrt_disc) / (2 * a);
    roots[1] = (-b - sqrt_disc) / (2 * a);
    return 2;
}

int solve_cubic(double a, double b, double c, double d, double roots[3]) {
    if (fabs(a) < TOLERANCE) {
        if (fabs(b) < TOLERANCE) {
            if (fabs(c) < TOLERANCE) return 0;
            roots[0] = -d / c;
            return 1;
        }
        return solve_quadratic(b, c, d, roots);
    }
    
    double b2 = b * b;
    double p = (3 * a * c - b2) / (3 * a * a);
    double q = (2 * b2 * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);
    double p3 = p * p * p;
    double disc = q * q / 4 + p3 / 27;
    
    if (disc > TOLERANCE) {
        double sqrt_disc = sqrt(disc);
        double u = cbrt(-q / 2 + sqrt_disc);
        double v = cbrt(-q / 2 - sqrt_disc);
        roots[0] = u + v - b / (3 * a);
        return 1;
    } else if (disc < -TOLERANCE) {
        double angle = acos(3 * q * sqrt(-3 / p3) / (2 * p));
        double r = 2 * sqrt(-p / 3);
        roots[0] = r * cos(angle / 3) - b / (3 * a);
        roots[1] = r * cos((angle + 2 * M_PI) / 3) - b / (3 * a);
        roots[2] = r * cos((angle + 4 * M_PI) / 3) - b / (3 * a);
        return 3;
    } else {
        double u = cbrt(-q / 2);
        roots[0] = 2 * u - b / (3 * a);
        roots[1] = -u - b / (3 * a);
        return 2;
    }
}

int solve_quartic(double a, double b, double c, double d, double e, double roots[4]) {
    if (fabs(a) < TOLERANCE) {
        return solve_cubic(b, c, d, e, roots);
    }
    
    double b3 = b / a;
    double c2 = c / a;
    double d1 = d / a;
    double e0 = e / a;
    
    double p = c2 - 3 * b3 * b3 / 8;
    double q = d1 + b3 * b3 * b3 / 8 - b3 * c2 / 2;
    double r = e0 - b3 * d1 / 4 + b3 * b3 * c2 / 16 - 3 * b3 * b3 * b3 * b3 / 256;
    
    if (fabs(q) < TOLERANCE) {
        double z_roots[2];
        int nz = solve_quadratic(1.0, p, r, z_roots);
        int n = 0;
        for (int i = 0; i < nz; i++) {
            if (z_roots[i] >= 0) {
                double sqrt_z = sqrt(z_roots[i]);
                roots[n++] = sqrt_z - b3 / 4;
                if (sqrt_z > TOLERANCE) {
                    roots[n++] = -sqrt_z - b3 / 4;
                }
            }
        }
        return n;
    }
   
    double cubic_roots[3];
    int nc = solve_cubic(1.0, 2*p, p*p - 4*r, -q*q, cubic_roots);
    int n = 0;
    double y_roots[4];
    
    for (int i = 0; i < nc; i++) {
        double z = cubic_roots[i];
        if (z < -TOLERANCE) continue;
        if (z < 0) z = 0;
        double g = sqrt(z);
        double denom = 2 * g;
        if (fabs(denom) < TOLERANCE) continue;
        
        double h = (z + p - q / g) / 2;
        double k = (z + p + q / g) / 2;
        
        double quad1_roots[2], quad2_roots[2];
        int n1 = solve_quadratic(1.0, g, h, quad1_roots);
        int n2 = solve_quadratic(1.0, -g, k, quad2_roots);
        
        for (int j = 0; j < n1; j++) y_roots[n++] = quad1_roots[j];
        for (int j = 0; j < n2; j++) y_roots[n++] = quad2_roots[j];
    }
    int valid_roots = 0;
    for (int i = 0; i < n; i++) {
        roots[valid_roots++] = y_roots[i] - b3 / 4;
    }
    return valid_roots;
}

void solve_t1_t2(
    double px, double py,
    double H00, double H01, double H02,
    double H10, double H11, double H12,
    double M00, double M01, double M02,
    double M10, double M11, double M12,
    double t1_sol[], double t2_sol[], int *num_sol
) {
    double K1 = px - H02 - M02;
    double K2 = py - H12 - M12;
    double Delta = H00 * H11 - H01 * H10;
    
    if (fabs(Delta) < TOLERANCE) {
        *num_sol = 0;
        return;
    }
    
    double P1 = H11 * K1 - H01 * K2;
    double Q1 = H11 * M00 - H01 * M10;
    double R1 = H11 * M01 - H01 * M11;
    
    double P2 = -H10 * K1 + H00 * K2;
    double Q2 = -H10 * M00 + H00 * M10;
    double R2 = -H10 * M01 + H00 * M11;
    
    double a = Q1 * Q1 + Q2 * Q2;
    double b = R1 * R1 + R2 * R2;
    double c_val = Q1 * R1 + Q2 * R2;
    double d_val = -2 * (P1 * Q1 + P2 * Q2);
    double e_val = -2 * (P1 * R1 + P2 * R2);
    double f_val = P1 * P1 + P2 * P2 - Delta * Delta;
    
    double A = a - b;
    double B = d_val;
    double C = b + f_val;
    double D = 2 * c_val;
    double E = e_val;
    
    double a4 = A * A + D * D;
    double a3 = 2 * (A * B + D * E);
    double a2 = 2 * A * C + B * B + E * E - D * D;
    double a1 = 2 * (B * C - D * E);
    double a0 = C * C - E * E;
   
    double U_roots[4];
    int nU = solve_quartic(a4, a3, a2, a1, a0, U_roots);
    int sol_count = 0;
    
    for (int i = 0; i < nU; i++) {
        double U = U_roots[i];
        std::cout << U << std::endl;
        if (U < -1 - TOLERANCE || U > 1 + TOLERANCE) continue;
        if (U < -1) U = -1;
        if (U > 1) U = 1;
        
        double denom = D * U + E;
        double V = 0;
        int valid_V = 0;
        
        if (fabs(denom) > TOLERANCE) {
            V = -(A * U * U + B * U + C) / denom;
            if (fabs(U * U + V * V - 1) < TOLERANCE) valid_V = 1;
        } else {
            double num = A * U * U + B * U + C;
            if (fabs(num) < TOLERANCE) {
                double V_sq = 1 - U * U;
                if (V_sq < 0) continue;
                double V1 = sqrt(V_sq);
                double V2 = -V1;
                
                double V_candidates[2] = {V1, V2};
                for (int k = 0; k < 2; k++) {
                    V = V_candidates[k];
                    double arg_x = K1 - (M00 * U + M01 * V);
                    double arg_y = K2 - (M10 * U + M11 * V);
                    double cos_t1 = (H11 * arg_x - H01 * arg_y) / Delta;
                    double sin_t1 = (-H10 * arg_x + H00 * arg_y) / Delta;
                    
                    double eq1 = -px + H00 * cos_t1 + H01 * sin_t1 + H02 + M00 * U + M01 * V + M02;
                    double eq2 = -py + H10 * cos_t1 + H11 * sin_t1 + H12 + M10 * U + M11 * V + M12;
                    
                    if (fabs(eq1) < TOLERANCE && fabs(eq2) < TOLERANCE) {
                        t1_sol[sol_count] = atan2(sin_t1, cos_t1);
                        t2_sol[sol_count] = atan2(V, U);
                        sol_count++;
                    }
                }
                continue;
            }
        }
        
        if (valid_V) {
            double arg_x = K1 - (M00 * U + M01 * V);
            double arg_y = K2 - (M10 * U + M11 * V);
            double cos_t1 = (H11 * arg_x - H01 * arg_y) / Delta;
            double sin_t1 = (-H10 * arg_x + H00 * arg_y) / Delta;
            
            double eq1 = -px + H00 * cos_t1 + H01 * sin_t1 + H02 + M00 * U + M01 * V + M02;
            double eq2 = -py + H10 * cos_t1 + H11 * sin_t1 + H12 + M10 * U + M11 * V + M12;
            
            if (fabs(eq1) < TOLERANCE && fabs(eq2) < TOLERANCE) {
                t1_sol[sol_count] = atan2(sin_t1, cos_t1);
                t2_sol[sol_count] = atan2(V, U);
                sol_count++;
            }
        }
    }
    
    *num_sol = sol_count;
}



void test_case(double py, int expected) {
    double t1_sol[4], t2_sol[4];
    int num_sol;

    double MET[2][2] = {{0}};
    MET[0][0] =  106.435841000000; 
    MET[0][1] = -141.293331000000; 

    double H1[3][3] = {{0}};
    H1[0][0] = -127.937382333990; H1[0][1] = -186.876109104662; H1[0][2] = -122.897649651631; 
    H1[1][0] = -347.113584765497; H1[1][1] =  55.131753793220 ; H1[1][2] = -355.655876974117; 

    double H2[3][3] = {{0}}; 
    H2[0][0] =  63.165309472213 ; H2[0][1] = -45.299561107755; H2[0][2] =  59.021273671266 ; 
    H2[1][0] = -185.929103473096; H2[1][1] = -40.467983180422; H2[1][2] = -176.446029370150;
    std::cout << "-----------------------------" << std::endl; 


    solve_t1_t2(
            MET[0][0], MET[0][1], 
            H1[0][0], H1[0][1], H1[0][2], H1[1][0], H1[1][1], H1[1][2], 
            H2[0][0], H2[0][1], H2[0][2], H2[1][0], H2[1][1], H2[1][2],
            t1_sol, t2_sol, &num_sol
    );
    printf("Orignal: %d \n", num_sol);
    for (int i = 0; i < num_sol; i++) {printf("Solution %d: t1 = %.6f rad, t2 = %.6f rad\n", i+1, t1_sol[i], t2_sol[i]);}
    std::cout << "===============" << std::endl;
    abort(); 
}

int main() {
    double cubic_roots[3];
    int nc = solve_cubic(1, 8.67362e-19, 5.81954e-05, -2.11758e-22, cubic_roots);
    std::cout << cubic_roots[0] << " | " << cubic_roots[1] << " | " << cubic_roots[2] << std::endl;    
    abort(); 




    test_case(2.0, 0);   // 0 solutions
    test_case(1.5, 1);   // 1 solution
    test_case(1.0, 2);   // 2 solutions
    test_case(0.5, 3);   // 3 solutions
    test_case(0.0, 4);   // 4 solutions
    return 0;
}
