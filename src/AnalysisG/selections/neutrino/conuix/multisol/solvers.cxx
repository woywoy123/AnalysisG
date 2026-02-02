#include "multisol/solvers.h"
#include "multisol/matrix.h"
#include <iostream>

long double det3(long double** data){
    return data[0][0] * data[1][1] * data[2][2] + data[0][1] * data[1][2] * data[2][0] 
         + data[0][2] * data[1][0] * data[2][1] - data[0][2] * data[1][1] * data[2][0]
         - data[0][1] * data[1][0] * data[2][2] - data[0][0] * data[1][2] * data[2][1];
}

long double det2(long double** data){
    return data[0][0] * data[1][1] - data[0][1] * data[1][0];
}

matrix_t roots_t::vec(){
    int s = 0;
    matrix_t eig = matrix_t(this -> num_r, 2); 
    if (s < this -> num_r){eig.at(s, 0) = this -> a.real(); eig.at(s, 1) = this -> a.imag(); s++;}
    if (s < this -> num_r){eig.at(s, 0) = this -> b.real(); eig.at(s, 1) = this -> b.imag(); s++;}
    if (s < this -> num_r){eig.at(s, 0) = this -> c.real(); eig.at(s, 1) = this -> c.imag(); s++;}
    if (s < this -> num_r){eig.at(s, 0) = this -> d.real(); eig.at(s, 1) = this -> d.imag(); s++;}
    return eig;
}


matrix_t circle(){
    matrix_t c(3, 3);
    c.at(0, 0) =  1.0L; 
    c.at(1, 1) =  1.0L;
    c.at(2, 2) = -1.0L; 
    return c; 
}


matrix_t identity(){
    matrix_t c(3, 3);
    c.at(0, 0) = 1.0L; 
    c.at(1, 1) = 1.0L;
    c.at(2, 2) = 1.0L; 
    return c; 
}

matrix_t S0(long double metx, long double mety){
    matrix_t v0(3, 3); 
    v0.at(0, 2) = metx; 
    v0.at(1, 2) = mety;
    return v0 - circle(); 
}

matrix_t V4(long double metx, long double mety, long double metz){
    matrix_t v0(4, 4); 
    v0.at(0, 2) = metx; 
    v0.at(1, 2) = mety; 
    v0.at(2, 2) = metz; 
    return v0; 
}

roots_t find_roots(long double a, long double b, long double c, long double tol){
    std::complex<long double> disc = b * b - 4 * a * c;
    roots_t r;
    if (disc.real() < tol){
        r.a = -b / (2.0L * a);
        r.num_r = 1; 
        return r;
    } 
    disc = std::sqrt(disc);
    r.a = (-b + disc) / (2 * a);
    r.b = (-b - disc) / (2 * a);
    r.num_r = 2; 
    return r; 
}


roots_t find_roots(long double a, long double b, long double c, long double d, long double tol) {
    roots_t r;
    
    if (std::fabs(a) < tol) {
        if (std::fabs(b) >= tol){return find_roots(b, c, d, tol);}
        if (std::fabs(c) >= tol) {
            r.a = -d / c;
            r.num_r = 1;
            return r;
        }
        r.num_r = 0;
        return r;
    }
   
    const long double three = 1.0L / 3.0L; 

    long double a1 = b / a;
    long double a2 = c / a;
    long double a3 = d / a;
    
    long double Q = (a1*a1 - 3.0L*a2) * three * three;
    long double R = (2.0L*a1*a1*a1 - 9.0L*a1*a2 + 27.0L*a3) / 54.0L;
    
    long double Q3 = Q * Q * Q;
    long double R2 = R * R;
    long double dis = Q3 - R2;
    
    if (std::fabs(dis) < tol) {dis = 0.0L;}
    if (dis >= 0 && Q3 != 0) {
        long double theta = std::acos(R / std::sqrt(Q3)) * three;
        long double sqrtQ = -2.0L * std::sqrt(Q);
        
        r.a = sqrtQ * std::cos(theta) - a1 * three;
        r.b = sqrtQ * std::cos(theta + 2.0L*M_PI * three) - a1 * three;
        r.c = sqrtQ * std::cos(theta + 4.0L*M_PI * three) - a1 * three;
        r.num_r = 3;
    } else {
        long double A = -std::copysign(1.0L, R) * std::cbrt(std::fabs(R) + std::sqrt(-dis));
        long double B = (std::fabs(A) < tol) ? 0.0L : Q / A;
        r.a = (A + B) - a1 * three;
        std::complex<long double> cr = std::complex<long double>(
            -0.5L*(A + B) - a1 * three, 
            std::sqrt(3.0L) * 0.5L * std::fabs(A - B)
        );
        r.b = cr;
        r.c = std::conj(cr);
        r.num_r = 3;
    }
    
    return r;
}

void swap_index(matrix_t* v, int idx){
    long double tmp = v -> at(idx, 0);
    v -> at(idx, 0) = v -> at(idx, 1);
    v -> at(idx, 1) = tmp; 
}

void multisqrt(long double y, long double roots[2], int *count){
    *count = 0;
    if (y < 0) return;
    if (!std::fabs(y)){roots[0] = 0; *count = 1; return;}
    long double r = pow(y, 0.5);
    roots[0] = -r; roots[1] = r;
    *count = 2;
}


void factor_degenerate(const matrix_t* G, matrix_t* lines, int* lc) {
    if (std::fabs(G -> at(0, 0)) == 0 && std::fabs(G -> at(1, 1)) == 0){
        lines -> at(0, 0) = G -> at(0, 1); 
        lines -> at(0, 1) = 0.0L;       
        lines -> at(0, 2) = G -> at(1, 2);
        lines -> at(1, 0) = 0.0L;       
        lines -> at(1, 1) = G -> at(0, 1); 
        lines -> at(1, 2) = G -> at(0, 2) - G -> at(1, 2);
        *lc = 2;
        return;
    }

    int swapxy = (std::fabs(G -> at(0, 0)) > std::fabs(G -> at(1, 1)));
    matrix_t Q = G -> clone(); 
    for (int i(0); i < 3*swapxy; i++){
        long double tmp = Q.at(0, i);
        Q.at(0, i) = Q.at(1, i);
        Q.at(1, i) = tmp;
    }

    for (int j(0); j < 3 * swapxy; j++){swap_index(&Q, j);}
    matrix_t Q_ = Q * (1.0 / Q.at(1, 1));
    matrix_t D_ = Q_.cofactor();
    long double q22 = D_.at(2, 2); 

    int r_count;
    long double r[2];
    if (-q22 <= 0){
        multisqrt(-D_.at(0, 0), r, &r_count);
        for (int i(0); i < r_count; ++i) {
            lines -> at(i, 0) = Q_.at(0, 1); 
            lines -> at(i, 1) = Q_.at(1, 1);
            lines -> at(i, 2) = Q_.at(1, 2) + r[i];
            if (!swapxy){continue;}
            swap_index(lines, i); 
        }
        *lc = r_count; 
        return; 
    } 
    double x0 = D_.at(0, 2) / q22; 
    double y0 = D_.at(1, 2) / q22;
    multisqrt(-q22, r, &r_count);
    for (int i(0); i < r_count; ++i) {
        lines -> at(i, 0) =  Q_.at(0, 1) + r[i]; 
        lines -> at(i, 1) =  Q_.at(1, 1); 
        lines -> at(i, 2) = -Q_.at(1, 1) *y0 - (Q_.at(0, 1) + r[i])*x0;
        if (!swapxy){continue;}
        swap_index(lines, i); 
    }
    *lc = r_count; 
}

int intersections_ellipse_line(matrix_t* ellipse, matrix_t* line, int k, matrix_t* pts){
    long double A = line -> at(k, 0);
    long double B = line -> at(k, 1);
    long double C = line -> at(k, 2);
    const long double epsilon = 1e-12;

    if (std::abs(A) <= epsilon) {return 0;}
    
    long double m00 = ellipse -> at(0, 0);
    long double m01 = ellipse -> at(0, 1);
    long double m02 = ellipse -> at(0, 2);
    long double m11 = ellipse -> at(1, 1);
    long double m12 = ellipse -> at(1, 2);
    long double m22 = ellipse -> at(2, 2);
    
    int pts_ = 0;
    if (std::abs(B) > epsilon) {
        long double m = -A / B;
        long double d = -C / B;
        
        long double a = m00 + 2 * m01 * m + m11 * m * m;
        long double b = 2 * (m01 * d + m02 + m11 * m * d + m12 * m);
        long double c = m11 * d * d + 2 * m12 * d + m22;
        long double dis = b * b - 4 * a * c;
        if (dis < -epsilon){return pts_;}

        if (dis < 0){dis = 0;}
        long double srt = std::sqrt(dis);
        for (int sign = -1; sign <= 1; sign += 2) {
            long double x = (-b + sign * srt) / (2 * a);
            long double y = m*x + d;
            if (std::isnan(x) || std::isnan(y)){continue;}
            pts -> at(pts_, 0) = x;
            pts -> at(pts_, 1) = y;
            pts -> at(pts_, 2) = 1.0;
            pts_++;
        }
        return pts_; 
    }

    long double x = -C / A;
    long double b = 2 * m01 * x + 2 * m12;
    long double c = m00 * x * x + 2 * m02 * x + m22;
    long double dis = b * b - 4 * m11 * c;
    if (dis < -epsilon) {return pts_;}

    if (dis < 0){dis = 0;}
    long double srt = std::sqrt(dis);
    for (int sign = -1; sign <= 1; sign += 2) {
        long double y = (-b + sign * srt) / (2 * m11);
        if (std::isnan(y)){continue;}
        pts -> at(pts_, 0) = x;
        pts -> at(pts_, 1) = y;
        pts -> at(pts_, 2) = 1.0;
        pts_++;
    }
    return pts_;
}

matrix_t* intersection_ellipses(matrix_t* A, matrix_t* B){
    bool swp = std::fabs(B -> det()) > std::fabs(A -> det()); 
    matrix_t A_ = (swp) ? B -> clone() : A -> clone(); 
    matrix_t B_ = (swp) ? A -> clone() : B -> clone(); 

    swp = false; 
    matrix_t sols; 
    matrix_t t   = A_.inv().dot(B_); 
    matrix_t eig = t.eigenvalues().vec();
    for (int x(0); x < eig.r; ++x){
        int lx = -1; 
        matrix_t line = matrix_t(3, 3);
        matrix_t G = B_ - A_ * eig.at(x, 0); 

        factor_degenerate(&G, &line, &lx); 
        for (int k(0); k < lx; ++k){
            matrix_t pts(2, 3); 
            if (!intersections_ellipse_line(&A_, &line, k, &pts)){continue;} 
            if (!swp){sols = pts; swp = true; continue;}
            sols = sols.cat(pts);  
        }
    }
    if (!swp){return nullptr;}
    return new matrix_t(sols); 
}
