#include "matrix.h"
#include <complex.h>

double costheta(particle* p1, particle* p2){
    double p2_1 = p1 -> p2();
    double p2_2 = p2 -> p2();  
    double pxx  = p1 -> px * p2 -> px + p1 -> py * p2 -> py + p1 -> pz * p2 -> pz; 
    return pxx / pow(p2_1 * p2_2, 0.5); 
}
double sintheta(particle* p1, particle* p2){return pow(1 - pow(costheta(p1, p2), 2), 0.5);}

double** matrix(int row, int col){
    double** mx = (double**)malloc(row*sizeof(double*));
    for (int x(0); x < row; ++x){mx[x] = (double*)malloc(col*sizeof(double));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = 0;}}
    return mx;  
}

double** dot(double** v1, double** v2, int r1, int c1, int r2, int c2){
    double** vo = matrix(r1, c2); 
    for (int x(0); x < r1; ++x){
        for (int j(0); j < c2; ++j){
            double sm = 0; 
            for (int y(0); y < r2; ++y){sm += v1[x][y] * v2[y][j];}
            vo[x][j] = sm; 
        }
    }
    return vo; 
}

double** T(double** v1, int r, int c){
    double** vo = matrix(r, c); 
    for (int x(0); x < r; ++x){for (int y(0); y < c; ++y){vo[y][x] = v1[x][y];}}
    return vo; 
}


double** scale(double** v, double s){
    double** o = matrix(3, 3); 
    for (int x(0); x < 3; ++x){for (int y(0); y < 3; ++y){o[x][y] = v[x][y]*s;}}
    return o; 
}


double** arith(double** v1, double** v2, double s){
    double** o = matrix(3, 3); 
    for (int x(0); x < 3; ++x){for (int y(0); y < 3; ++y){o[x][y] = v1[x][y] + s*v2[x][y];}}
    return o; 
}










double abcd(double a, double b, double c, double d){return a*d - b*c;}

double** cof(double** v){
    double** ov = matrix(3, 3); 
    ov[0][0] =  abcd(v[1][1], v[1][2], v[2][1], v[2][2]); 
    ov[0][1] = -abcd(v[1][0], v[1][2], v[2][0], v[2][2]); 
    ov[0][2] =  abcd(v[1][0], v[1][1], v[2][0], v[2][1]);

    ov[1][0] = -abcd(v[0][1], v[0][2], v[2][1], v[2][2]); 
    ov[1][1] =  abcd(v[0][0], v[0][2], v[2][0], v[2][2]); 
    ov[1][2] = -abcd(v[0][0], v[0][1], v[2][0], v[2][1]); 

    ov[2][0] =  abcd(v[0][1], v[0][2], v[1][1], v[1][2]); 
    ov[2][1] = -abcd(v[0][0], v[0][2], v[1][0], v[1][2]); 
    ov[2][2] =  abcd(v[0][0], v[0][1], v[1][0], v[1][1]); 
    return ov; 
}

double det(double** v){
    double a =  v[0][0] * abcd(v[1][1], v[1][2], v[2][1], v[2][2]); 
    double b = -v[0][1] * abcd(v[1][0], v[1][2], v[2][0], v[2][2]); 
    double c =  v[0][2] * abcd(v[1][0], v[1][1], v[2][0], v[2][1]); 
    return a + b + c; 
}

double** inv(double** v){
    double a =  v[0][0] * abcd(v[1][1], v[1][2], v[2][1], v[2][2]); 
    double b = -v[0][1] * abcd(v[1][0], v[1][2], v[2][0], v[2][2]); 
    double c =  v[0][2] * abcd(v[1][0], v[1][1], v[2][0], v[2][1]); 
    double det = a + b + c; 
    double** o = matrix(3, 3); 
    if (det == 0){return o;}
    det = 1.0/det; 

    double** co = cof(v); 
    double** ct = T(co, 3, 3); 
    for (int x(0); x < 3; ++x){
        for (int y(0); y < 3; ++y){o[x][y] = ct[x][y]*det;}
    }
    clear(co, 3, 3); 
    clear(ct, 3, 3); 
    return o; 
}

double** unit(){
    double** m = matrix(3, 3);
    m[0][0] =  1; m[1][1] =  1; m[2][2] = -1;
    return m; 
}

double** smatx(double px, double py, double pz){
    double** o = matrix(3, 3);
    o[0][0] = -1; o[1][1] = -1;
    o[0][2] = px; o[1][2] = py; 
    o[2][2] = pz+1; 
    return o; 
}


void clear(double** mx, int row, int col){
    for (int x(0); x < row; ++x){free(mx[x]);}
    free(mx); 
}

void print(double** mx, int prec, int w){
    std::cout << std::fixed << std::setprecision(prec); 
    std::cout << std::setw(w) << mx[0][0] << " " << std::setw(w) << mx[0][1] << " " << std::setw(w) << mx[0][2] << "\n";
    std::cout << std::setw(w) << mx[1][0] << " " << std::setw(w) << mx[1][1] << " " << std::setw(w) << mx[1][2] << "\n";
    std::cout << std::setw(w) << mx[2][0] << " " << std::setw(w) << mx[2][1] << " " << std::setw(w) << mx[2][2] << "\n";
    std::cout << std::endl;
}


void print_(double** mx, int row, int col, int prec, int w){
    std::cout << std::fixed << std::setprecision(prec); 
    for (int x(0); x < row; ++x){
        for (int y(0); y < col; ++y){std::cout << std::setw(w) << mx[x][y] << " ";}
        std::cout << "\n"; 
    }
    std::cout << std::endl;
}







double** find_roots(double a, double b, double c){
    double s = a/3.0; 
    double p = b - a*a / 3.0; 
    double q = c - a*b / 3.0 + (2 * a * a * a) / 27.0; 
    double** sol = matrix(2, 3);
    if (fabs(p) < 1e-12 &&  fabs(q) < 1e-12){
        sol[0][0] = -s; sol[0][1] = -s; sol[0][2] = -s; 
        return sol; 
    }

    if (fabs(p) < 1e-12){
        std::complex<double> w0 = -q; 
        std::complex<double> om = std::complex<double>(-0.5,  0.5 * pow(3, 0.5)); 
        std::complex<double> o2 = std::complex<double>(-0.5, -0.5 * pow(3, 0.5)); 
        w0 = pow(w0, 1.0/3.0);

        std::complex<double> s1 = w0 - s; 
        std::complex<double> s2 = w0 * om - s; 
        std::complex<double> s3 = w0 * o2 - s; 
        sol[0][0] = s1.real(); sol[0][1] = s2.real(); sol[0][2] = s3.real(); 
        sol[1][0] = s1.imag(); sol[1][1] = s2.imag(); sol[1][2] = s3.imag(); 
        return sol; 
    }
    
    double u = -q / 2.0 + pow(q*q / 4.0 + p*p*p / 27.0, 0.5); 
    double f = (fabs(u) < 1e-12) ? 0.0 : pow(u, 1.0/3.0); 
    std::complex<double> w = (fabs(f) < 1e-12) ? 0.0 : f - p / (3.0*f); 
    std::complex<double> ds = w*w - 4.0*(w*w + p); 
    std::complex<double> w1 = (-w + pow(ds, 0.5))/2.0 - s; 
    std::complex<double> w2 = (-w - pow(ds, 0.5))/2.0 - s;
    w = w - s;  
    sol[0][0] = w.real(); sol[0][1] = w1.real(); sol[0][2] = w2.real(); 
    sol[1][0] = w.imag(); sol[1][1] = w1.imag(); sol[1][2] = w2.imag(); 
    return sol; 
} 


double** find_roots(double a, double b, double c, double d, double e){
    a = 1.0/a; b = b*a; c = c*a; d = d*a; e = e*a; a = 1.0; 
    
    double p = c - (3.0 * b*b) / 8.0; 
    double q = d - (b * c) / 2.0 + pow(b, 3) / 8.0; 
    double r = e - (3.0 * b*b*b*b) / 256.0 + (b * b * c) / 16.0 - (b * d)/4.0; 
    double s = b / 4.0; 
   
    // row 0: real, row 1: imag 
    double** sol = matrix(2, 4);  
    if (fabs(q) < 1e-12){
        std::complex<double> disc = p * p - 4 * r; 
        std::complex<double> z0 = pow((-p + pow(disc, 0.5)) / 2.0, 0.5);
        std::complex<double> z1 = pow((-p - pow(disc, 0.5)) / 2.0, 0.5); 
        std::complex<double> s1 = ( z0 - s); 
        std::complex<double> s2 = (-z0 - s); 
        std::complex<double> s3 = ( z1 - s); 
        std::complex<double> s4 = (-z1 - s); 
        sol[0][0] = s1.real(); sol[0][1] = s2.real(); sol[0][2] = s3.real(); sol[0][3] = s4.real(); 
        sol[1][0] = s1.imag(); sol[1][1] = s2.imag(); sol[1][2] = s3.imag(); sol[1][3] = s4.imag(); 
        return sol; 
    }

    double** sl = find_roots(2.0*p, p*p-4.0*r, -q*q); 
    std::complex<double> w0 = std::complex(sl[0][0], sl[1][0]); 
    std::complex<double> r0 = pow(w0, 0.5); 
    std::complex<double> d1 = pow(r0*r0 - 2.0*(p + w0 - q / r0), 0.5); 
    std::complex<double> d2 = pow(r0*r0 - 2.0*(p + w0 + q / r0), 0.5); 

    std::complex<double> s1 = (-r0 + d1)/2.0 - s; 
    std::complex<double> s2 = (-r0 - d1)/2.0 - s; 
    std::complex<double> s3 = ( r0 + d2)/2.0 - s; 
    std::complex<double> s4 = ( r0 - d2)/2.0 - s; 

    sol[0][0] = s1.real(); sol[0][1] = s2.real(); sol[0][2] = s3.real(); sol[0][3] = s4.real(); 
    sol[1][0] = s1.imag(); sol[1][1] = s2.imag(); sol[1][2] = s3.imag(); sol[1][3] = s4.imag(); 
    return sol; 
}

double** get_intersection_angle(double** H1, double** H2){
    auto solx =[](
            double r, double i, 
            double** r0, double** r1, double** r2, double** r3,
            double w00, double w01, double w10, double w11, 
            double h10, double h11, double h12, 
            double h20, double h21, double h22,
            double r00, double r11
    ) -> bool {
        if (-1 > r || r > 1 || i){return false;}
        double x = 1 - r*r;
        double vp = pow(x, 0.5); 
        double vn = -vp; 

        double x1p = w00*r + w01*vp + r00; 
        double y1p = w10*r + w11*vp + r11; 

        double x1n = w00*r + w01*vn + r00; 
        double y1n = w10*r + w11*vn + r11; 
        
        bool p = fabs(x1p*x1p + y1p*y1p - 1) < 1e-6;
        bool n = fabs(x1n*x1n + y1n*y1n - 1) < 1e-6;
        p *= fabs(h10*x1p + h11*y1p + h12 - (h20*r + h21*vp + h22)) < 1e-6; 
        n *= fabs(h10*x1n + h11*y1n + h12 - (h20*r + h21*vn + h22)) < 1e-6; 
        if (!p && !n){return false;}
        double a1 = std::atan2(y1p, x1p); 
        double a3 = std::atan2(y1n, x1n); 
        double a2 = std::atan2(vp, r); 
        double a4 = std::atan2(vn, r);  
        if (p){*r0 = new double(a1); *r1 = new double(a2);}
        if (n){*r2 = new double(a3); *r3 = new double(a4);} 
        return true; 
    }; 



    double d1 = H2[0][2] - H1[0][2]; 
    double d2 = H2[1][2] - H1[1][2]; 

    double x1 = H1[0][0]*H1[1][1] - H1[0][1]*H1[1][0]; 
    double x2 = H2[0][0]*H2[1][1] - H2[0][1]*H2[1][0]; 
    if (fabs(x1) < 1e-6 || fabs(x2) < 1e-6){return nullptr;}
    x1 = 1.0/x1; x2 = 1.0/x2; 
    double h1_00 =  H1[1][1]*x1; double h1_01 = -H1[0][1]*x1; 
    double h1_10 = -H1[1][0]*x1; double h1_11 =  H1[0][0]*x1; 
    double w00 = h1_00*H2[0][0] + h1_01 * H2[1][0]; double w01 = h1_00*H2[0][1] + h1_01 * H2[1][1]; 
    double w10 = h1_10*H2[0][0] + h1_11 * H2[1][0]; double w11 = h1_10*H2[0][1] + h1_11 * H2[1][1]; 
    double r00 = h1_00*d1 + h1_01*d2;               double r11 = h1_10*d1 + h1_11*d2; 
    double p  = w01*w01 + w11*w11 - w00*w00 - w10*w10; 
    double tv = r00*r00 + r11*r11 - 1 + w00*w00 + w10*w10; 

    double q  = 2*(w00*w01 + w10*w11); 
    double rv = 2*(w00*r00 + w10*r11); 
    double sv = 2*(w01*r00 + w11*r11); 

    double a = p*p + q*q;
    double b = 2*(q*sv - p*rv); 
    double c = sv*sv - 2*p*(p+tv) + rv*rv - q*q; 
    double d = 2*(rv * (p+tv)-q*sv); 
    double e = pow(p+tv, 2) - sv*sv;
    double** roots = find_roots(a, b, c, d, e); 
    
    double** vou = matrix(8, 2); 
    for (int x(0); x < 4; ++x){
        double* r1 = nullptr; double* r2 = nullptr; 
        double* r3 = nullptr; double* r4 = nullptr; 
        bool p = solx(
            roots[0][x], roots[1][x], &r1, &r2, &r3, &r4, w00, w01, w10, w11, 
            H1[2][0], H1[2][1], H1[2][2], H2[2][0], H2[2][1], H2[2][2], r00, r11
        ); 
        if (r1 && r2){vou[x  ][0] = *r1; vou[x  ][1] = *r2;}
        if (r3 && r4){vou[x+4][0] = *r3; vou[x+4][1] = *r4;}
        if (r1){delete r1;}
        if (r2){delete r2;}
        if (r3){delete r3;}
        if (r4){delete r4;}
    }
    clear(roots, 2, 4);
    return vou; 
}


double find_real_eigenvalue(double** M, double* rx){
    double a = -(M[0][0] + M[1][1] + M[2][2]);
    double b = M[0][0]*M[1][1] + M[0][0]*M[2][2] + M[1][1]*M[2][2] - M[0][1]*M[1][0] - M[0][2]*M[2][0] - M[1][2]*M[2][1];
    double c = -det(M);
    double** r = find_roots(a, b, c);
    for (int i(0); i < 3; ++i){
        if (fabs(r[1][i])){continue;} 
        *rx = r[0][i];
        clear(r, 2, 3); 
        return true;
    }
    clear(r, 2, 3);
    return false;
}




// Factor degenerate conic
void multisqrt(double y, double roots[2], int *count) {
    *count = 0;
    if (y < 0) return;
    if (fabs(y) < 0) {roots[0] = 0; *count = 1; return;}
    double r = pow(y, 0.5);
    roots[0] = -r; roots[1] = r;
    *count = 2;
}

void swap_index(double** v, int idx){
    double tmp = v[idx][0];  
    v[idx][0] = v[idx][1]; 
    v[idx][1] = tmp; 
}

void factor_degenerate(double** G, double** lines, int* lc, double* q0) {
    if (fabs(G[0][0]) == 0 && fabs(G[1][1]) == 0) {
        lines[0][0] = G[0][1]; lines[0][1] = 0;       lines[0][2] = G[1][2];
        lines[1][0] = 0;       lines[1][1] = G[0][1]; lines[1][2] = G[0][2] - G[1][2];
        *lc = 2; *q0 = 0;
        return;
    }
    int swapxy = (fabs(G[0][0]) > fabs(G[1][1]));
    double** Q = scale(G, 1); 
    for (int i(0); i < 3*swapxy; i++){
        double tmp = Q[0][i];
        Q[0][i] = Q[1][i];
        Q[1][i] = tmp;
    }
    for (int j(0); j < 3*swapxy; j++){swap_index(Q, j);}
    double** Q_ = scale(Q, 1.0/Q[1][1]);
    double** D_ = cof(Q_);
    clear(Q, 3, 3);
    double q22 = -D_[2][2]; 

    int r_count;
    double r[2];
    if (q22 < 0){
        multisqrt(-D_[0][0], r, &r_count);
        for (int i(0); i < r_count; ++i, ++(*lc)) {
            lines[*lc][0] = Q_[0][1]; 
            lines[*lc][1] = Q_[1][1];
            lines[*lc][2] = Q_[1][2] + r[i];
            if (!swapxy){continue;}
            swap_index(lines, *lc); 
        }
    } 
    else {
        double x0 = D_[0][2] / -q22; double y0 = D_[1][2] / -q22;
        multisqrt(q22, r, &r_count);
        for (int i(0); i < r_count; ++i, ++(*lc)) {
            lines[*lc][0] =  Q_[0][1] + r[i];
            lines[*lc][1] =  Q_[1][1], 
            lines[*lc][2] = -Q_[1][1]*y0 - (Q_[0][1] + r[i])*x0;
            if (!swapxy){continue;}
            swap_index(lines, *lc); 
        }
    }
    clear(D_, 3, 3); 
    clear(Q_, 3, 3);  
    *q0 = q22;
}




int intersections_ellipse_line(double** ellipse, double* line, double** points, double zero){
    double** C = matrix(3, 3);
    for (int i(0); i < 3; ++i){
        C[0][i] = line[1] * ellipse[2][i] - line[2] * ellipse[1][i];
        C[1][i] = line[2] * ellipse[0][i] - line[0] * ellipse[2][i];
        C[2][i] = line[0] * ellipse[1][i] - line[1] * ellipse[0][i];
    }


    double a = -(C[0][0] + C[1][1] + C[2][2]);
    double b =   C[0][0] * C[1][1] + C[0][0] * C[2][2] + C[1][1] * C[2][2] 
               - C[0][1] * C[1][0] - C[0][2] * C[2][0] - C[1][2] * C[2][1];

    double** cf = cof(C); 

    print(cf); 


    double** eigenvals = find_roots(a, b, -det(C));
    print_(eigenvals, 2, 3); 

    int point_count = 0;
    for (int i(0); i < 3; ++i) {
        if (fabs(eigenvals[1][i]) > 0){continue;}
        double vec[3] = {
            cf[0][0] * eigenvals[0][i] + cf[0][1] + cf[0][2],
            cf[1][0] + cf[1][1] * eigenvals[0][i] + cf[1][2],
            cf[2][0] + cf[2][1] + cf[2][2] * eigenvals[0][i]
        };
        
        if (!fabs(vec[2])){continue;}
        double v[3] = {vec[0]/vec[2], vec[1]/vec[2], 1.0};
        //double line_val    = dot_product(line, v);
        //double ellipse_val = dot_product(v, (Vector3){
        //    ellipse.data[0][0]*v.x + ellipse.data[0][1]*v.y + ellipse.data[0][2]*v.z,
        //    ellipse.data[1][0]*v.x + ellipse.data[1][1]*v.y + ellipse.data[1][2]*v.z,
        //    ellipse.data[2][0]*v.x + ellipse.data[2][1]*v.y + ellipse.data[2][2]*v.z
        //});
        //double error = line_val*line_val + ellipse_val*ellipse_val;
        //if (error < zero) {
        //    points[point_count++] = v;
        //    if (point_count >= 2) break;
        //}
    }
    return point_count;
}








void intersection_ellipses(double** A, double** B, double eps){
    double** A_ = nullptr; double** B_ = nullptr; 
    bool swp = fabs(det(B)) > fabs(det(A)); 
    if (swp){A_ = B; B_ = A;}
    else {A_ = A; B_ = B;}
    
    double** AT = inv(A_);
    double** t  = dot(AT, B_); 
    clear(AT, 3, 3);
    double e = 0; 
    if (!find_real_eigenvalue(t, &e)){return;}
    clear(t, 3, 3);   
  
    int lc = 0; double q0;  
    double** line = matrix(2, 3); 
    double** G = arith(B_, A_, -e); 
    factor_degenerate(G, line, &lc, &q0); 
    clear(G, 3, 3); 
  
    std::cout << "+> " << lc << std::endl;
    int pc = 0;
    double** all_points = matrix(4, 3); 
    for (int i(0); i < lc; ++i) {
        double** pts = matrix(2, 3);
        int count = intersections_ellipse_line(A_, line[i], pts, eps);
//        for (int j(0); j < count; ++j, ++pc){all_points[pc] = pts[j];}
    }





}



