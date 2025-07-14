#include "matrix.h"

double trace(double** A){return A[0][0] + A[1][1] + A[2][2];}
double m_00(double**  M){return M[1][1] * M[2][2] - M[1][2] * M[2][1];}
double m_01(double**  M){return M[1][0] * M[2][2] - M[1][2] * M[2][0];}
double m_02(double**  M){return M[1][0] * M[2][1] - M[1][1] * M[2][0];}
double m_10(double**  M){return M[0][1] * M[2][2] - M[0][2] * M[2][1];}
double m_11(double**  M){return M[0][0] * M[2][2] - M[0][2] * M[2][0];}
double m_12(double**  M){return M[0][0] * M[2][1] - M[0][1] * M[2][0];}
double m_20(double**  M){return M[0][1] * M[1][2] - M[0][2] * M[1][1];}
double m_21(double**  M){return M[0][0] * M[1][2] - M[0][2] * M[1][0];}
double m_22(double**  M){return M[0][0] * M[1][1] - M[0][1] * M[1][0];}
double det(double**   v){return v[0][0] * m_00(v) - v[0][1] * m_01(v) + v[0][2] * m_02(v);}


double** matrix(int row, int col){
    double** mx = (double**)malloc(row*sizeof(double*));
    for (int x(0); x < row; ++x){mx[x] = (double*)malloc(col*sizeof(double));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = 0;}}
    return mx;  
}

void clear(double** mx, int row, int col){
    for (int x(0); x < row; ++x){free(mx[x]);}
    free(mx); 
}

void _copy(double* dst, double* src, int lx){
    for (int x(0); x < lx; ++x){dst[x] = src[x];}
}

double** _copy(double** dst, double** src, int lx, int ly, bool del){
    for (int x(0); x < lx; ++x){_copy(dst[x], src[x], ly);}
    if (del){clear(src, lx, ly);}
    return dst; 
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

double** dot(double** v1, double** v2, bool del, int r1, int c1, int r2, int c2){
    double** v = dot(v1, v2, r1, c1, r2, c2); 
    if (del){clear(v1, r1, c1);}
    return v; 
}

double** T(double** v1, int r, int c){
    double** vo = matrix(c, r); 
    for (int x(0); x < c; ++x){
        for (int y(0); y < r; ++y){vo[x][y] = v1[y][x];}
    }
    return vo; 
}


double** diag(double** v, int idx){
    double** o = matrix(idx, idx);
    for (int x(0); x < idx; ++x){o[x][x] = v[x][x];}
    return o; 
}

double** scale(double** v, double s){
    double** o = matrix(3, 3); 
    for (int x(0); x < 3; ++x){for (int y(0); y < 3; ++y){o[x][y] = v[x][y]*s;}}
    return o; 
}

double** scale(double** v, int idx, int idy, double s){
    for (int x(0); x < idx; ++x){for (int y(0); y < idy; ++y){v[x][y] = v[x][y]*s;}}
    return v; 
}

double** Rz(double angle){
    double** rz = matrix(3, 3); 
    rz[0][0] =  std::cos(angle); 
    rz[0][1] = -std::sin(angle); 
    rz[1][0] =  std::sin(angle); 
    rz[1][1] =  std::cos(angle); 
    rz[2][2] = 1.0;
    return rz; 
}


double** Rxyz(double** M, double alpha, double beta, double gamma){

    //ca, sa = np.cos(_alpha), np.sin(_alpha)
    //cg, sg = np.cos(_gamma), np.sin(_gamma)
    //cb, sb = np.cos(_beta) , np.sin(_beta)
    //
    //R = np.array([
    //    [cb*cg, sa*sb*cg - ca*sg, ca*sb*cg + sa*sg], 
    //    [cb*sg, sa*sb*sg + ca*cg, ca*sb*sg - sa*cg], 
    //    [-sb  , sa*cb           , ca*cb           ]
    //])
    //S = R.dot(

    double ca = std::cos(alpha); 
    double sa = std::sin(alpha); 
    double cg = std::cos(gamma); 
    double sg = std::sin(gamma); 
    double cb = std::cos(beta); 
    double sb = std::sin(beta); 

    double** R = matrix(3, 3); 
    R[0][0] = cb*cg; 
    R[1][0] = cb*sg; 
    R[2][0] = -sb; 

    R[0][1] = sa*sb*cg - ca*sg; 
    R[1][1] = sa*sb*sg + ca*cg; 
    R[2][1] = sa*cb; 

    R[0][2] = ca*sb*cg + sa*sg; 
    R[1][2] = ca*sb*sg - sa*cg; 
    R[2][2] = ca*cb;   
       
    double** RT = T(R); 
    RT = dot(RT, M, true);  
    RT = dot(RT, R, true); 
    clear(R, 3, 3); 
    return RT; 
}


double** Ry(double angle){
    double** ry = matrix(3, 3); 
    ry[0][0] = std::cos(angle); 
    ry[0][2] = std::sin(angle); 
    ry[1][1] = 1.0; 
    ry[2][0] = -std::sin(angle); 
    ry[2][2] =  std::cos(angle);
    return ry; 
}


double** Rx(double angle){
    double** rx = matrix(3, 3); 
    rx[0][0] = 1.0; 
    rx[1][1] =  std::cos(angle); 
    rx[1][2] = -std::sin(angle); 
    rx[2][1] =  std::sin(angle);
    rx[2][2] =  std::cos(angle); 
    return rx; 
}



double** arith(double** v1, double** v2, double s, int idx, int idy){
    double** o = matrix(idx, idy); 
    for (int x(0); x < idx; ++x){for (int y(0); y < idy; ++y){o[x][y] = v1[x][y] + s*v2[x][y];}}
    return o; 
}

void cross(double* vx, double r1[3], double r2[3]){
    vx[0] = r1[1] * r2[2] - r1[2] * r2[1];
    vx[1] = r1[2] * r2[0] - r1[0] * r2[2];
    vx[2] = r1[0] * r2[1] - r1[1] * r2[0];
}

double** cross(double** C, double* v){
    double** vXc = matrix(3, 3); 
    for (int i(0); i < 3; ++i){
        vXc[0][i] = v[1] * C[2][i] - v[2] * C[1][i];
        vXc[1][i] = v[2] * C[0][i] - v[0] * C[2][i];
        vXc[2][i] = v[0] * C[1][i] - v[1] * C[0][i];
    }
    return vXc; 
}

double _mag(double* v){ 
    double m = 0; 
    for (int i(0); i < 3; ++i){m += v[i]*v[i];}
    return pow(m, 0.5);
}

double** cof(double** v){
    double** ov = matrix(3, 3); 
    ov[0][0] =  m_00(v); ov[1][0] = -m_10(v); ov[2][0] =  m_20(v);
    ov[0][1] = -m_01(v); ov[1][1] =  m_11(v); ov[2][1] = -m_21(v);
    ov[0][2] =  m_02(v); ov[1][2] = -m_12(v); ov[2][2] =  m_22(v);
    return ov; 
}

double** inv(double** v){
    double det_ = det(v);
    det_ = (!det_) ? 0.0 : 1.0/det_; 
    double** co = cof(v); 
    double** ct = T(co, 3, 3); 
    double** o  = scale(ct, det_);  
    clear(co, 3, 3); clear(ct, 3, 3); 
    return o; 
}




double** minors(double** in) {
    double** minor = matrix(4, 4); 
    for (int i(0); i < 4; ++i) {
        for (int j(0); j < 4; ++j){
            double** mnr = matrix(3, 3); 
            int mr = 0;
            for (int k(0); k < 4; ++k) {
                if (k == i){continue;}
                int ml = 0;
                for (int l(0); l < 4; ++l) {
                    if (l == j){continue;}
                    mnr[mr][ml] = in[k][l];
                    ++ml;
                }
                ++mr;
            }
            minor[i][j] = det(mnr);
            clear(mnr, 3, 3);
        }
    }
    return minor; 
}


double** inv4(double** in){
    double** mirs = minors(in);
    double** out = matrix(4, 4); 

    double det = 0.0;
    for (int j = 0; j < 4; j++) {
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        det += in[0][j] * sign * mirs[0][j];
    }
    if (fabs(det) < 1e-10){return out;}
    double inv_det = 1.0 / det;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            out[i][j] = sign * mirs[j][i] * inv_det;
        }
    }

    return out;
}


bool find_real_eigenvalue(double** M, double* rx){
    double a = -trace(M); 
    double b = m_00(M) + m_11(M) + m_22(M); 
    double c = -det(M);
    clear(M, 3, 3);   

    int sx = 0; 
    double** r = find_roots(1.0, a, b, c, &sx);
    for (int i(0); i < sx; ++i){
        if (fabs(r[1][i])){continue;} 
        *rx = r[0][i];
        clear(r, 2, 3); 
        return true;
    }
    clear(r, 2, sx);
    return false;
}


void factor_degenerate(double** G, double** lines, int* lc, double* q0) {
    if (fabs(G[0][0]) == 0 && fabs(G[1][1]) == 0) {
        lines[0][0] = G[0][1]; lines[0][1] = 0;       lines[0][2] = G[1][2];
        lines[1][0] = 0;       lines[1][1] = G[0][1]; lines[1][2] = G[0][2] - G[1][2];
        *lc = 2; *q0 = 0;
        clear(G, 3, 3); 
        return;
    }
    double** Q = scale(G, 1); 

    int swapxy = (fabs(G[0][0]) > fabs(G[1][1]));
    for (int i(0); i < 3*swapxy; i++){
        double tmp = Q[0][i];
        Q[0][i] = Q[1][i];
        Q[1][i] = tmp;
    }
    for (int j(0); j < 3*swapxy; j++){swap_index(Q, j);}
    double** Q_ = scale(Q, 1.0/Q[1][1]);
    double** D_ = cof(Q_);
    double  q22 = -D_[2][2]; 

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
    clear(G, 3, 3); clear(D_, 3, 3);
    clear(Q, 3, 3); clear(Q_, 3, 3);
    *q0 = q22;
}

double** get_eigen(double** A){
    int sx = 0; 
    double q     = m_00(A) + m_11(A) + m_22(A); 
    double** eig = find_roots(1.0, -trace(A), q, -det(A), &sx); 
    double** eiv = matrix(3, 4); 
    for (int i(0); i < 3; ++i){
        if (fabs(eig[1][i]) >= 1e-9){continue;}
        eiv[i][3] = 1; 

        double B[3][3] = {{0}}; 
        for (int j(0); j < 3; ++j){_copy(B[j], A[j], 3);} 
        for (int j(0); j < 3; ++j){ B[j][j] -= eig[0][i];}

        double r1[3] = {B[0][0], B[0][1], B[0][2]};
        double r2[3] = {B[1][0], B[1][1], B[1][2]};

        cross(eiv[i], r1, r2); 
        if (_mag(eiv[i]) < 1e-9){
            double r3[3] = {B[2][0], B[2][1], B[2][2]};
            cross(eiv[i], r1, r3); 
        }
        double mag = _mag(eiv[i]);
        if (mag < 1e-9){continue;}
        eiv[i][0] = eiv[i][0]/mag; 
        eiv[i][1] = eiv[i][1]/mag; 
        eiv[i][2] = eiv[i][2]/mag;
    }
    clear(eig, 2, 3); 
    return eiv; 
}

int intersections_ellipse_line(double** ellipse, double* line, double** pts){
    double** C = cross(ellipse, line);
    double** eign = get_eigen(C);

    int pt = 0;
    for (int i(0); i < 3; ++i){
        if (!eign[i][3] || !eign[i][2]){continue;}

        double** v = matrix(3, 1); 
        for (int j(0); j < 3; ++j){v[j][0] = eign[i][j];}

        double** l1 = dot(&line, v, 1, 3, 3, 1); 
        double** l2 = dot(ellipse, v, 3, 3, 3, 1);
        double** lt = T(l2, 3, 1);
        double** l3 = dot(lt, v, 1, 3, 3, 1); 

        for (int j(0); j < 3; ++j){pts[pt][j] = eign[i][j]/eign[i][2];}

        pts[pt][3] = (std::log10(pow(l3[0][0], 2) + pow(l1[0][0], 2)));  
        pts[pt][4] = 1.0; ++pt; 

        clear(l1, 1, 1); clear(l2, 3, 1); clear(v, 3, 1); 
        clear(lt, 1, 3); clear(l3, 1, 1);
    }
    clear(eign, 3, 4); clear(C, 3, 3);
    return pt;
}


int intersection_ellipses(double** A, double** B, double*** lines, double*** pts, double*** sols){
    bool swp = fabs(det(B)) > fabs(det(A)); 
    int lc = 0; int lx = 0; double q0 = -1;  

    double** A_ = (swp) ? B : A; 
    double** B_ = (swp) ? A : B; 
    
    double** AT = inv(A_);
    double** t  = dot(AT, B_); 
    clear(AT, 3, 3);

    double e = 0; 
    if (!find_real_eigenvalue(t, &e)){return lx;}
    double** line = matrix(2, 3); 
    double** G = arith(B_, A_, -e); 
    factor_degenerate(G, line, &lc, &q0); 

    double** sol_    = matrix(1, 6); 
    double** all_pts = matrix(6, 3); 
    for (int i(0); i < lc; ++i){
        double** _pts = matrix(3, 5);
        for (int j(0); j < intersections_ellipse_line(A_, line[i], _pts); ++j){
            if (!_pts[j][4]){continue;}
            _copy(all_pts[lx], _pts[j], 3); 
            sol_[0][lx] = _pts[j][3]; ++lx; 
        }
        clear(_pts, 3, 5); 
    }
    double** a_pts = matrix(lx, 3);
    double** a_sol = matrix(1, lx); 
    for (int x(0); x < lx; ++x){
        _copy(a_pts[x], all_pts[x], 3); 
        a_sol[0][x] = sol_[0][x]; 
    }
    clear(all_pts, 6, 3); clear(sol_, 1, 6); 
    *pts = a_pts; *sols = a_sol; *lines = line; 
    return lx; 
}



