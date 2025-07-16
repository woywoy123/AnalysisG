#include "matrix.h"
#include "mtx.h"


double** matrix(int row, int col){
    double** mx = (double**)malloc(row*sizeof(double*));
    for (int x(0); x < row; ++x){mx[x] = (double*)malloc(col*sizeof(double));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = 0;}}
    return mx;  
}

void clear(double** mx, int row, int col){
    if (!mx){return;}
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


double** arith(double** v1, double** v2, double s, int idx, int idy){
    double** o = matrix(idx, idy); 
    for (int x(0); x < idx; ++x){for (int y(0); y < idy; ++y){o[x][y] = v1[x][y] + s*v2[x][y];}}
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
            //minor[i][j] = det(mnr);
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
    if (fabs(det) < 1e-10){clear(mirs, 4, 4); return out;}
    double inv_det = 1.0 / det;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            out[i][j] = sign * mirs[j][i] * inv_det;
        }
    }
    clear(mirs, 4, 4); 
    return out;
}








void swap_index(double** v, int idx){
    double tmp = v[idx][0];  
    v[idx][0] = v[idx][1]; 
    v[idx][1] = tmp; 
}

void multisqrt(double y, double roots[2], int *count){
    *count = 0;
    if (y < 0) return;
    if (!fabs(y)){roots[0] = 0; *count = 1; return;}
    double r = pow(y, 0.5);
    roots[0] = -r; roots[1] = r;
    *count = 2;
}

void factor_degenerate(mtx G, mtx* lines, int* lc, double* q0) {
    if (fabs(G._m[0][0]) == 0 && fabs(G._m[1][1]) == 0) {
        lines -> assign(0, 0, G._m[0][1]); 
        lines -> assign(0, 1, 0);       
        lines -> assign(0, 2, G._m[1][2]);
        lines -> assign(1, 0, 0);       
        lines -> assign(1, 1, G._m[0][1]); 
        lines -> assign(1, 2, G._m[0][2] - G._m[1][2]);
        *lc = 2; *q0 = 0;
        return;
    }
    mtx Q = G.copy(); 
    int swapxy = (fabs(G._m[0][0]) > fabs(G._m[1][1]));
    for (int i(0); i < 3*swapxy; i++){
        double tmp = Q._m[0][i];
        Q.assign(0, i, Q._m[1][i]);
        Q.assign(1, i, tmp);
    }

    for (int j(0); j < 3*swapxy; j++){swap_index(Q._m, j);}
    mtx Q_ = Q * (1.0/Q._m[1][1]);
    mtx D_ = Q_.cof();
    double  q22 = -D_._m[2][2]; 

    int r_count;
    double r[2];
    if (q22 < 0){
        multisqrt(-D_._m[0][0], r, &r_count);
        for (int i(0); i < r_count; ++i, ++(*lc)) {
            lines -> assign(*lc, 0, Q_._m[0][1]); 
            lines -> assign(*lc, 1, Q_._m[1][1]);
            lines -> assign(*lc, 2, Q_._m[1][2] + r[i]);
            if (!swapxy){continue;}
            swap_index(lines -> _m, *lc); 
        }
    } 
    else {
        double x0 = D_._m[0][2] / -q22; double y0 = D_._m[1][2] / -q22;
        multisqrt(q22, r, &r_count);
        for (int i(0); i < r_count; ++i, ++(*lc)) {
            lines -> assign(*lc, 0,  Q_._m[0][1] + r[i]); 
            lines -> assign(*lc, 1,  Q_._m[1][1]); 
            lines -> assign(*lc, 2, -Q_._m[1][1]*y0 - (Q_._m[0][1] + r[i])*x0);
            if (!swapxy){continue;}
            swap_index(lines -> _m, *lc); 
        }
    }
    *q0 = q22;
}

int intersections_ellipse_line(mtx* ellipse, mtx* line, mtx* pts){
    mtx* eign = ellipse -> cross(line).eigenvector(); 
    eign -> print(12, 15); 
    abort(); 
    int pt = 0;
    for (int i(0); i < 3; ++i){
        if (!eign -> valid(i, 1) || !eign -> _m[i][2]){continue;}
        mtx* sl = eign -> slice(i); 
        double z = 1.0/sl -> _m[0][2]; 
        pts -> assign(pt, 0, sl -> _m[0][0]*z); 
        pts -> assign(pt, 1, sl -> _m[0][1]*z);
        pts -> assign(pt, 2, sl -> _m[0][2]*z); 
        pts -> print(12, 15); 

        mtx v  = sl -> T(); 
        mtx l1 = line -> dot(v);
        mtx l2 = sl -> dot(ellipse).dot(v);
        mtx vl = l1*l1 + l2*l2; 
        pts -> assign(pt, 3, std::log10(vl._m[0][0])); 
        pts -> assign(pt, 4, 1.0); 
        delete sl; ++pt; 
    }
    delete eign; 
    delete line; 
    return pt;
}


int intersection_ellipses(mtx* A, mtx* B, mtx** lines, mtx** pts, mtx** sols){
    bool swp = fabs(B -> det()) > fabs(A -> det()); 
    int lc = 0; int lx = 0; double q0 = -1;  
    mtx A_ = (swp) ? B -> copy() : A -> copy(); 
    mtx B_ = (swp) ? A -> copy() : B -> copy(); 
    mtx t  = A_.inv().dot(B_); 

    mtx* ex = t.eigenvalues(); 
    double e = 0; 
    bool exf = false;
    for (int x(0); x < ex -> dim_j; ++x){
        if (!ex -> valid(0, x)){continue;}
        exf = true; e = ex -> _m[0][x];
        break;
    }
    delete ex; 
    if (!exf){return lx;}
    mtx G = B_ - e*A_;
    mtx line = mtx(2, 3); 

    factor_degenerate(G, &line, &lc, &q0); 
    mtx sol_ = mtx(1, 6); 
    mtx all_ = mtx(6, 3); 
    for (int i(0); i < lc; ++i){
        mtx _pts(3, 5); 
        for (int j(0); j < intersections_ellipse_line(&A_, line.slice(i), &_pts); ++j){
            if (!_pts._m[j][4]){continue;}
            all_.copy(&_pts, lx, j, 3); 
            sol_._m[0][lx] = _pts._m[j][3]; ++lx; 
        }
    }
    *lines = new mtx(line); 
    *sols  = new mtx(1, lx); 
    *pts   = new mtx(lx, 3); 
    (*sols) -> copy(&sol_, 0, 0, lx); 
    for (int x(0); x < lx; ++x){(*pts) -> copy(&all_, x);}
    return lx; 
}



