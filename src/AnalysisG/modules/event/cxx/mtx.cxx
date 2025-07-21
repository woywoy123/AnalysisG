#include <templates/solvers.h>
#include <templates/mtx.h>
#include <iostream>
#include <iomanip>

mtx::mtx(){}
mtx::mtx(int idx, int idy){
    this -> _m = _matrix(idx, idy);
    this -> _b = _mask(idx, idy); 
    this -> dim_i = idx; this -> dim_j = idy; 
}

mtx::mtx(mtx* in){
    this -> _m = in -> _m; 
    this -> _b = in -> _b; 
    this -> dim_i = in -> dim_i;
    this -> dim_j = in -> dim_j; 
    in -> _m = nullptr;     
    in -> _b = nullptr; 
}

mtx::mtx(mtx& in){
    this -> _m = in._m; 
    this -> _b = in._b;
    in._m = nullptr;
    in._b = nullptr; 
    this -> dim_i = in.dim_i;
    this -> dim_j = in.dim_j; 
}

mtx::~mtx(){
    if (this -> _m){
        for (int x(0); x < this -> dim_i; ++x){free(this -> _m[x]);}
        free(this -> _m); 
        this -> _m = nullptr; 
    }
    if (this -> _b){
        for (int x(0); x < this -> dim_i; ++x){free(this -> _b[x]);}
        free(this -> _b); 
        this -> _b = nullptr; 
    }
}


void mtx::print(int prec, int width){
    if (!this -> _m){std::cout << "null pointer error" << std::endl; return;}
    std::cout << std::fixed << std::setprecision(prec); 
    for (int x(0); x < this -> dim_i; ++x){
        for (int y(0); y < this -> dim_j; ++y){
            std::cout << std::setw(width) << this -> _m[x][y] << " ";
        }
        std::cout << "\n"; 
    }
    std::cout << std::endl;
}


mtx mtx::T(){
    mtx out(this -> dim_j, this -> dim_i); 
    double** vo = out._m; 
    bool**   vb = out._b; 
    for (int x(0); x < this -> dim_j; ++x){
        for (int y(0); y < this -> dim_i; ++y){
            vo[x][y] = this -> _m[y][x];
            vb[x][y] = this -> _b[y][x]; 
        }
    }
    return out; 
}

mtx mtx::dot(const mtx* other){
    return this -> dot(*other);
}

mtx mtx::dot(const mtx& other){
    mtx out = mtx(this -> dim_i, other.dim_j); 
    for (int x(0); x < this -> dim_i; ++x){
        for (int j(0); j < other.dim_j; ++j){
            double sm = 0; 
            for (int y(0); y < other.dim_i; ++y){sm += this -> _m[x][y] * other._m[y][j];}
            out._m[x][j] = sm; 
        }
    }
    return out;  
}

mtx mtx::cross(mtx* r1){
    mtx vXc(3, 3); 
    for (int i(0); i < this -> dim_j; ++i){
        vXc.assign(0, i, r1 -> _m[0][1] * this -> _m[2][i] - r1 -> _m[0][2] * this -> _m[1][i]);
        vXc.assign(1, i, r1 -> _m[0][2] * this -> _m[0][i] - r1 -> _m[0][0] * this -> _m[2][i]);
        vXc.assign(2, i, r1 -> _m[0][0] * this -> _m[1][i] - r1 -> _m[0][1] * this -> _m[0][i]);
    }
    return vXc; 
}

mtx mtx::cross(mtx* r1, mtx* r2){
    mtx vx(1, 3); 
    vx.assign(0, 0, r1 -> _m[0][1] * r2 -> _m[0][2] - r1 -> _m[0][2] * r2 -> _m[0][1]);
    vx.assign(0, 1, r1 -> _m[0][2] * r2 -> _m[0][0] - r1 -> _m[0][0] * r2 -> _m[0][2]);
    vx.assign(0, 2, r1 -> _m[0][0] * r2 -> _m[0][1] - r1 -> _m[0][1] * r2 -> _m[0][0]);
    return vx; 
}

mtx mtx::inv(){
    auto inv3x3 =[this]() -> mtx{
        double det_ = this -> det();
        det_ = (!det_) ? 0.0 : 1.0/det_; 
        return det_*this -> cof().T();
    }; 
    if (this -> dim_i == 3 && this -> dim_j == 3){return inv3x3();}
    return mtx(this -> dim_i, this -> dim_j); 
}

mtx mtx::diag(){
    mtx o(this -> dim_i, this -> dim_i); 
    for (int x(0); x < this -> dim_i; ++x){o.assign(x, x, 1);}
    return o; 
}



mtx* mtx::eigenvector(){
    auto mag =[this](mtx* r) -> double{
        double m = 0; 
        for (int i(0); i < 3; ++i){m += r -> _m[0][i]*r -> _m[0][i];}
        return std::pow(m, 0.5);
    }; 
    double q = this -> m_00() + this -> m_11() + this -> m_22(); 
    mtx* eig = solve_cubic(1.0, -this -> trace(), q, -this -> det()); 
    mtx* eiv = new mtx(3, 3); 
    for (int i(0); i < 3; ++i){
        if (fabs(eig -> _m[1][i]) >= eig -> tol){continue;}
        double e = eig -> _m[0][i]; 
        mtx B = this -> copy() - this -> diag()*e; 

        mtx r1(1, 3);
        r1.assign(0, 0, B._m[0][0]); 
        r1.assign(0, 1, B._m[0][1]); 
        r1.assign(0, 2, B._m[0][2]); 

        mtx r2(1, 3);
        r2.assign(0, 0, B._m[1][0]); 
        r2.assign(0, 1, B._m[1][1]); 
        r2.assign(0, 2, B._m[1][2]); 
        mtx cx = this -> cross(&r1, &r2); 
        if (mag(&cx) < eig -> tol){
            mtx r3(1, 3);
            r3.assign(0, 0, B._m[2][0]); 
            r3.assign(0, 1, B._m[2][1]); 
            r3.assign(0, 2, B._m[2][2]); 
            cx = this -> cross(&r1, &r3); 
        }
        double mag_ = mag(&cx);
        if (mag_ > eig -> tol){cx = cx * (1.0/mag_);}
        eiv -> assign(i, 0, cx._m[0][0]); 
        eiv -> assign(i, 1, cx._m[0][1]); 
        eiv -> assign(i, 2, cx._m[0][2]); 
    }
    delete eig; 
    return eiv; 
}

mtx* mtx::eigenvalues(){
    double a = - this -> trace(); 
    double b =   this -> m_00() + this -> m_11() + this -> m_22(); 
    double c = - this -> det(); 
    return solve_cubic(1.0, a, b, c);  
}

bool mtx::valid(int idx, int idy){
    if (idx >= this -> dim_i || idy >= this -> dim_j){return false;}
    return this -> _b[idx][idy]; 
}

bool mtx::assign(int idx, int idy, double val, bool valid){
    if (this -> dim_i < idx || this -> dim_j < idy){return false;}
    this -> _m[idx][idy] = val;
    this -> _b[idx][idy] = valid; 
    return true;
}

bool mtx::unique(int id1, int id2, double v1, double v2){
    int idx = 0;  
    for (int x(0); x < this -> dim_j; ++x){
        bool sm = true; 
        sm *= pow(this -> _m[id1][x] - v1, 2) < this -> tol;
        sm *= pow(this -> _m[id2][x] - v2, 2) < this -> tol;
        sm *= this -> _b[id1][x]; 
        if (sm){return true;}
        idx += _b[id1][x]; 
        if (this -> _b[id1][x]){continue;}
        break; 
    }
    this -> _m[id1][idx] = v1; this -> _b[id1][idx] = true;
    this -> _m[id2][idx] = v2; this -> _b[id2][idx] = true; 
    return false;
}


mtx* mtx::slice(int idx){
    mtx* ox = new mtx(1, this -> dim_j); 
    _copy(ox -> _m[0], this -> _m[idx], this -> dim_j); 
    return ox; 
}

mtx* mtx::cat(const mtx* v2){
    int lx = this -> dim_i + v2 -> dim_i; 

    int i(0); 
    mtx* vo = new mtx(lx, this -> dim_j); 
    for (int x(0); x < lx; ++x, ++i){
        const mtx* vx = (x < this -> dim_i) ? this : v2; 
        i = (i < this -> dim_i) ? i : 0; 
        vo -> copy(vx, x, i , this -> dim_j); 
    }
    return vo; 
}

mtx mtx::copy(){
    mtx out(this -> dim_i, this -> dim_j); 
    for (int x(0); x < this -> dim_i; ++x){
        for (int y(0); y < this -> dim_j; ++y){
            out._m[x][y] = this -> _m[x][y]; 
            out._b[x][y] = this -> _b[x][y]; 
        }
    }
    return out; 
}

void mtx::copy(const mtx* ipt, int idx, int idy){
    if (idy < 0){idy = ipt -> dim_j;}
    for (int x(0); x < idy; ++x){
        this -> _m[idx][x] = ipt -> _m[idx][x];
        this -> _b[idx][x] = ipt -> _b[idx][x]; 
    }
}


void mtx::copy(const mtx* ipt, int idx, int jdx, int idy){
    if (idy < 0){idy = ipt -> dim_j;}
    for (int x(0); x < idy; ++x){
        this -> _m[idx][x] = ipt -> _m[jdx][x];
        this -> _b[idx][x] = ipt -> _b[jdx][x]; 
    }
}


mtx& mtx::operator=(const mtx& o){
    if (this == &o){return *this;}
    this -> dim_i = o.dim_i; 
    this -> dim_j = o.dim_j; 
    if (!this -> _m){this -> _m = _matrix(this -> dim_i, this -> dim_j);}
    if (!this -> _b){this -> _b = _mask(this -> dim_i, this -> dim_j);}

    _copy(this -> _m, o._m, o.dim_i, o.dim_j); 
    _copy(this -> _b, o._b, o.dim_i, o.dim_j); 
    return *this; 
}


mtx mtx::cof(){
    mtx out(this -> dim_i, this -> dim_j); 
    out._m[0][0] =   this -> m_00(); 
    out._m[1][0] = - this -> m_10(); 
    out._m[2][0] =   this -> m_20();
    out._m[0][1] = - this -> m_01(); 
    out._m[1][1] =   this -> m_11(); 
    out._m[2][1] = - this -> m_21();
    out._m[0][2] =   this -> m_02(); 
    out._m[1][2] = - this -> m_12(); 
    out._m[2][2] =   this -> m_22();
    return out; 
}

double mtx::trace(){return _trace(this -> _m);}
double mtx::m_00(){return _m_00(this -> _m);}
double mtx::m_01(){return _m_01(this -> _m);}
double mtx::m_02(){return _m_02(this -> _m);}
double mtx::m_10(){return _m_10(this -> _m);}
double mtx::m_11(){return _m_11(this -> _m);}
double mtx::m_12(){return _m_12(this -> _m);}
double mtx::m_20(){return _m_20(this -> _m);}
double mtx::m_21(){return _m_21(this -> _m);}
double mtx::m_22(){return _m_22(this -> _m);}
double mtx::det(){ return  _det(this -> _m);}
