#include "nunu.h"

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
    return mx;  
}

double** dot(double** v1, double** v2, int r1, int c1, int r2, int c2){
    double** vo = matrix(r1, c2); 
    for (int x(0); x < r1; ++x){
        for (int j(0); j < c2; ++j){
            for (int y(0); y < r2; ++y){
                vo[x][j] += v1[x][y] * v2[y][j]; 
            }
        }
    }
    return vo; 
}

double** T(double** v1, int r, int c){
    double** vo = matrix(r, c); 
    for (int x(0); x < r; ++x){
        for (int y(0); y < c; ++y){vo[y][x] = v1[x][y];}
    }
    return vo; 
}



void clear(double** mx, int row, int col){
    for (int x(0); x < row; ++x){free(mx[x]);}
    free(mx); 
}

void print(double** mx){
    std::cout << std::fixed << std::setprecision(5); 
    std::cout << std::setw(10) << mx[0][0] << " " << std::setw(10) << mx[0][1] << " " << std::setw(10) << mx[0][2] << "\n";
    std::cout << std::setw(10) << mx[1][0] << " " << std::setw(10) << mx[1][1] << " " << std::setw(10) << mx[1][2] << "\n";
    std::cout << std::setw(10) << mx[2][0] << " " << std::setw(10) << mx[2][1] << " " << std::setw(10) << mx[2][2] << "\n";
    std::cout << std::endl;
}








particle::particle(double px, double py, double pz, double e){
    this -> px = px; this -> py = py;
    this -> pz = pz; this -> e  = e; 
}
double particle::p(){return pow(this -> p2(), 0.5);}
double particle::p2(){return pow(this -> px, 2) + pow(this -> py, 2) + pow(this -> pz, 2);}
double particle::m2(){return pow(this -> e, 2) - this -> p2();}
double particle::beta(){return this -> p() / this -> e;}
double particle::phi(){return std::atan2(this -> py, this -> px);}
double particle::theta(){return std::atan2(pow(pow(this -> px, 2) + pow(this -> py, 2), 0.5), this -> pz);}
particle::~particle(){}; 








nusol::nusol(particle* b, particle* l, double mW, double mT){
    this -> b = b; 
    this -> l = l;
    this -> mw = mW; 
    this -> mt = mT; 
    this -> _s = sintheta(this -> b, this -> l); 
    this -> _c = costheta(this -> b, this -> l); 
}


double nusol::Sx(){
    double p_mu = this -> l -> p(); 
    double b_mu = this -> l -> beta(); 
    double m2_mu = this -> l -> m2(); 

    double x0 = (m2_mu - pow(this -> mw, 2))/(2*this -> l -> e); 
    double sx = (x0 * b_mu - p_mu * (1 - pow(b_mu, 2))) / pow(b_mu, 2); 
    return sx;
}

double nusol::Sy(){
    double x0p = -(pow(this -> mt, 2) - pow(this -> mw, 2) - this -> b -> m2())/ (2 * this -> b -> e); 
    return (x0p / this -> b -> beta() - this -> _c * this -> Sx()) / this -> _s; 
}

double nusol::w(){return (this -> l -> beta() / this -> b -> beta() - this -> _c)/ this -> _s;}
double nusol::om2(){return pow(this -> w(), 2) + 1 - pow(this -> l -> beta(), 2);}
double nusol::x1(){
    double sx = this -> Sx();
    double sy = this -> Sy(); 
    return sx - (sx + this -> w() * sy)/this -> om2();
}

double nusol::y1(){
    double sx = this -> Sx();
    double sy = this -> Sy(); 
    double w_ = this -> w(); 
    return sy - (sx + w_ * sy)*w_/this -> om2();
}


void nusol::Z2_coeff(double* A, double* B, double* C){
    double m2_m   = this -> l -> m2(); 
    double beta_m = this -> l -> beta();
    double beta_b = this -> b -> beta(); 
    double D1     = -(m2_m + pow(this -> mt, 2) - this -> b -> m2())/(2 * this -> b -> e * this -> _s * beta_b); 
    double D2     = -(this -> l -> e * beta_m / (this -> b -> e * beta_b) + this -> _c)/ this -> _s;  
    double w      = this -> w();
    double o2     = this -> om2(); 
    double P      = 1 - (1+w*D2)/o2; 
    *A = pow(P, 2) * o2 - pow(D2 - w, 2) + pow(beta_m, 2); 
    *B = 2*(-w*P*D1 - D1*(D2 - w) + beta_m * this -> l -> e); 
    *C = pow((-w * D1 / o2), 2)*o2 - pow(D1, 2) + m2_m; 
}


double nusol::Z2(){
    double A, B, C; 
    double sx = this -> Sx(); 
    this -> Z2_coeff(&A, &B, &C);
    return A*pow(sx, 2) + B*sx + C; 
}

double** nusol::H_tilde(){
    if (this -> h_tilde){return this -> h_tilde;}
    double z2 = this -> Z2(); 
    double** _matrix = matrix(3, 3); 
    _matrix[0][0] = pow(z2 / this -> om2(), 0.5); 
    _matrix[1][0] = this -> w() * pow(z2 / this -> om2(), 0.5); 
    _matrix[2][1] = pow(z2, 0.5); 
    _matrix[0][2] = this -> x1() - this -> l -> p(); 
    _matrix[1][2] = this -> y1(); 
    this -> h_tilde = _matrix; 
    return _matrix; 
}

double** nusol::R_T(){  
    if (this -> r_t){return this -> r_t;}
    double phi_mu = this -> l -> phi(); 
    double theta_mu = this -> l -> theta();

    double** rz = matrix(3, 3);
    rz[0][0] =  std::cos(-phi_mu); 
    rz[0][1] = -std::sin(-phi_mu); 
    rz[1][0] =  std::sin(-phi_mu); 
    rz[1][1] =  std::cos(-phi_mu); 
    rz[2][2] = 1.0;

    double** ry = matrix(3, 3);
    ry[0][0] = std::cos(0.5*M_PI - theta_mu); 
    ry[0][2] = std::sin(0.5*M_PI - theta_mu); 
    ry[1][1] = 1.0; 
    ry[2][0] = -std::sin(0.5*M_PI - theta_mu); 
    ry[2][2] =  std::cos(0.5*M_PI - theta_mu);

    double** bv = matrix(3, 1); 
    bv[0][0] = this -> b -> px;
    bv[1][0] = this -> b -> py;
    bv[2][0] = this -> b -> pz; 

    double** vz = dot(rz, bv, 3, 3, 3, 1); 
    double** vy = dot(ry, vz, 3, 3, 3, 1);
    double  psi = -atan2(vy[2][0], vy[1][0]); 

    double** rx = matrix(3, 3);
    rx[0][0] = 1.0; 
    rx[1][1] =  std::cos(psi); 
    rx[1][2] = -std::sin(psi); 
    rx[2][1] =  std::sin(psi);
    rx[2][2] =  std::cos(psi); 

    double** rxt = T(rx, 3, 3); 
    double** ryt = T(ry, 3, 3); 
    double** rzt = T(rz, 3, 3); 
    double** rxy = dot(ryt, rxt, 3, 3, 3, 3); 
    double** ryz = dot(rzt, rxy, 3, 3, 3, 3);  
    clear(rz, 3, 3); 
    clear(ry, 3, 3);
    clear(bv, 3, 1); 
    clear(vz, 3, 1); 
    clear(vy, 3, 1);
    clear(rx, 3, 3); 
    clear(rxt, 3, 3); 
    clear(ryt, 3, 3);  
    clear(rzt, 3, 3); 
    clear(rxy, 3, 3);
    this -> r_t = ryz; 
    return ryz; 
}

double** nusol::H(){
    if (this -> h){return this -> h;}
    double** rt = this -> R_T(); 
    double** ht = this -> H_tilde(); 
    this -> h = dot(rt, ht, 3, 3, 3, 3); 
    return this -> h; 
}


nusol::~nusol(){
    delete this -> b; 
    delete this -> l;
    if (this -> h_tilde){clear(this -> h_tilde, 3, 3);}
    if (this -> r_t){clear(this -> r_t, 3, 3);}
    if (this -> h){clear(this -> h, 3, 3);}
}

nunu::nunu(particle* b1, particle* b2, particle* l1, particle* l2){
    this -> nu1 = new nusol(b1, l1, 80.385, 172.62);
    this -> nu2 = new nusol(b2, l2, 80.385, 172.62);
}

void nunu::generate(){
    double** m = this -> nu1 -> H(); 
    print(m); 
}

nunu::~nunu(){
    delete this -> nu1; 
    delete this -> nu2; 
}

