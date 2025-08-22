#include <reconstruction/odeRK.h>

ellipse_t operator+(const ellipse_t& a, const ellipse_t& b){
    ellipse_t out; 
    out.A = a.A + b.A; out.vA = a.vA + b.vA;  
    out.B = a.B + b.B; out.vB = a.vB + b.vB;  
    out.C = a.C + b.C; out.vC = a.vC + b.vC; 
    out.t = a.t + b.t; out.z  = a.z  + b.z;
    return out; 
}

ellipse_t operator*(const ellipse_t& a, double s){
    ellipse_t out; 
    out.A = a.A * s; out.vA = a.vA * s;
    out.B = a.B * s; out.vB = a.vB * s; 
    out.C = a.C * s; out.vC = a.vC * s; 
    out.t = a.t * s; out.z  = a.z  * s;
    return out; 
}

void ellipse_t::print(){
    std::cout << "..........." << std::endl; 
    this -> A.print(); 
    this -> B.print(); 
    this -> C.print();
    this -> vA.print(); 
    this -> vB.print(); 
    this -> vC.print(); 
}


odeRK::odeRK(std::vector<multisol*>* sols, vec3 met, int iter, double step){
    this -> _data = sols; 
    this -> met_ = met; 
    this -> itr  = iter; 
    this -> dt_  = step; 
    this -> nsx  = sols -> size();  
    this -> _state.resize(this -> nsx); 

    for (size_t i(0); i < sols -> size(); ++i){
        this -> _state[i].t = sols -> at(i) -> dp_dt(); 
        this -> _state[i].z = 1.0; 
    }
    this -> update_t(); 
}

odeRK::~odeRK(){}

void odeRK::update_t() {
    for (size_t i = 0; i < this -> nsx; ++i) {
        multisol* slx = this -> _data -> at(i);
        matrix H  = slx -> H(this -> _state[i].t, this -> _state[i].z);
        this -> _state[i].A  = vec3{ H.at(0, 0),  H.at(1, 0),  H.at(2, 0)};
        this -> _state[i].B  = vec3{ H.at(0, 1),  H.at(1, 1),  H.at(2, 1)};
        this -> _state[i].C  = vec3{ H.at(0, 2),  H.at(1, 2),  H.at(2, 2)};

        matrix dH = slx -> dHdt(this -> _state[i].t, this -> _state[i].z);
        this -> _state[i].vA = vec3{dH.at(0, 0), dH.at(1, 0), dH.at(2, 0)};
        this -> _state[i].vB = vec3{dH.at(0, 1), dH.at(1, 1), dH.at(2, 1)};
        this -> _state[i].vC = vec3{dH.at(0, 2), dH.at(1, 2), dH.at(2, 2)};
    }
}

void odeRK::solve(){
    double bst = std::numeric_limits<double>::max(); 
    matrix _phi(this -> nsx, 1); 
    matrix _scl(this -> nsx, 1); 
    std::vector<double> current_t; 
    for (size_t i(0); i < this -> nsx; ++i){current_t.push_back(this -> _state[i].t);}
    for (int s(0); s < this -> itr; ++s){
        this -> rk4(this -> dt_); 
        double r = this -> solve_z_phi();
       
        std::cout << "Iter " << s; 
        for (int x(0); x < this -> nsx; ++x){std::cout << std::string((!x) ? ", t: (" : "") << this -> _state[x].t << std::string((x == this -> nsx-1) ? ") " : ", ");}
        for (int x(0); x < this -> nsx; ++x){std::cout << std::string((!x) ? ", z: (" : "") << this -> _state[x].z << std::string((x == this -> nsx-1) ? ") " : ", ");}
        std::cout << std::endl;
    }
}



double odeRK::ghost_angle(int ni) {
    multisol* slx = this -> _data->at(ni);
    double t_i = this -> _state[ni].t;
    vec3 p_met_dir = (this -> met_ - slx -> center(t_i, this -> _state[ni].z));
    vec3 l_dr = slx -> H(t_i, this -> _state[ni].z).inverse() * p_met_dir;
    return std::fmod(std::atan2(l_dr.y, l_dr.x) + 2 * M_PI, 2 * M_PI);
}

double odeRK::residual(std::vector<double> wg, std::vector<double> phx){
    vec3 res{0, 0, 0};
    for (size_t i = 0; i < this -> nsx; ++i){res = res + this -> _data -> at(i) -> v(this -> _state[i].t, wg[i], phx[i]);}
    return (res - this -> met_).mag();
}


double odeRK::solve_z_phi(){
    matrix A(3, 3 * this -> nsx); 
    matrix b(3, 1); 
    b.at(0, 0) = this -> met_.x;
    b.at(1, 0) = this -> met_.y; 
    b.at(2, 0) = this -> met_.z; 

    // Fill matrix A and adjust vector b
    for (int i(0); i < this -> nsx; ++i) {
        const ellipse_t state = this -> _state[i]; 
        
        A.at(0, 3 * i    ) = state.A.x; A.at(1, 3 * i    ) = state.A.y; A.at(2, 3 * i    ) = state.A.z;
        A.at(0, 3 * i + 1) = state.B.x; A.at(1, 3 * i + 1) = state.B.y; A.at(2, 3 * i + 1) = state.B.z;
        A.at(0, 3 * i + 2) = state.C.x; A.at(1, 3 * i + 2) = state.C.y; A.at(2, 3 * i + 2) = state.C.z;
    }
    matrix A_T = A.T();  
    matrix AAT = A.dot(A_T); 
    matrix X   = A_T * (AAT.inverse() * b); 

    std::vector<double> weights(this -> nsx);
    std::vector<double> phis(this -> nsx);
    for (int i(0); i < this -> nsx; ++i) {
        double u = X.at(3*i  , 0);
        double v = X.at(3*i+1, 0);
        weights[i] = std::pow(u*u + v*v, 0.5);
        phis[i]    = std::atan2(v, u);
        this -> _state[i].z = weights[i]; 
    }
    
    return this -> residual(weights, phis);
}


std::vector<double> odeRK::plane_align(const std::vector<ellipse_t>& current_state) {

    std::vector<double> dt_ds(this -> nsx);

    for (int i = 0; i < this -> nsx; ++i) {
        multisol* slx = this -> _data -> at(i);
        double t_i = current_state[i].t;

        // --- 1. Get current geometric state from the multisol object ---
        vec3 C_i = slx -> center(t_i, this -> _state[i].z);
        vec3 n_i = slx -> normal(t_i, this -> _state[i].z);

        // --- 2. Calculate the signed distance from P_met to the plane ---
        double delta_i = n_i.dot(this -> met_ - C_i);

        // --- 3. Calculate the derivatives of the geometric state w.r.t. t_i ---
        matrix dH_dti = slx -> dHdt(t_i, this -> _state[i].z);

        // Derivative of the center vector (dC_i/dt_i) is the 3rd column of dH/dt
        vec3 dC_dti = {dH_dti.at(0, 2), dH_dti.at(1, 2), dH_dti.at(2, 2)};

        // For the normal's derivative, we need the derivatives of the axis vectors A and B
        vec3 A_i = current_state[i].A;
        vec3 B_i = current_state[i].B;
        vec3 dA_dti = {dH_dti.at(0, 0), dH_dti.at(1, 0), dH_dti.at(2, 0)};
        vec3 dB_dti = {dH_dti.at(0, 1), dH_dti.at(1, 1), dH_dti.at(2, 1)};

        // Use the product rule for cross products to get dn_i/dt_i
        vec3 dn_dti = dA_dti.cross(B_i) + A_i.cross(dB_dti);

        // --- 4. Calculate the total derivative of the distance, ∂δᵢ/∂tᵢ ---
        double d_delta_d_ti = dn_dti.dot(this -> met_ - C_i) - n_i.dot(dC_dti);

        // --- 5. Define the final update rule for t_i ---
        // This is the "velocity" for t_i that will cause delta_i to shrink.
        if (std::abs(d_delta_d_ti) < 1e-9){dt_ds[i] = 0;} 
        else {dt_ds[i] = -this -> dt_ * delta_i / d_delta_d_ti;}
    }
    return dt_ds;
}


std::vector<double> odeRK::plane_rk4(const std::vector<double>& t_initial) {
    // Note: The state (A, B, C vectors) must be updated before each derivative call.
    //this -> update_t(t_initial);
    std::vector<double> k1 = plane_align(this -> _state);

    std::vector<double> t_k2(this->nsx);
    for (size_t j = 0; j < nsx; ++j) t_k2[j] = t_initial[j] + 0.5 * this -> dt_ * k1[j];
    //this -> update_t(t_k2);
    std::vector<double> k2 = plane_align(this->_state);

    // k3
    std::vector<double> t_k3(this->nsx);
    for(size_t j = 0; j < nsx; ++j) t_k3[j] = t_initial[j] + 0.5 * this -> dt_ * k2[j];
    //this -> update_t(t_k3);
    std::vector<double> k3 = plane_align(this->_state);

    // k4
    std::vector<double> t_k4(this->nsx);
    for(size_t j = 0; j < nsx; ++j) t_k4[j] = t_initial[j] + this -> dt_ * k3[j];
    //this -> update_t(t_k4);
    std::vector<double> k4 = plane_align(this->_state);

    // Calculate the final updated t vector
    std::vector<double> t_final(this->nsx);
    for(size_t j = 0; j < nsx; ++j){t_final[j] = t_initial[j] + (this -> dt_ / 6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);}
    return t_final;
}


std::vector<ellipse_t> odeRK::derivative(const std::vector<ellipse_t>& dS){
    std::vector<ellipse_t> dx; dx.resize(this -> nsx); 
    for (int i(0); i < this -> nsx; ++i){
        const ellipse_t& cur = dS[i]; 

        matrix d2H = this -> _data -> at(i) -> d2Hdt2(cur.t, cur.z); 
        dx[i].vA = vec3{d2H.at(0, 0), d2H.at(1, 0), d2H.at(2, 0)}; 
        dx[i].vB = vec3{d2H.at(0, 1), d2H.at(1, 1), d2H.at(2, 1)}; 
        dx[i].vC = vec3{d2H.at(0, 2), d2H.at(1, 2), d2H.at(2, 2)}; 

        dx[i].A = cur.vA; 
        dx[i].B = cur.vB; 
        dx[i].C = cur.vC; 

        dx[i].t = 1;
        dx[i].z = 1; 
    }
    return dx; 
}

void odeRK::rk4(double dt){
    std::vector<ellipse_t> s1 = this -> _state; 
    std::vector<ellipse_t> k1 = this -> derivative(this -> _state);
    for (int i(0); i < this -> nsx; ++i){s1[i] = s1[i] + k1[i] * (0.5*dt);}

    std::vector<ellipse_t> s2 = this -> _state; 
    std::vector<ellipse_t> k2 = this -> derivative(s1);
    for (int i(0); i < this -> nsx; ++i){s2[i] = s2[i] + k2[i] * (0.5*dt);}

    std::vector<ellipse_t> s3 = this -> _state; 
    std::vector<ellipse_t> k3 = this -> derivative(s2);
    for (int i(0); i < this -> nsx; ++i){s3[i] = s3[i] + k3[i] * dt;}

    std::vector<ellipse_t> k4 = this -> derivative(s3);
    for (int x(0); x < this -> nsx; ++x){
        this -> _state[x] = this -> _state[x] + (k1[x] + k2[x] * 2.0 + k3[x] * 2.0 + k4[x]) * (dt / 6.0);
    }
}
