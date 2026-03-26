#include <conuic/atomics.h>
#include <conuic/data.h>
#include <math.h>

long double convert(double v){return (long double)v;}
long double convert(int v){return (long double)v;}

long double mag2(kinematics_t* v1, kinematics_t* v2){
    long double o = 0;
    o += v1 -> px * v2 -> px;
    o += v1 -> py * v2 -> py;
    o += v1 -> pz * v2 -> pz;
    return o; 
}

long double costh(kinematics_t* v1, kinematics_t* v2){
    long double v11 = mag2(v1,v1);
    long double v22 = mag2(v2,v2);
    long double v12 = mag2(v1, v2);
    return  v12 / std::sqrt(v11 * v22); 
}


long double tn_cos(long double tn){return 1.0L / std::sqrt(1.0L + tn * tn);}
long double tn_sin(long double tn){return   tn / std::sqrt(1.0L + tn * tn);}

long double omega(kinematics_t* jx, kinematics_t* lx, int sign){
    long double r = lx -> b / jx -> b; 
    long double c = costh(jx, lx);
    long double s = std::sqrt((1.0L - c) *(1.0L + c)); 
    return (1.0L / s) * (convert(sign) * r - c); 
}

long double Omega(kinematics_t* jx, kinematics_t* lx, int sign){
    long double w = omega(jx, lx, sign);
    return std::sqrt(w * w + (1.0L - lx -> b) * (1.0L + lx -> b)); 
}

long double Gamma(branches_t* plus, branches_t* minus, int sign){
    long double n = plus -> tpsi + convert(sign) * minus -> tpsi;
    long double O = (sign > 0) ? plus -> O : minus -> O; 
    return n / (O*O); 
}

long double delta(branches_t* plus, branches_t* minus, int sign){
    long double bl = plus -> bl; 
    long double O2 = plus -> O * minus -> O; 
    long double w2 = plus -> tpsi * minus -> tpsi; 
    long double n = ((1.0L - bl) * ( 1.0L + bl ) - w2 + convert(sign) * O2); 
    return - plus -> tth * n * 0.5L; 
}

long double lm_dt(delta_t* dt, int sign){
    long double da = dt -> alpha_m;
    long double sc = (sign > 0) ? std::cos(da) : std::sin(da); 
    return convert(sign) * (sc * sc) / (dt -> calp * dt -> calm); 
}

// ------------------------------------------------------------------ //
long double SigmasE(delta_t* dt, branches_t* br, int sign){
    long double dl = (sign > 0) ? dt -> dp : dt -> dm; 
    return br -> cpsi - dl * br -> spsi;
}

long double LambdaE(delta_t* dt, branches_t* br, int sign){
    long double dl = (sign > 0) ? dt -> dp : dt -> dm; 
    return br -> spsi + dl * br -> cpsi;
}

long double m_nuG(
        delta_t* dt, branches_t* br, kinematics_t* kl, 
        long double tau, long double phi, int sign, int eps
){
    long double dl = (sign > 0) ? dt -> dp : dt -> dm; 
    long double Sig = SigmasE(dt, br, sign); 
    long double Lmb = LambdaE(dt, br, sign); 
    long double O = br -> O; 
    long double w = br -> w;

    long double ml = kl -> m;  
    long double el = kl -> e; 
    
    hyper_t hx(tau);  
    angular_t ax(phi); 
    long double d1 = (dl * br -> tpsi * el * el - ml * ml); 
    long double d2 = (kl -> p * Lmb * hx.sinh * ax.cos - convert(eps) * O * Sig * hx.cosh  * el); 
    return d1 / d2 ;  
}



