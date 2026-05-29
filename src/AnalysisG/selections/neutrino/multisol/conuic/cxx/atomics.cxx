#include <conuic/atomics.h>
#include <math.h>

long double convert(double v){return (long double)v;}
long double nsqrt(long double v){return std::sqrt((1.0L - v) * (1.0L + v));}
long double psqrt(long double v){return std::sqrt((1.0L + v*v));}
long double tn_sin(long double v){return    v / psqrt(v);}
long double cs_sin(long double v){return        nsqrt(v);}
long double sn_cos(long double v){return        nsqrt(v);}
long double tn_cos(long double v){return 1.0L / psqrt(v);}
long double cs_tan(long double v){return        nsqrt(v);}
long double sn_tan(long double v){return    v / nsqrt(v);}

long double dots(kinematics_t* v1, kinematics_t* v2){
    long double o = 0; 
    o += v1 -> px * v2 -> px; 
    o += v1 -> py * v2 -> py; 
    o += v1 -> pz * v2 -> pz; 
    return o; 
}

long double costheta(kinematics_t* bq, kinematics_t* lp){
    long double v11 = dots(bq, bq);
    long double v22 = dots(lp, lp);
    long double v12 = dots(lp, bq);
    return v12 / std::sqrt(v11 * v22); 
}

long double pw(long double v, int mul){
    long double o = 1.0L; 
    for (int x(0); x < mul; ++x){o *= v;}
    return o; 
}

