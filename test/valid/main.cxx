#include <iostream>
#include <vector>
#include "structs.h"
#include "nusol_ref.h"
#include "nusol.h"

reference_t test_ref(std::vector<particle> bjets, std::vector<particle> muons, int x, int y){
    double mT = 172.68; 
    double mW = 80.385; 

    std::cout << "------- test reference ---------" << std::endl; 
    nusol nu1(bjets[x], muons[y]); 
    reference_t nu = nu1.update(mT, mW); 
    print(nu.x0p  );
    print(nu.x0   );
    print(nu.Sx   );
    print(nu.Sy   );
    print(nu.x1   );
    print(nu.y1   );
    print(std::pow(nu.Z2, 0.5));
    matrix m1 = nu.HT; 
    matrix h1 = nu.H; 
    m1.print();
    h1.print(); 
    return nu; 
}





int main(){
    std::vector<particle> muons = {
        {-14.306, -47.019,   3.816,  49.299},
        { 14.979,  42.693, -79.570,  91.534},
        { 60.236,  69.537, 166.586, 190.302},
        { 20.575,  27.615, 100.242, 294.501}
    };

    std::vector<particle> bjets = {
        {-19.766, -40.022,  69.855,  83.191},
        {-23.487, 116.748, -64.443, 136.770},
        {114.379, -48.805, 167.815, 209.192},
        { 19.069, -58.705, -10.629,  62.940}
    };

    // [-4965.17737922   542.98711237 -8820.4039176 ] -1073245.8078694863
    double t1 = 0.933180786795;
    double z1 = 98.52560; 

    nusol_rev nu_1(bjets[1], muons[3]); 
    geo_t geo1 = nu_1.geometry(t1, z1); 
    
    nusol_rev nu_2(bjets[2], muons[3]); 
    geo_t geo2 = nu_2.geometry(t1, z1); 

    nusol_rev nu_3(bjets[1], muons[1]); 
    geo_t geo3 = nu_3.geometry(t1, z1); 


    nu_1.export_ellipse("1", t1, z1); 
    nu_1.export_ellipse("1", t1, z1, 100); 

    nu_2.export_ellipse("2", t1, z1); 
    nu_2.export_ellipse("2", t1, z1, 100); 

    nu_3.export_ellipse("3", -0.1, 100); 
    nu_3.export_ellipse("3", -0.1, 100, 100); 
 

    //geo_t ix = nu_1.intersection(&nu_2, t1, z1, t1, z1); 
    //print(ix.nu1 -> _p1);
    //print(ix.nu1 -> _p2); 
    //print(ix.nu2 -> _p1);
    //print(ix.nu2 -> _p2); 
    //ix.r0.print();
    //ix.d.print(); 
    //ix.nu1 -> _pts1.print(); 
    //ix.nu1 -> _pts2.print(); 
    //ix.nu2 -> _pts1.print(); 
    //ix.nu2 -> _pts2.print(); 

    //reference_t nu_r = test_ref(bjets, muons, 1, 3); 
    //rev_t tr  = nu_2.translate(nu_r.Sx, nu_r.Sy); 
    //mass_t ms = nu_2.masses(0.9, 98.0); 

   

 
   





    return 0;
}


