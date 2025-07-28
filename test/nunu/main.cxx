#include <vector>
#include <cmath>
#include <iostream>
#include "linalg.h"
#include "nusol.h"
#include "geometry.h"

int main() {
    std::vector<Vec4> leptons = {
        {14.9791, 42.6930, -79.5701, 91.5341},
        {60.2364, 69.5374,  166.586, 190.302},
        {20.5758, 27.6150,  100.242, 294.501}
    };
    std::vector<Vec4> bjets = {
        {-23.4876,  116.748, -64.4432, 136.770},
        { 114.379, -48.8050, 167.815 , 209.192},
        { 19.0699, -58.7056, -10.629 ,  62.940}
    };

    std::vector<double> masses = {172.6, 80.4, 172.6, 80.4, 172.6, 80.4};
    EllipseSystem system(leptons, bjets, masses);

    system.analyze_system();

    return 0;
}
