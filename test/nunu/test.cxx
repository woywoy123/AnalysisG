#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <iostream>

#include "linalg.h"
#include "nusol.h"
#include "geometry.h"
//#include "ellipse.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    std::vector<Mat3> ellipses; 
    std::vector<double> masses = {172.6, 80.4, 172.6, 80.4, 172.6, 80.4};


    for (size_t i = 0; i < leptons.size(); ++i) {
        std::cout << "--- Analyzing Event " << i+1 << " ---" << std::endl;
        NuSol solver(bjets[i], leptons[i], masses[2*i], masses[2*i+1]);

        // --- GOAL 1: Check for and get the physical Z²=0 boundary ---
        std::cout << "\n1. Analyzing the physical boundary (Z² = 0):" << std::endl;
        std::vector<std::pair<double, double>> boundary_points = solver.get_boundary_points();
        
        if (boundary_points.empty()) {
            std::cout << "   -> No real solution exists for the Z²=0 boundary." << std::endl;
        } else {
            std::cout << "   -> Found " << boundary_points.size() << " points on the Z²=0 boundary." << std::endl;
            // Optionally print a few points
            // std::cout << "      Example point: (λ_T=" << boundary_points[0].first << ", λ_W=" << boundary_points[0].second << ")" << std::endl;
        }

        // --- GOAL 2: Get the contour line for the nominal masses ---
        std::cout << "\n2. Analyzing the contour for nominal masses (Z² = " << solver.getZ2() << "):" << std::endl;
        
        // Get the Z² value at the nominal masses to define the target contour
        double z2_target = solver.getZ2();
        if (z2_target < 0) {
            std::cout << "   -> Cannot generate contour; Z² at nominal masses is negative." << std::endl;
        } else {
            std::vector<std::pair<double, double>> contour_points = solver.get_contour_points(z2_target);
            if (contour_points.empty()){
                 std::cout << "   -> Could not generate contour points." << std::endl;
            } else {
                std::cout << "   -> Found " << contour_points.size() << " points on the Z²=" << z2_target << " contour." << std::endl;
                // std::cout << "      Example point: (λ_T=" << contour_points[0].first << ", λ_W=" << contour_points[0].second << ")" << std::endl;
            }
        }
        std::cout << "\n" << std::endl;
    }

//    for (size_t i(0); i < leptons.size(); ++i){
//        NuSol solver(bjets[i], leptons[i], masses[2*i], masses[2*i+1]);
//        std::cout << "Z2 (original): " << solver.getZ2() << std::endl;
//        solver.get_boundary(); 
//        ellipses.push_back(std::get<1>(solver.getH())); 
//    }
//
    //EllipseSystem system(leptons, bjets, masses);
    //system.analyze_system();

    return 0;
}
