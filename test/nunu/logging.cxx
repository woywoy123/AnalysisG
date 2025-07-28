#include "logging.h"
#include "linalg.h"
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>

Logger::Logger(const std::string& fname, const std::string& felip, const std::vector<Mat3>* ellipses){
    this -> file    = new std::ofstream(fname);
    this -> file_el = new std::ofstream(felip); 
    this -> Hs      = ellipses; 
    *this -> file_el << "ellipse_index,point_index,x,y,z\n";
    this -> Ellipses(); 
}

Logger::~Logger(){
    this-> file -> close(); 
    this -> file_el -> close(); 
    delete this -> file_el; 
    delete this -> file;
}

void Logger::writeHeader(size_t n){
    *this -> file << "iteration,objective,centroid_x,centroid_y,centroid_z";
    for (size_t i = 0; i < n; i++){*this -> file << ",angle_" << i;}
    for (size_t i = 0; i < n; i++) {
        *this -> file << ",point_" << i;
        *this -> file << "_x,point_" << i;
        *this -> file << "_y,point_" << i << "_z";
    }
    *this -> file << "\n";
}

void Logger::log(
    int iter, double objective, 
    const Vec3& centroid, 
    const std::vector<double>& angles, 
    const std::vector<Vec3>& points
){
    if (this -> firstWrite) {
        this -> writeHeader(angles.size());
        this -> firstWrite = false;
    }
    
    *this -> file << iter << ","; 
    *this -> file << std::setprecision(15) << objective << ","; 
    *this -> file << centroid.x << "," << centroid.y << "," << centroid.z;
    for (double a : angles) {*this -> file << "," << a;}
    for (const Vec3& p : points){*this -> file << "," << p.x << "," << p.y << "," << p.z;}
    *this -> file << "\n";
    this -> file -> flush();  
}

void Logger::Ellipses(const std::vector<Mat3>* ellipse){
    if (ellipse){this -> Hs = ellipse;}
    if (!this -> Hs){return;}
    for (size_t i = 0; i < this -> Hs -> size(); i++) {
        for (int j = 0; j < this -> resolution; j++) {
            double theta = 2 * M_PI * j / this -> resolution;
            Vec3 u(std::cos(theta), std::sin(theta), 1.0);
            Vec3 p = this -> Hs -> at(i) * u;
            *this -> file_el << i << "," << j << "," << p.x << "," << p.y << "," << p.z << "\n";
        }
    }
    this -> Hs = nullptr; 
    this -> file_el -> flush();
}
