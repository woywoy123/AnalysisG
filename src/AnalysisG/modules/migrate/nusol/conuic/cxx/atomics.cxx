#include <conuic/constants.h>
#include <conuic/branches.h>
#include <conuic/atomics.h>
#include <iostream>
#include <iomanip> 
                   
void debug_s(std::string name, long double val){
    std::cout << std::fixed << std::setprecision(12);
    std::cout << name << ": " << val << " | ";
}

void debug_s(std::string name, long double val, bool branch){
    if (branch){name += " (+)";}
    else {name += " (-)";}
    std::cout << std::fixed << std::setprecision(12);
    std::cout << name << ": " << val << " | ";
}

void debug_s(branches_t val){
    debug_s(val.name, val.p, true);
    debug_s(val.name, val.m, false); 
    std::cout << std::endl;
}

void debug_s(std::string name, long double val, long double truth, long double tol){
    long double lm = std::abs(val) + std::abs(truth)*0.5;
    lm = (std::abs(val) - std::abs(truth))/lm; 

    std::cout << std::fixed << std::setprecision(18);
    if (lm > tol){std::cout << name << "(!): " << val << " | " << truth << " | " << lm << std::endl;}
    else {        std::cout << name << "(+): " << val << " | " << truth << " | " << lm << std::endl;}
} 

long double  signs(int sign, long double  v1, long double  v2){return (sign > 0) ? v1 : v2;}
long double* signs(int sign, long double* v1, long double* v2){return (sign > 0) ? v1 : v2;}
angular_t*   signs(int sign, angular_t* v1, angular_t* v2){return (sign > 0) ? v1 : v2;}
angular_t    signs(int sign, angular_t  v1, angular_t  v2){return (sign > 0) ? v1 : v2;}
matrix_t*    signs(int sign,  matrix_t* v1,  matrix_t* v2){return (sign > 0) ? v1 : v2;}

