#ifndef H_BRANCHES_CONUIC
#define H_BRANCHES_CONUIC

#include <conuic/angular.h>
#include <common/matrix.h>
#include <string>

struct branches_t {
    branches_t(); 
    branches_t(long double p_, long double m_, std::string name_); 
    branches_t(angular_t   p_, angular_t   m_, std::string name_); 

    ~branches_t(); 
    angular_t   Apair(int sign);  
    long double pair(int sign);  

    long double p = 0;
    long double m = 0;
    angular_t pA, mA; 

    std::string name = ""; 
};

#endif
