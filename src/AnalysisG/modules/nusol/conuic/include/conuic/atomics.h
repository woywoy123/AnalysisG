#ifndef H_ATOMICS_CONUIC
#define H_ATOMICS_CONUIC

#include <iostream>
#include <string> 

struct branches_t;
struct angular_t; 
struct matrix_t;

template <typename g>
void flush(g** data){
    if (!*data){return;}
    delete *data; *data = nullptr; 
}; 

void debug_s(std::string name, long double val, long double truth, long double tol); 
void debug_s(std::string name, long double val, bool branch); 
void debug_s(std::string name, long double val); 
void debug_s(branches_t val);

long double  signs(int sign, long double  v1, long double  v2);
long double* signs(int sign, long double* v1, long double* v2);
angular_t    signs(int sign, angular_t  v1, angular_t  v2); 
angular_t*   signs(int sign, angular_t* v1, angular_t* v2); 
matrix_t*    signs(int sign,  matrix_t* v1,  matrix_t* v2); 

#endif
