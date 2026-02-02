#ifndef H_MULTISOL_MATRIX
#define H_MULTISOL_MATRIX

#include "multisol/solvers.h"

struct matrix_t {
    public:
        ~matrix_t();
        matrix_t(int _r = 3, int _c = 3); 
        matrix_t(const matrix_t& other); 
        
        matrix_t cat(const matrix_t& o); 
        matrix_t clone(); 
        matrix_t clone() const; 

        matrix_t T() const;
        matrix_t dot(const matrix_t& o);

        matrix_t inv(); 
        matrix_t adj(); 
        matrix_t cofactor(); 
        matrix_t minor(); 
        matrix_t minor(int row, int col); 
        matrix_t cross(const matrix_t* r1); 
        matrix_t cross(matrix_t* r1, matrix_t* r2); 

        matrix_t nullspace(long double eps = 1e-12); 

        matrix_t eigenvector(); 
        matrix_t eigenvector(long double lbd, long double eps = 1e-12); 
        roots_t  eigenvalues(); 

        long double det(); 

        long double trace(); 
        long double& at(int _r, int _c);
        const long double& at(int _r, int _c) const;
        matrix_t at(int _r); 

        matrix_t& operator=(const matrix_t& other);

        matrix_t operator+(const matrix_t& o);
        matrix_t operator+(const matrix_t& o) const;

        matrix_t operator-(const matrix_t& o);
        matrix_t operator-(const matrix_t& o) const;

        matrix_t operator*(long double s);
        matrix_t operator*(long double s) const;

        void print(int p = 9);
        void print(int p = 9) const;

        int r = -1; 
        int c = -1;

    private:
        long double** data = nullptr; 
}; 


#endif
