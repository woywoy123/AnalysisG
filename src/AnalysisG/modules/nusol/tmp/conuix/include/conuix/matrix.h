#ifndef MATRIX_H
#define MATRIX_H

struct matrix_t {
    matrix_t(int _r = 3, int _c = 3); 
    ~matrix_t(); 

    matrix_t dot(const matrix_t& o); 
    matrix_t T() const; 

    const double& at(int _r, int _c) const; 
    double& at(int _r, int _c); 

    matrix_t& operator=(const matrix_t& other);
    matrix_t  operator+(const matrix_t& o) const;
    matrix_t  operator-(const matrix_t& o) const;
    matrix_t  operator*(double s         ) const;

    void print(int p); 

    int r = -1, c = -1;
    double** data = nullptr; 
}; 


#endif
