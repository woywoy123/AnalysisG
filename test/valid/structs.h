#ifndef STRUCT_H
#define STRUCT_H

class particle; 

template <typename T>
void safe(T** o){
    if (!*o){return;}
    delete *o; *o = nullptr;
}; 

struct vec3 {
    double x = 0, y = 0, z = 0;
    vec3 operator+(const vec3& o) const; 
    vec3 operator-(const vec3& o) const;
    vec3 operator*(double s) const;
    vec3 cross(const vec3& o) const;
    double dot(const vec3& o) const;
    double mag2() const;
    double mag() const; 
    void print() const; 
};

struct vec4 {
    double x = 0, y = 0, z = 0, w = 0;
    vec4 operator+(const vec4& o) const;
    vec4 operator-(const vec4& o) const;
    vec4 operator*(double s) const;
    double dot(const vec4& o) const;
    void print();
};


struct matrix {
    matrix(int r=3, int c=3);
    matrix(const matrix& other);
    ~matrix();

    double mag() const; 
    static matrix I(int size);
    matrix T() const;
    matrix inverse();

    matrix& operator=(const matrix& other);
    matrix  operator+(const matrix& o) const;
    matrix  operator-(const matrix& o) const;
    matrix  operator*(const matrix& o) const;
    matrix  operator*(double s       ) const;
    vec4    operator*(const vec4& v  ) const; 
    vec3    operator*(const vec3& v  ) const;

    void print(int p = 5);
    double& at(int r, int c); 
    const double& at(int r, int c) const; 
    int rows() const { return this -> r; }
    int cols() const { return this -> c; }

    double** data = nullptr;
    int r = 0, c = 0;
};

double cos_theta(const particle* b, const particle* mu); 
void print(double v, int p = 12); 

#endif
