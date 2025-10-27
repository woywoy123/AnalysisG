#ifndef H_CONUIX_MATRIX
#define H_CONUIX_MATRIX

struct matrix_t {
    public:
        matrix_t(int _r = 3, int _c = 3); 
        ~matrix_t();
        matrix_t dot(const matrix_t& o);
        matrix_t T() const;
       
        long double& at(int _r, int _c);
        const long double& at(int _r, int _c) const;

        matrix_t& operator=(const matrix_t& other);
        matrix_t& operator+(const matrix_t& o) const;
        matrix_t& operator-(const matrix_t& o) const;
        matrix_t& operator*(long double s) const;

        void print(int p = 9);

    private:
        int r = -1; 
        int c = -1;
        long double** data = nullptr; 
}; 






#endif
