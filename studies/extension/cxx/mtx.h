#ifndef H_MTX
#define H_MTX 
#include <cmath>

class mtx; 
mtx operator*(double scale , const mtx& o2); 
mtx operator*(const mtx& o1, double scale); 
mtx operator*(const mtx& o1, const mtx& o2); 
mtx operator+(const mtx& o1, const mtx& o2); 
mtx operator-(const mtx& o1, const mtx& o2);

mtx* find_roots(double a, double b, double c); 
mtx* find_roots(double a, double b, double c, double d); 
mtx* find_roots(double a, double b, double c, double d, double e); 







class mtx {
    public: 
        mtx(); 
        mtx(int idx, int idy); 
        mtx(mtx* in); 
        mtx(mtx& in); 
        ~mtx(); 

        double trace(); 
        double det(); 

        mtx  copy(); 
        void copy(const mtx* ipt, int idx, int idy = -1); 
        void copy(const mtx* ipt, int idx, int jdy, int idy); 
        bool assign(int idx, int idy, double val, bool valid = true); 
        bool unique(int id1, int id2, double v1, double v2); 

        mtx* cat(const mtx* v2); 
        mtx* slice(int idx); 
        mtx* eigenvalues(); 
        mtx* eigenvector(); 

        mtx T();
        mtx inv(); 
        mtx cof(); 
        mtx diag(); 

        mtx dot(const mtx& other); 
        mtx dot(const mtx* other); 

        mtx cross(mtx* r1); 
        mtx cross(mtx* r1, mtx* r2); 

        mtx Rz(double angle);
        mtx Ry(double angle);
        mtx Rx(double angle); 

        bool valid(int idx, int idy); 
        void print(int prec = 6, int width = 12); 

        mtx& operator=(const mtx& o); 
        double** _m = nullptr; 
        bool  ** _b = nullptr; 

        double m_00();
        double m_01();
        double m_02();
        double m_10();
        double m_11();
        double m_12();
        double m_20();
        double m_21();
        double m_22();

        int dim_i = 0; 
        int dim_j = 0; 
        double tol = 1e-9; 

    private:
        friend mtx operator*(double scale , const mtx& o2); 
        friend mtx operator*(const mtx& o1, double scale); 
        friend mtx operator+(const mtx& o1, const mtx& o2); 
        friend mtx operator-(const mtx& o1, const mtx& o2);
        friend mtx operator*(const mtx& o1, const mtx& o2); 

        void scale(double** v, int idx, int idy, double s); 

        double trace(double** A);
        double m_00(double**  M);
        double m_01(double**  M);
        double m_02(double**  M);
        double m_10(double**  M);
        double m_11(double**  M);
        double m_12(double**  M);
        double m_20(double**  M);
        double m_21(double**  M);
        double m_22(double**  M);
        double det(double**   v);
        void   _copy(double*  dst, double* src, int lx); 
        void   _copy(bool*    dst,   bool* src, int lx); 
        void   _copy(double** src, int lx, int ly); 
        void   _copy(bool**   src, int lx, int ly); 

        double** _matrix(int row, int col); 
        bool**   _mask(int row, int col); 

}; 






#endif
