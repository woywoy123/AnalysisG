#ifndef H_NUSOL_SOLVERS
#define H_NUSOL_SOLVERS

class mtx; 
// -------- root finders ------- //
mtx* find_roots(double a, double b, double c); 
mtx* find_roots(double a, double b, double c, double d); 
mtx* solve_cubic(double a, double b, double c, double d); 
mtx* find_roots(double a, double b, double c, double d, double e); 

// -------- matrix generators ------- //
mtx* smatx(double px, double py, double pz); 
mtx* Rz(double angle); 
mtx* Ry(double angle); 
mtx* Rx(double angle); 
mtx* unit(); 

// -------- primitives --------- //
void _arith(double**  o, double** v2, double s, int idx, int idy); 
void _scale(double**  v, double**  f, int idx, int idy, double s); 

void _copy(double*  dst, double* src, int lx); 
void _copy(bool*    dst, bool*   src, int lx); 

void _copy(double** dst, double** src, int lx, int ly); 
void _copy(bool**   dst, bool**   src, int lx, int ly); 

double** _matrix(int row, int col);
bool**   _mask(int   row, int col); 

double _trace(double** A);
double _m_00(double**  M);
double _m_01(double**  M);
double _m_02(double**  M);
double _m_10(double**  M);
double _m_11(double**  M);
double _m_12(double**  M);
double _m_20(double**  M);
double _m_21(double**  M);
double _m_22(double**  M);
double _det (double**  M); 

// ------- operators ------- //
mtx operator*(double scale, const mtx& o2);
mtx operator*(const mtx& o1, double scale);
mtx operator*(const mtx& o1, const mtx& o2);
mtx operator+(const mtx& o1, const mtx& o2); 
mtx operator-(const mtx& o1, const mtx& o2);

// ------- conics ------- //
void swap_index(double** v, int idx); 
void multisqrt(double y, double roots[2], int *count); 
void factor_degenerate(mtx G, mtx* lines, int* lc, double* q0); 

int  intersections_ellipse_line(mtx* ellipse, mtx* line, mtx* pts); 
int  intersection_ellipses(mtx* A, mtx* B, mtx** lines, mtx** pts, mtx** sols); 
mtx* intersection_angle(mtx* H1, mtx* H2, mtx* MET, int* n_sols); 

// --------- misc -------- //
mtx    make_ellipse(mtx* H, double angle); 
double distance(mtx* H1, double a1, mtx* H2, double a2); 




#endif
