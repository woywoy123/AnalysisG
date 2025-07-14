#ifndef H_MATRIX 
#define H_MATRIX
#include "tools.h"

void _copy(double* dst, double* src, int lx); 
double** _copy(double** dst, double** src, int lx, int ly, bool del = false); 

void clear(double** mx, int row, int col); 
double** matrix(int row, int col); 

double trace(double** A); 
double  m_00(double** M); 
double  m_01(double** M); 
double  m_02(double** M); 
double  m_10(double** M); 
double  m_11(double** M); 
double  m_12(double** M); 
double  m_20(double** M); 
double  m_21(double** M); 
double  m_22(double** M); 
double   det(double** v); 

double** inv(double** v); 
double** inv4(double** v); 

double** cof(double** v); 
double** scale(double** v, double s); 
double** diag(double** v, int idx); 
double** scale(double** v, int idx, int idy, double s); 
double** arith(double** v1, double** v2, double s = 1, int idx = 3, int idy = 3); 

double** Rz(double angle); 
double** Ry(double angle); 
double** Rx(double angle); 
double** Rxyz(double** M, double alpha, double beta, double gamma); 

double** T(double** v1, int r = 3, int c = 3); 
double** dot(double** v1, double** v2, int r1 = 3, int c1 = 3, int r2 = 3, int c2 = 3); 
double** dot(double** v1, double** v2, bool del, int r1 = 3, int c1 = 3, int r2 = 3, int c2 = 3);

int intersection_ellipses(double** A, double** B, double*** lines, double*** pts, double*** sols); 

#endif
