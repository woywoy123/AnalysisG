#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cmath>

class Vec2 {
public:
    double x, y;
    Vec2(double x_val = 0, double y_val = 0) : x(x_val), y(y_val) {}
};

class Vec3 {
    public:
        double x, y, z;
        Vec3(double x = 0, double y = 0, double z = 0);
        Vec3 operator+(const Vec3& rhs) const;
        Vec3 operator-(const Vec3& rhs) const;
        Vec3 operator*(double scalar)   const;
        Vec3 operator/(double scalar)   const;
        bool operator<(const Vec3& rhs) const;

        Vec3 cross(const Vec3& rhs)     const;
        Vec3 normalize()                const;

        double norm()                   const;
        double normSquared()            const;
        double dot(const Vec3& rhs)     const;
};

class Vec4 {
    public:
        double px, py, pz, e;
        Vec4(double px = 0, double py = 0, double pz = 0, double e = 0);
        Vec3 pvec()    const;
        double p()     const;
        double phi()   const;
        double beta()  const;
        double theta() const;
        double mass()  const;
        double mass2() const; 
};

class Mat3 {
    public:
        double data[3][3];
        Mat3();
        Mat3(const std::vector<std::vector<double>>& d);
        Vec3 operator*(const Vec3& v) const;
        Mat3 operator*(const Mat3& rhs) const;
        Vec3 get_col(int j) const;
        Mat3 transpose() const;
};

double calculate_distance_squared(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets,
    std::vector<Vec3>& out_points
);

std::vector<double> calculate_gradient_analytical(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets
);

#endif
