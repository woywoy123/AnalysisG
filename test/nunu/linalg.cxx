#include "geometry.h"
#include "linalg.h"
#include "nusol.h"

Vec3::Vec3(double x, double y, double z): 
    x(x), y(y), z(z) {
}

Vec3 Vec3::operator+(const Vec3& rhs) const {
    return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
}

Vec3 Vec3::operator-(const Vec3& rhs) const {
    return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
}

Vec3 Vec3::operator*(double scalar) const {
    return Vec3(x * scalar, y * scalar, z * scalar);
}

Vec3 Vec3::operator/(double scalar) const {
    return Vec3(x / scalar, y / scalar, z / scalar);
}

double Vec3::dot(const Vec3& rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
}

Vec3 Vec3::cross(const Vec3& rhs) const {
    return Vec3(
            y * rhs.z - z * rhs.y, 
            z * rhs.x - x * rhs.z, 
            x * rhs.y - y * rhs.x
    );
}

double Vec3::norm() const {
    return std::sqrt(x * x + y * y + z * z);
}

double Vec3::normSquared() const {
    return x * x + y * y + z * z;
}

bool Vec3::operator<(const Vec3& rhs) const {
    if (std::abs(x - rhs.x) > 1e-9) { return x < rhs.x; }
    if (std::abs(y - rhs.y) > 1e-9) { return y < rhs.y; }
    if (std::abs(z - rhs.z) > 1e-9) { return z < rhs.z; }
    return false;
}

Vec3 Vec3::normalize() const {
    double n = norm();
    if (n > 1e-9) { return Vec3(x / n, y / n, z / n); }
    return Vec3();
}

Vec4::Vec4(double px, double py, double pz, double e) :
    px(px), py(py), pz(pz), e(e) {}

Vec3 Vec4::pvec() const {return Vec3(px, py, pz);}

double Vec4::p() const {
    return std::sqrt(px * px + py * py + pz * pz);
}

double Vec4::beta() const {
    return this -> e > 0 ? this -> p() / this -> e : 0;
}

double Vec4::phi() const {
    return std::atan2(py, px);
}

double Vec4::theta() const {
    double p_mag = p();
    return p_mag > 0 ? std::acos(pz / p_mag) : 0;
}

double Vec4::mass() const {
    double m_sq = this -> mass2();
    return m_sq > 0 ? std::sqrt(m_sq) : 0.0;
}

double Vec4::mass2() const {
    double o = 0; 
    o += this -> px * this -> px; 
    o += this -> py * this -> py;
    o += this -> pz * this -> pz;
    return this -> e * this -> e - o; 
}


Mat3::Mat3() {
    for (size_t i(0); i < 3; ++i) {
        for (size_t j(0); j < 3; ++j) {
            data[i][j] = (i == j);
        }
    }
}
Mat3::Mat3(const std::vector<std::vector<double>>& d) {
    for (size_t i(0); i < 3; ++i) {
        for (size_t j(0); j < 3; ++j) {
            data[i][j] = d[i][j];
        }
    }
}
Vec3 Mat3::operator*(const Vec3& v) const {
    return Vec3(
        data[0][0] * v.x + data[0][1] * v.y + data[0][2] * v.z,
        data[1][0] * v.x + data[1][1] * v.y + data[1][2] * v.z,
        data[2][0] * v.x + data[2][1] * v.y + data[2][2] * v.z
    );
}
Mat3 Mat3::operator*(const Mat3& rhs) const {
    Mat3 result;
    for (size_t i(0); i < 3; ++i) {
        for (size_t j(0); j < 3; ++j) {
            result.data[i][j] = 0;
            for (size_t k(0); k < 3; ++k) {
                result.data[i][j] += data[i][k] * rhs.data[k][j];
            }
        }
    }
    return result;
}
Vec3 Mat3::get_col(int j) const {
    return Vec3(data[0][j], data[1][j], data[2][j]);
}
Mat3 Mat3::transpose() const {
    Mat3 t;
    for (size_t i(0); i < 3; ++i) {
        for (size_t j(0); j < 3; ++j) {
            t.data[i][j] = data[j][i];
        }
    }
    return t;
}

double calculate_distance_squared(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets,
    std::vector<Vec3>& out_points
) {
    MultiShapeConnector connector;
    for (size_t i(0); i < leptons.size(); ++i) {
        double mt = mass_params[i * 2];
        double mw = mass_params[i * 2 + 1];
        NuSol solver(bjets[i], leptons[i], mt, mw);
        std::pair<bool, Mat3> H_res = solver.getH();
        if (!H_res.first) { return -1.0; }
        connector.addEllipse(H_res.second);
    }
    connector.compute(out_points);
    if (out_points.size() < 2) { return 0.0; }
    Vec3 centroid;
    for (size_t i(0); i < out_points.size(); ++i) {
        centroid = centroid + out_points[i];
    }
    centroid = centroid / out_points.size();
    double d_sq_sum = 0.0;
    for (size_t i(0); i < out_points.size(); ++i) {
        d_sq_sum += (out_points[i] - centroid).normSquared();
    }
    return d_sq_sum;
}

std::vector<double> calculate_gradient_analytical(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets
) {
    std::vector<double> grad(mass_params.size(), 0.0);
    std::vector<Vec3> final_points;
    double d2_base = calculate_distance_squared(
        mass_params, leptons, bjets, final_points
    );
    if (d2_base < 0 || final_points.size() < 2) { return grad; }

    Vec3 centroid;
    for (size_t i(0); i < final_points.size(); ++i) {
        centroid = centroid + final_points[i];
    }
    centroid = centroid / final_points.size();

    std::vector<Vec3> d_d2_dp;
    for (size_t i(0); i < final_points.size(); ++i) {
        d_d2_dp.push_back((final_points[i] - centroid) * 2.0);
    }

    for (size_t i(0); i < leptons.size(); ++i) {
        NuSol solver(bjets[i], leptons[i], mass_params[2 * i], mass_params[2 * i + 1]);
        std::pair<bool, std::pair<Mat3, Mat3>> dH_res = solver.getH_derivatives();
        if (!dH_res.first) { continue; }

        Mat3 dH_dmt = dH_res.second.first;
        Mat3 dH_dmw = dH_res.second.second;

        Vec3 dp_dmt = dH_dmt.get_col(2);
        Vec3 dp_dmw = dH_dmw.get_col(2);

        grad[2 * i] = d_d2_dp[i].dot(dp_dmt);
        grad[2 * i + 1] = d_d2_dp[i].dot(dp_dmw);
    }
    return grad;
}
