#include "geometry.h"
#include "logging.h"
#include "nusol.h"
#include <limits>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <string> 
#include <chrono> 
#include <thread> 
                  
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void write_ellipse_to_obj(const Mat3& H, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) { throw std::runtime_error("Cannot open file: " + filename); }

    Vec3 c = H.get_col(2);
    Vec3 v1 = H.get_col(0);
    Vec3 v2 = H.get_col(1);
    int resolution = 100;

    for (size_t i(0); i < resolution; ++i) {
        double angle = 2.0 * M_PI * i / resolution;
        Vec3 p = c + v1 * std::cos(angle) + v2 * std::sin(angle);
        file << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }

    file << "l ";
    for (size_t i(1); i <= resolution; ++i) { file << i << " "; }
    file << "1\n";
    file.close();
}

void ConvexHull3D::write_to_obj(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) { throw std::runtime_error("Cannot open file: " + filename); }
    for (size_t i(0); i < this->points.size(); ++i) {
        file << "v " << this->points[i].x << " "
             << this->points[i].y << " "
             << this->points[i].z << "\n";
    }
    for (size_t i(0); i < this->faces.size(); ++i) {
        file << "f " << this->faces[i].a + 1 << " "
             << this->faces[i].b + 1 << " "
             << this->faces[i].c + 1 << "\n";
    }
    file.close();
}


ConvexHull3D::Face::Face(int a, int b, int c) : a(a), b(b), c(c) {}
ConvexHull3D::ConvexHull3D(const std::vector<Vec3>& input_points) : points(input_points) {
    if (points.size() < 4) { return; }
    buildHull();
}

double ConvexHull3D::computeVolume() {
    if (faces.empty()) { return 0.0; }
    double v = 0.0;
    for (size_t i(0); i < faces.size(); ++i) {
        v += (points[faces[i].a].cross(points[faces[i].b])).dot(points[faces[i].c]);
    }
    return std::abs(v) / 6.0;
}

void ConvexHull3D::computeCentroid() {
    centroid = Vec3(0, 0, 0);
    for (size_t i(0); i < points.size(); ++i) {
        centroid = centroid + points[i];
    }
    if (points.empty()) { return; }
    centroid = centroid / points.size();
}

void ConvexHull3D::orientFace(Face& f) {
    f.normal = (points[f.b] - points[f.a]).cross(points[f.c] - points[f.a]);
    if (f.normal.dot(points[f.a] - centroid) >= 0) { return; }
    std::swap(f.b, f.c);
    f.normal = f.normal * -1;
}

void ConvexHull3D::buildHull() {
    std::vector<Vec3> unique_points;
    for (size_t p_idx(0); p_idx < points.size(); ++p_idx) {
        bool found = false;
        for (size_t up_idx(0); up_idx < unique_points.size(); ++up_idx) {
            if ((points[p_idx] - unique_points[up_idx]).normSquared() < 1e-12) {
                found = true;
                break;
            }
        }
        if (!found) { unique_points.push_back(points[p_idx]); }
    }

    if (unique_points.size() < 4) { return; }
    points = unique_points;
    computeCentroid();

    int i0 = 0;
    int i1 = findFarthestPoint(i0);
    int i2 = findFarthestFromLine(i0, i1);
    int i3 = findFarthestFromPlane(i0, i1, i2);
    if (i3 == -1) { return; }

    faces.push_back(Face(i0, i1, i2));
    faces.push_back(Face(i0, i2, i3));
    faces.push_back(Face(i0, i3, i1));
    faces.push_back(Face(i1, i3, i2));
    for (size_t i(0); i < faces.size(); ++i) { orientFace(faces[i]); }

    std::vector<bool> is_used(points.size(), false);
    is_used[i0] = true;
    is_used[i1] = true;
    is_used[i2] = true;
    is_used[i3] = true;

    for (size_t idx(0); idx < points.size(); ++idx) {
        if (is_used[idx]) { continue; }
        addPoint(static_cast<int>(idx));
    }
}

int ConvexHull3D::findFarthestPoint(int from) {
    int farthest = -1;
    double maxDistSq = -1.0;
    for (size_t i(0); i < points.size(); ++i) {
        if (static_cast<int>(i) == from){ continue; }
        double distSq = (points[i] - points[from]).normSquared();
        if (distSq <= maxDistSq){continue;}
        maxDistSq = distSq;
        farthest = static_cast<int>(i);
    }
    return farthest;
}

int ConvexHull3D::findFarthestFromLine(int a, int b) {
    int farthest = -1;
    double maxDistSq = -1.0;
    Vec3 ab = points[b] - points[a];
    if (ab.normSquared() < 1e-12) { return -1; }
    for (size_t i(0); i < points.size(); ++i) {
        if (static_cast<int>(i) == a || static_cast<int>(i) == b) { continue; }
        Vec3 ap = points[i] - points[a];
        double distSq = (ap.cross(ab)).normSquared() / ab.normSquared();
        if (distSq <= maxDistSq){continue;}
        farthest = static_cast<int>(i);
        maxDistSq = distSq;
    }
    return farthest;
}

int ConvexHull3D::findFarthestFromPlane(int a, int b, int c) {
    int farthest = -1;
    double maxDist = -1.0;
    Vec3 normal = (points[b] - points[a]).cross(points[c] - points[a]);
    if (normal.norm() < 1e-12) { return -1; }
    for (size_t i(0); i < points.size(); ++i) {
        if (static_cast<int>(i) == a || static_cast<int>(i) == b || static_cast<int>(i) == c) { continue; }
        double dist = std::abs(normal.dot(points[i] - points[a])) / normal.norm();
        if (dist <= maxDist){continue;}
        farthest = static_cast<int>(i);
        maxDist = dist;
    }
    return farthest;
}

void ConvexHull3D::addPoint(int idx) {
    Vec3 P = points[idx];
    std::vector<int> visible_face_indices;
    for (size_t i(0); i < faces.size(); ++i) {
        if (faces[i].normal.dot(P - points[faces[i].a]) <= 1e-9){continue;}
        visible_face_indices.push_back(static_cast<int>(i));
    }
    if (visible_face_indices.empty()) { return; }

    std::map<std::pair<int, int>, int> edge_count;
    for (size_t i(0); i < visible_face_indices.size(); ++i) {
        int face_idx = visible_face_indices[i];
        Face& f = faces[face_idx];
        addEdge(edge_count, f.a, f.b);
        addEdge(edge_count, f.b, f.c);
        addEdge(edge_count, f.c, f.a);
    }

    std::sort(visible_face_indices.rbegin(), visible_face_indices.rend());
    for (size_t i(0); i < visible_face_indices.size(); ++i) {
        faces.erase(faces.begin() + visible_face_indices[i]);
    }

    for (std::map<std::pair<int, int>, int>::const_iterator it(edge_count.begin());
        it != edge_count.end(); ++it) {
        if (it->second != 1) { continue; }
        faces.emplace_back(it->first.first, it->first.second, idx);
        orientFace(faces.back());
    }
}

void ConvexHull3D::addEdge(std::map<std::pair<int, int>, int>& edge_count, int a, int b) {
    if (a > b) { std::swap(a, b); }
    edge_count[{a, b}]++;
}




void MultiShapeConnector::addEllipse(const Mat3& H) {
    ellipses.emplace_back(H.get_col(2), H.get_col(0), H.get_col(1));
}

Vec3 MultiShapeConnector::findClosestPointOnEllipse(
        const Ellipse& ell, const Vec3& p_ext
) const {
    double min_dist_sq = std::numeric_limits<double>::max();
    Vec3 closest_point = ell.c;
    int resolution = 200;
    for (size_t i(0); i < resolution; ++i) {
        double angle = 2 * M_PI * i / resolution;
        Vec3 p = ell.c + ell.v1 * std::cos(angle) + ell.v2 * std::sin(angle);
        double dist_sq = (p - p_ext).normSquared();
        if (dist_sq >= min_dist_sq){continue;}
        min_dist_sq   = dist_sq;
        closest_point = p;
    }
    return closest_point;
}

double MultiShapeConnector::compute(std::vector<Vec3>& all_points, bool verbose) {
    all_points.clear();
    const size_t n = ellipses.size();
    for (size_t i(0); i < ellipses.size(); ++i){all_points.push_back(ellipses[i].c);}
    for (int iter(0); iter < 100; ++iter) {
        std::vector<Vec3> prev_points = all_points;
        for (size_t i(0); i < n; ++i) {
            Vec3 cent(0, 0, 0);
            for (size_t j(0); j < n; ++j){cent = cent + all_points[j] * (i != j);}
            all_points[i] = findClosestPointOnEllipse(ellipses[i], cent / (n - 1));
        }

        double mvn = 0.0;
        for (size_t i(0); i < n; ++i){mvn += (all_points[i] - prev_points[i]).norm();}
        if (mvn >= 1e-9) {continue;}
        if (verbose) { std::cout << "  Convergence reached." << std::endl; }
        break;
    }

    if (all_points.size() < 4) { return 0.0; }
    ConvexHull3D hull(all_points);
    hull.write_to_obj("convex_hull.obj");
    std::cout << "  -> Saved convex_hull.obj" << std::endl;
    return hull.computeVolume();
}

std::vector<Vec3> ConvexHull3D::computeVolumeGradient() const {
    std::vector<Vec3> grad(points.size(), Vec3(0, 0, 0));
    for (size_t i(0); i < faces.size(); ++i) {
        const Face& f = faces[i];
        Vec3 A = points[f.a];
        Vec3 B = points[f.b];
        Vec3 C = points[f.c];

        grad[f.a] = grad[f.a] + (B - A).cross(C - A) * (1.0 / 6.0);
        grad[f.b] = grad[f.b] + (C - B).cross(A - B) * (1.0 / 6.0);
        grad[f.c] = grad[f.c] + (A - C).cross(B - C) * (1.0 / 6.0);
    }
    return grad;
}

std::vector<Vec3> EllipseSystem::compute_points() {
    MultiShapeConnector connector;
    for (size_t i(0); i < leptons.size(); ++i) {
        NuSol solver(bjets[i], leptons[i], mass_params[2 * i], mass_params[2 * i + 1]);
        std::pair<bool, Mat3> H_res = solver.getH();
        if (!H_res.first) {continue;}
        connector.addEllipse(H_res.second); 
    }
    std::vector<Vec3> pts;
    connector.compute(pts, false);
    return pts;
}

std::vector<std::vector<Vec3>> EllipseSystem::compute_jacobian() {
    size_t n = leptons.size();
    std::vector<std::vector<Vec3>> jacobian(n, std::vector<Vec3>(2));
    for (size_t i(0); i < n; ++i) {
        NuSol solver(bjets[i], leptons[i], mass_params[2 * i], mass_params[2 * i + 1]);
        std::pair<bool, std::pair<Mat3, Mat3>> dH_res = solver.getH_derivatives();
        if (!dH_res.first) { continue; }
        jacobian[i][0] = dH_res.second.first.get_col(2);
        jacobian[i][1] = dH_res.second.second.get_col(2);
    }
    return jacobian;
}

std::vector<std::vector<double>> EllipseSystem::compute_distance_matrix(
    const std::vector<Vec3>& c_pts
) {
    size_t n = c_pts.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (size_t i(0); i < n; ++i) {
        for (size_t j(0); j < n; ++j) {D[i][j] = (c_pts[i] - c_pts[j]).normSquared();}
    }
    return D;
}

std::vector<std::vector<double>> EllipseSystem::compute_distance_jacobian(
    const std::vector<Vec3>& current_points,
    const std::vector<std::vector<Vec3>>& point_jacobian
) {
    size_t n = current_points.size();
    size_t p = mass_params.size();
    std::vector<std::vector<double>> dist_jac(n * n, std::vector<double>(p, 0.0));

    for (size_t i(0); i < n; ++i) {
        for (size_t j(0); j < n; ++j) {
            size_t idx = i * n + j;
            Vec3 diff = current_points[i] - current_points[j];
            for (size_t k(0); k < n; ++k) {
                double p1_dj = 2 * diff.dot(point_jacobian[k][0]); 
                double p2_dj = 2 * diff.dot(point_jacobian[k][1]); 
                if (i == k) {
                    dist_jac[idx][2 * k    ] = p1_dj;
                    dist_jac[idx][2 * k + 1] = p2_dj;
                }
                if (j == k) {
                    dist_jac[idx][2 * k    ] -= p1_dj;
                    dist_jac[idx][2 * k + 1] -= p2_dj;
                }
            }
        }
    }
    return dist_jac;
}

std::vector<std::vector<double>> EllipseSystem::compute_hessian_approx(
    const std::vector<std::vector<double>>& dist_jac
) {
    size_t dim = dist_jac[0].size();
    size_t n_points_sq = dist_jac.size();

    std::vector<std::vector<double>> hess(dim, std::vector<double>(dim, 0.0));
    for (size_t i(0); i < dim; ++i) {
        for (size_t j(0); j < dim; ++j) {
            for (size_t k(0); k < n_points_sq; ++k) {
                hess[i][j] += dist_jac[k][i] * dist_jac[k][j];
            }
        }
    }
    return hess;
}

void EllipseSystem::eigen_decomposition(
    const std::vector<std::vector<double>>& A,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<double>>& eigenvectors
) {
    const size_t n = A.size();
    const double eps = 1e-10;
    const int max_iter = 100;

    eigenvectors = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    for (size_t i(0); i < n; ++i) { eigenvalues[i] = A[i][i]; }
    for (size_t i(0); i < n; ++i) { eigenvectors[i][i] = 1.0; }

    std::vector<std::vector<double>> B = A;

    for (int iter(0); iter < max_iter; ++iter) {
        double max_off_diag = 0.0;
        size_t p(0), q(0);

        for (size_t i(0); i < n; ++i) {
            for (size_t j(i + 1); j < n; ++j) {
                if (std::fabs(B[i][j]) <= max_off_diag) {continue;}
                max_off_diag = std::fabs(B[i][j]);
                p = i; q = j;
            }
        }
        if (max_off_diag < eps) { break; }
        double theta_denom = B[q][q] - B[p][p];
        double theta = 0.0;
        if (std::fabs(theta_denom) < 1e-12) {theta = M_PI / 4.0;} 
        else {theta = 0.5 * std::atan2(2 * B[p][q], theta_denom);}

        double c = std::cos(theta);
        double s = std::sin(theta);

        double Bpp_val = B[p][p];
        double Bqq_val = B[q][q];
        double Bpq_val = B[p][q];

        B[p][p] = c * c * Bpp_val - 2 * c * s * Bpq_val + s * s * Bqq_val;
        B[q][q] = s * s * Bpp_val + 2 * c * s * Bpq_val + c * c * Bqq_val;
        B[p][q] = B[q][p] = 0.0;

        for (size_t i(0); i < n; ++i) {
            if (i == p || i == q) { continue; }

            double Bip = B[i][p];
            double Biq = B[i][q];

            B[i][p] = c * Bip - s * Biq;
            B[p][i] = B[i][p];

            B[i][q] = s * Bip + c * Biq;
            B[q][i] = B[i][q];
        }

        for (size_t i(0); i < n; ++i) {
            double eip = eigenvectors[i][p];
            double eiq = eigenvectors[i][q];
            eigenvectors[i][p] = c * eip - s * eiq;
            eigenvectors[i][q] = s * eip + c * eiq;
        }
    }
    for (size_t i(0); i < n; ++i) { eigenvalues[i] = B[i][i]; }
}

void EllipseSystem::analyze_system() {
    double learning_rate = 0.1;
    double tolerance     = 1e-6;
    long max_iterations  = 100000;

    this -> log_ = new Logger("optimization_log.csv", "ellipses.csv"); 
    for (int iter(0); iter < max_iterations; ++iter) {

        std::vector<Mat3> els = {}; 
        for (size_t i(0); i < leptons.size(); ++i) {
            NuSol solver(bjets[i], leptons[i], mass_params[2*i], mass_params[2*i+1]);
            els.push_back(std::get<1>(solver.getH())); 
        }
        this -> log_ -> Ellipses(&els); 

        std::vector<Vec3> current_points_for_dist;
        double current_obj_val = calculate_distance_squared(
            this->mass_params, this->leptons, this->bjets, current_points_for_dist
        );

        std::vector<double> grad_analytical = calculate_gradient_analytical(
            this->mass_params, this->leptons, this->bjets
        );

        double grad_norm_sq = 0.0;
        for (size_t i(0); i < grad_analytical.size(); ++i) {
            double g_val = grad_analytical[i];
            grad_norm_sq += g_val * g_val;
        }
        double grad_norm = std::sqrt(grad_norm_sq);
        std::cout << "  Iteration " << iter + 1
            << ": Objective = " << std::fixed << std::setprecision(6)
            << current_obj_val
            << ", Grad Norm = " << std::scientific << grad_norm
            << std::fixed << std::endl;

        if (grad_norm < tolerance) {
            std::cout << "  Convergence reached. Gradient norm below tolerance." << std::endl;
            break;
        }

        this -> points = this -> compute_points();
        std::vector<std::vector<Vec3>> point_jac  = this->compute_jacobian();
        std::vector<std::vector<double>> dist_jac = this->compute_distance_jacobian(this -> points, point_jac);
        std::vector<std::vector<double>> hessian_approx = this->compute_hessian_approx(dist_jac);

        this -> log_ -> log(iter, current_obj_val, Vec3(0, 0, 0), {0, 0}, {});
        for (size_t i(0); i < mass_params.size(); ++i) {
            double H_ii = hessian_approx[i][i];
            H_ii = (H_ii > 1e-9) ? 1.0 / H_ii : 1.0;
            this -> mass_params[i] -= learning_rate * grad_analytical[i] * H_ii;
        }
    }
    std::cout << "\nOptimization finished." << std::endl;
    std::cout << "Final mass_params: ";
    for (size_t i(0); i < this -> mass_params.size(); ++i) {std::cout << this -> mass_params[i] << " ";}
    std::cout << std::endl;

    for (size_t i(0); i < leptons.size(); ++i) {
        NuSol solver(bjets[i], leptons[i], mass_params[2*i], mass_params[2*i+1]);
        std::pair<bool, Mat3> H_res = solver.getH();
        if (!H_res.first){continue;}
        std::string ellipse_filename = "final_ellipse_" + std::to_string(i) + ".obj";
        write_ellipse_to_obj(H_res.second, ellipse_filename);
        std::cout << "  -> Saved " << ellipse_filename << std::endl;
    }

    ConvexHull3D final_hull(this->points);
    std::string final_hull_filename = "final_convex_hull.obj";
    final_hull.write_to_obj(final_hull_filename);
    std::cout << "  -> Saved " << final_hull_filename << std::endl;

    std::vector<std::vector<double>> final_dist_matrix = this->compute_distance_matrix(this->points);
    std::cout << "\nFinal Distance Matrix (elements squared):" << std::endl;
    for (size_t x(0); x < final_dist_matrix.size(); ++x) {
        for (size_t y(0); y < final_dist_matrix[x].size(); ++y) {
            std::cout << std::fixed << std::setprecision(4) << final_dist_matrix[x][y] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Connected points (vertices defining the volume):" << std::endl;
    for (size_t i(0); i < this->points.size(); ++i) {
        std::cout << "  Point " << i << ": ("
                  << std::fixed << std::setprecision(4) << this->points[i].x << ", "
                  << this->points[i].y << ", " << this->points[i].z << ")" << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
    delete this -> log_; 
}

std::vector<double> compute_volume_gradient(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets
) {
    MultiShapeConnector connector;
    std::vector<Mat3> H_matrices;
    for (size_t i(0); i < leptons.size(); ++i) {
        NuSol solver(bjets[i], leptons[i], mass_params[2 * i], mass_params[2 * i + 1]);
        std::pair<bool, Mat3> H_res = solver.getH();
        if (!H_res.first) { return std::vector<double>(mass_params.size(), 0.0); }
        connector.addEllipse(H_res.second);
        H_matrices.push_back(H_res.second);
    }

    std::vector<Vec3> points_computed;
    connector.compute(points_computed, false);
    if (points_computed.size() < 4) { return std::vector<double>(mass_params.size(), 0.0); }

    ConvexHull3D hull(points_computed);
    std::vector<Vec3> volume_grad = hull.computeVolumeGradient();
    std::vector<double> grad(mass_params.size(), 0.0);
    for (size_t i(0); i < points_computed.size(); ++i) {
        NuSol solver(bjets[i], leptons[i], mass_params[2 * i], mass_params[2 * i + 1]);
        std::pair<bool, std::pair<Mat3, Mat3>> dH_res = solver.getH_derivatives();
        if (!dH_res.first) { continue; }

        Vec3 dp_dmt = dH_res.second.first.get_col(2);
        Vec3 dp_dmw = dH_res.second.second.get_col(2);

        grad[2 * i] = volume_grad[i].dot(dp_dmt);
        grad[2 * i + 1] = volume_grad[i].dot(dp_dmw);
    }

    return grad;
}

std::vector<std::vector<double>> compute_volume_hessian(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets
) {
    const double eps = 1e-3;
    const size_t n = mass_params.size();
    std::vector<std::vector<double>> hess(n, std::vector<double>(n, 0.0));
    for (size_t i(0); i < n; ++i) {
        std::vector<double> m_plus = mass_params;
        std::vector<double> m_minus = mass_params;
        m_plus[i]  += eps;
        m_minus[i] -= eps;
        std::vector<double> plus  = compute_volume_gradient(m_plus , leptons, bjets);
        std::vector<double> minus = compute_volume_gradient(m_minus, leptons, bjets);
        for (size_t j(0); j < n; ++j){hess[j][i] = (plus[j] - minus[j]) / (2 * eps);}
    }

    for (size_t i(0); i < n; ++i) {
        for (size_t j(0); j < n; ++j) {
            if (i == j){continue;}
            double v = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = v; hess[j][i] = v;
        }
    }
    return hess;
}
