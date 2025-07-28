#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "linalg.h"
#include <vector>
#include <string>
#include <map>
class Logger; 

void write_ellipse_to_obj(const Mat3& H, const std::string& filename);
std::vector<double> compute_volume_gradient(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets
);

std::vector<std::vector<double>> compute_volume_hessian(
    const std::vector<double>& mass_params,
    const std::vector<Vec4>& leptons,
    const std::vector<Vec4>& bjets
);

class ConvexHull3D {
    public:
        struct Face {
            int a;
            int b;
            int c;
            Vec3 normal;
            Face(int a, int b, int c);
        };
    
        double computeVolume();
        std::vector<Vec3> points;
        std::vector<Face> faces;
        ConvexHull3D(const std::vector<Vec3>& input_points);
        std::vector<Vec3> computeVolumeGradient() const;
        void write_to_obj(const std::string& filename);
    
    private:
        Vec3 centroid;
        void computeCentroid();
        void orientFace(Face& f);
        void buildHull();
        int findFarthestPoint(int from);
        int findFarthestFromLine(int a, int b);
        int findFarthestFromPlane(int a, int b, int c);
        void addPoint(int idx);
        void addEdge(std::map<std::pair<int, int>, int>& edge_count, int a, int b);
};

class MultiShapeConnector {
    private:
        struct Ellipse {
            Vec3 c;
            Vec3 v1;
            Vec3 v2;
            Ellipse(const Vec3& center, const Vec3& vec1, const Vec3& vec2):
                c(center), v1(vec1), v2(vec2){}
        };
    
        std::vector<Ellipse> ellipses;
        Vec3 findClosestPointOnEllipse(const Ellipse& ell, const Vec3& p_ext) const;
    
    public:
        void addEllipse(const Mat3& H);
        double compute(std::vector<Vec3>& all_points, bool verbose = false);
};

class EllipseSystem
{
    private:
        std::vector<Vec4> leptons;
        std::vector<Vec4> bjets;
        std::vector<double> mass_params;
        Logger* log_ = nullptr; 
 
    public:
        std::vector<Vec3> points;
        std::vector<Vec3> compute_points();
        void analyze_system();
    
        EllipseSystem(
            const std::vector<Vec4>& l,
            const std::vector<Vec4>& b,
            const std::vector<double>& m
        ) : leptons(l), bjets(b), mass_params(m) {}
    
        std::vector<std::vector<Vec3>> compute_jacobian();
        std::vector<std::vector<double>> compute_distance_matrix(
            const std::vector<Vec3>& points
        );
    
        std::vector<std::vector<double>> compute_distance_jacobian(
            const std::vector<Vec3>& points,
            const std::vector<std::vector<Vec3>>& point_jacobian
        );
    
        std::vector<std::vector<double>> compute_hessian_approx(
            const std::vector<std::vector<double>>& dist_jac
        );
    
        void eigen_decomposition(
            const std::vector<std::vector<double>>& A,
            std::vector<double>& eigenvalues,
            std::vector<std::vector<double>>& eigenvectors
        );
};

#endif
