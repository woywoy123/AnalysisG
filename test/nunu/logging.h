#ifndef LOGGER_H
#define LOGGER_H
#include <vector>
#include <string>

class Vec3; 
class Mat3; 

class Logger 
{
    public:
        Logger(
                const std::string& fname, 
                const std::string& felp, 
                const std::vector<Mat3>* ellipses = nullptr
        );

        void Ellipses(const std::vector<Mat3>* ellipse = nullptr); 
        void writeHeader(size_t n); 
        void log(
            int iter, double objective, 
            const Vec3& centroid, 
            const std::vector<double>& angles, 
            const std::vector<Vec3>& points
        ); 
        ~Logger(); 

    private: 
        std::ofstream* file_el = nullptr;
        std::ofstream* file = nullptr;
        const std::vector<Mat3>* Hs;
        bool firstWrite = true;
        int resolution = 100; 
};
#endif
