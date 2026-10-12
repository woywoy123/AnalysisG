#ifndef AVERAGE_DATA_H
#define AVERAGE_DATA_H
#include <metrics/particle.h>
#include <metrics/samples.h>
#include <tools/tools.h>


class atomic_t : public tools {

    public: 
        atomic_t();
        ~atomic_t(); 
        
        long add_truth(double m); 
        long add_value(double m); 

        std::vector<double> truth; 
        std::vector<double> masses; 
        std::vector<double> weight;

        std::string* mode  = nullptr;
        std::string* kfold = nullptr;
        std::string* epoch = nullptr;
        std::string* model = nullptr; 
        std::string*   prc = nullptr; 

        long index = -1;
}; 




class pdax_t : public tools {
    
    public:

        pdax_t();
        ~pdax_t(); 

        void dmake(pdax_t* ptr); 

        std::vector<std::vector<double>>* mlp = nullptr;
        std::vector<double>* weight = nullptr; 
        std::vector<double>* masses = nullptr; 
        
        std::vector<long>* index = nullptr; 
        std::vector<int>* ntrus = nullptr; 
        std::vector<int>* ntops = nullptr; 

        std::string mode  = "";
        std::string prc   = "";
        std::string kfold = "";
        std::string epoch = "";
        std::string model = ""; 
};

#endif
