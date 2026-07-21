#ifndef ROC_H
#define ROC_H
#include <plotting/plotting.h>

struct roc_t {
    int cls = 0; 
    int kfold = 0; 
    std::string model = ""; 
        
    std::vector<double> _auc = {}; 
    std::vector<std::vector<double>> tpr_ = {}; 
    std::vector<std::vector<double>> fpr_ = {}; 

    std::vector<std::vector<int>>*     truth = nullptr;
    std::vector<std::vector<double>>* scores = nullptr; 
};

class roc: public plotting
{
    public:
        roc(); 
        ~roc(); 

        void build_ROC(
            std::string name, int kfold, 
            std::vector<int>* label, 
            std::vector<std::vector<double>>* scores
        ); 
        std::vector<roc_t*> get_ROC(); 

        std::map<std::string, std::map<int, std::vector<std::vector<double>>*>> roc_data = {}; 
        std::map<std::string, std::map<int, std::vector<std::vector<int>>*>>      labels = {};  

    private: 
        std::vector<roc_t*> ptr_roc = {}; 

        template <typename g>
        std::vector<std::vector<g>>* generate(size_t x, size_t y){
            typename std::vector<g> v(y, 0); 
            return new std::vector<std::vector<g>>(x, v);
        }
}; 

#endif 
