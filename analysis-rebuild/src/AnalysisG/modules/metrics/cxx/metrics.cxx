#include <metrics/metrics.h>

metrics::metrics(){}
metrics::~metrics(){}

void metrics::register_model(model_template* mod, int kfold){
    std::string name = mod -> name; 
    std::cout << name << std::endl;



}

