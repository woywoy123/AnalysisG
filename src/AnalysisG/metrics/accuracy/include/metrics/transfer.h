#ifndef AVERAGE_LAYER_H
#define AVERAGE_LAYER_H

#include <metrics/samples.h>
#include <tools/tools.h>

class pdata {
    public:
        pdata();
        ~pdata(); 

        std::vector<double> pt; 
        std::vector<double> eta; 
        std::vector<double> phi; 
        std::vector<double> energy; 
        std::vector<double> mass; 
        std::vector<double> score; 
        std::vector<int> chn;

        std::string type = ""; 
        std::string mode = ""; 
}; 


class edata : public tools
{
    public:
        edata();
        ~edata(); 

        std::vector<pdata*> training   = {}; 
        std::vector<pdata*> validation = {}; 
        std::vector<pdata*> evaluation = {}; 

        std::vector<double> ntop_score = {}; 
        int ntop_tru = 0; 
        int ntop_prd = 0; 
        int dsid     = 0; 
        int prc_idx  = 0; 

        double acc_edge  = 0; 

        int kfold = 0;
        int epoch = 0;
        std::string mrk = ""; 
}; 






#endif
