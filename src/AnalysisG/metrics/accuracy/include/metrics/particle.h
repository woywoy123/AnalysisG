#ifndef AVERAGE_PARTICLE_H
#define AVERAGE_PARTICLE_H
#include <templates/particle_template.h>

enum class channel_e {leptonic, hadronic, undefined};
enum class object_e  {truth, valid, failed, undefined}; 
enum class content_e {bquark, lepton, other, undefined};

class rectop : public particle_template {
    public:
        rectop(); 
        ~rectop(); 

        void decode(int chn); 
        
        channel_e  chn = channel_e::undefined; 
        object_e   idn =  object_e::undefined; 
        std::map<content_e, int> composition; 
        double score = 0; 
}; 








#endif
