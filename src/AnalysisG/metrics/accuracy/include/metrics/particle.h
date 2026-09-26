#ifndef AVERAGE_PARTICLE_H
#define AVERAGE_PARTICLE_H
#include <templates/particle_template.h>

enum class channel_e {leptonic, hadronic, undefined};
enum class object_e  {truth, valid, failed, undefined}; 
enum class content_e {bquark, lepton, other, undefined};

class rectop {
    public:
        rectop(); 
        ~rectop(); 

        void decode(int chn); 
        bool valid(); 

        double pt    = 0;
        double eta   = 0; 
        double phi   = 0; 
        double mass  = 0; 
        double score = -1.0; 

        object_e   idn =  object_e::undefined; 
        object_e   idt =  object_e::undefined; 

        channel_e  chn = channel_e::undefined; 
        std::map<content_e, int> composition; 
}; 

#endif
