#ifndef VALIDATION_H
#define VALIDATION_H

#include <templates/selection_template.h>

struct packet_t {

    ~packet_t(); 

    std::vector<particle_template*> bquarks  = {}; 
    std::vector<particle_template*> leptons  = {}; 
    std::vector<particle_template*> neutrino = {}; 
    std::vector<particle_template*> objects  = {}; 

    std::vector<particle_template*> static_nu1 = {}; 
    std::vector<particle_template*> static_nu2 = {}; 
    std::vector<double> static_distances  = {}; 

    std::vector<particle_template*> dynamic_nu1 = {}; 
    std::vector<particle_template*> dynamic_nu2 = {}; 
    std::vector<double> dynamic_distances = {}; 

    std::string name = ""; 
    std::string device = "cuda:0"; 

    double met = 0; 
    double phi = 0;
    double null = 1e-10; 
    double step = 1e-9;
    double tolerance = 1e-6;
    unsigned int timeout = 100; 
    bool _is_marked = false; 
}; 


class validation: public selection_template
{
    public:
        validation();
        ~validation() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;
        void reconstruction(packet_t* data); 
        void update_state(packet_t* data); 

        int num_device = 1; 
        double masstop = 172.62*1000; 
        double massw   = 80.385*1000; 

    private:
        std::vector<packet_t*> storage = {}; 

        particle_template* nu_tru = nullptr; 
        particle_template* b_qrk  = nullptr; 
        particle_template* lepton = nullptr; 

        template <typename g>
        packet_t* build_packet(g* evnt, std::string name){
            if (!this -> storage.size()){this -> storage.reserve(4);}
            size_t idx = this -> storage.size(); 
            packet_t* data = new packet_t(); 
            data -> met = evnt -> met; 
            data -> phi = evnt -> phi; 
            data -> name = name; 
            data -> device = "cuda:" + std::string(this -> threadIdx % this -> num_device); 
            this -> storage.push_back(data); 
            return this -> storage[idx]; 
        }


        template<typename g>
        std::vector<g*> upcast(std::vector<particle_template*>* inpt){
            typename std::vector<g*> out; 
            for (size_t x(0); x < inpt -> size(); ++x){out.push_back((g*)inpt -> at(x));}
            return out; 
        }

        template<typename g>
        std::vector<g*> upcast(std::map<std::string, particle_template*>* inpt){
            std::vector<particle_template*> out = this -> vectorize(inpt); 
            return this -> upcast<g>(&out); 
        }




};

#endif
