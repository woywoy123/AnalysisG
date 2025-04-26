#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H

#include <templates/selection_template.h>

class neutrino; 

struct packet_t {

    ~packet_t(); 

    std::vector<particle_template*> particles = {}; 
    std::vector<particle_template*> t_bquarks  = {}; 
    std::vector<particle_template*> t_leptons  = {}; 
    std::vector<particle_template*> t_neutrino = {}; 

    std::vector<int>  matched_bquarks = {}; 
    std::vector<int>  matched_leptons = {};

    std::vector<double> distance = {}; 
    std::vector<double> chi2_nu1 = {}; 
    std::vector<double> chi2_nu2 = {}; 

    std::vector<particle_template*> nu1 = {}; 
    std::vector<particle_template*> nu2 = {}; 

    std::string name = ""; 
    std::string device = "cuda:0"; 

    double met = 0; 
    double phi = 0;
    double null = 1e-5; 
    double perturb = 1e-1; 

    long steps = 20;
    bool _is_marked = false; 
}; 








class combinatorial: public selection_template
{
    public:
        combinatorial();
        ~combinatorial() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;
        void update_state(packet_t* data); 
        void reconstruction(packet_t* data); 

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
            if (!this -> num_device){this -> num_device = 1;}
            data -> device = "cuda:" + std::to_string(this -> threadIdx % this -> num_device); 
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
