#ifndef CHILDRENKINEMATICS_H
#define CHILDRENKINEMATICS_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

struct kinematic_t {
    float pt = 0; 
    float energy = 0; 
    float eta = 0; 
    float phi = 0; 
    int pdgid = 0; 
}; 

struct misc_t {
    kinematic_t kin; 
    bool is_lep = false; 
    float mass_clust = 0; 
    float delta_R = 0; 
    float frc_energy = 0;
    float frc_pt = 0; 
}; 

struct perms_t {
    bool RR = false; 
    bool SS = false; 
    bool RS = false; 
    bool CT = false; 
    bool FT = false; 
    float delta_R = 0; // between children
    float top_pt  = 0; 
    float top_e   = 0; 
    float mass    = 0; 
}; 




class childrenkinematics: public selection_template
{
    public:
        childrenkinematics();
        ~childrenkinematics() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::vector<kinematic_t> res_kinematics = {}; 
        std::vector<kinematic_t> spec_kinematics = {}; 

        std::vector<misc_t>  res_decay_mode = {}; 
        std::vector<misc_t>  spec_decay_mode = {}; 
        std::vector<perms_t> top_clusters = {}; 

    private:

        template <typename g>
        void dump_kinematics(std::vector<kinematic_t>* data, g* p){
            kinematic_t kx; 
            kx.pt = (p -> pt) / 1000; 
            kx.energy = (p -> e) / 1000;
            kx.eta  = p -> eta; 
            kx.phi  = p -> phi; 
            kx.pdgid = p -> pdgid; 
            data -> push_back(kx); 
        }

        template <typename g>
        void dump_kinematics(kinematic_t* data, g* p){
            data -> pt = (p -> pt) / 1000; 
            data -> energy = (p -> e) / 1000;
            data -> eta  = p -> eta; 
            data -> phi  = p -> phi; 
            data -> pdgid = p -> pdgid; 
        }
};

#endif
