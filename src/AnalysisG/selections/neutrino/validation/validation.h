#ifndef NEUTRINO_VALIDATION_H
#define NEUTRINO_VALIDATION_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>


class nu: public particle_template
{
    public:
        using particle_template::particle_template; 
        double distance;
}; 

class bquark: public particle_template
{
    public:
        using particle_template::particle_template; 
}; 

class tquark: public particle_template
{
    public:
        using particle_template::particle_template; 
}; 

class lepton: public particle_template
{
    public:
        using particle_template::particle_template; 
}; 

class boson: public particle_template
{
    public:
        using particle_template::particle_template; 
}; 


struct package {
    double met; 
    double phi; 

    std::vector<nu*>     truth_nus; 
    std::vector<tquark*> truth_tops; 
    std::vector<boson*>  truth_bosons;

    std::vector<lepton*> reco_leptons; 
    std::vector<boson*>  reco_bosons;

    std::vector<lepton*> truth_leptons;
    std::vector<bquark*> truth_bquarks; 

    std::vector<bquark*> truth_bjets; 
    std::vector<tquark*> truth_jets_top; 

    std::vector<bquark*> bjets; 
    std::vector<tquark*> jets_top; 
    std::vector<tquark*> lepton_jets_top;

    // ---------------------------- //
    std::vector<nu*> c1_reconstructed_children_nu; 
    std::vector<nu*> c1_reconstructed_truthjet_nu; 
    std::vector<nu*> c1_reconstructed_jetchild_nu; 
    std::vector<nu*> c1_reconstructed_jetlep_nu; 
     
    std::vector<nu*> c2_reconstructed_children_nu; 
    std::vector<nu*> c2_reconstructed_truthjet_nu; 
    std::vector<nu*> c2_reconstructed_jetchild_nu; 
    std::vector<nu*> c2_reconstructed_jetlep_nu; 
}; 



class validation: public selection_template
{
    public:
        validation();
        virtual ~validation(); 
        selection_template* clone(); 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::map<std::string, package> data_out;

        // static masses
        double masstop = 172.62*1000; 
        double massw   = 80.385*1000; 

    private:

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

        std::vector<nu*> build_neutrinos(
                std::vector<bquark*>* bqs, std::vector<lepton*>* lpt, 
                std::vector<tquark*>* tps, std::vector<boson*>* wbs,
                double met, double phi, std::vector<nu*>* truth
        ); 



};

#endif
