#ifndef EVENTS_SSML_MC20_H
#define EVENTS_SSML_MC20_H

#include <ssml_mc20/particles.h>
#include <templates/event_template.h>

class ssml_mc20: public event_template
{
    public:
        ssml_mc20(); 
        virtual ~ssml_mc20(); 
        unsigned long long eventNumber; 

        float event_category;

        float weight_mc; 
        float weight_pileup;  
        float weight_beamspot; 
        float weight_jvt_effSF; 
        float weight_lep_tightSF; 
        float weight_ftag_effSF; 
        float global_trigger_SF; 

        float phi; 
        float met;
        float met_sum; 

        float HT_all;
        int pass_ssee; 
        int pass_ssem; 
        int pass_ssmm; 

        int pass_eem_zveto; 
        int pass_eee_zveto; 
        int pass_emm_zveto; 
        int pass_mmm_zveto; 
        int pass_llll_zveto; 

        int pass_eem; 
        int pass_eee; 
        int pass_emm; 
        int pass_mmm; 

        bool broken_event = false; 

        int n_electrons;
        int n_fjets;
        int n_jets; 
        int n_leptons; 
        int n_muons; 

        std::vector<particle_template*> Tops; 
        std::vector<particle_template*> TruthChildren; 
        std::vector<particle_template*> Zprime; 
        std::vector<particle_template*> Leptsn; 
        std::vector<particle_template*> Electrons; 

        std::vector<particle_template*> Jets; 
        std::vector<particle_template*> Leptons; 
        std::vector<particle_template*> Detector; 
        std::vector<particle_template*> TruthJets;

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private:
        std::map<std::string, jet*>      m_jets; 
        std::map<std::string, muon*>     m_muons; 
        std::map<std::string, electron*> m_electrons; 
        std::map<std::string, lepton*>   m_leptons; 

        std::map<std::string, zboson*>   m_zprime; 
        std::map<std::string, top*>      m_tops; 

        std::map<std::string, parton*>    m_partons; 
        std::map<std::string, truthjet*>  m_truthjets; 

        template <typename G>
        std::map<int, G*> sort_by_index(std::map<std::string, G*>* ipt){
            std::map<int, G*> data = {}; 
            typename std::map<std::string, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){data[int(ix -> second -> index)] = ix -> second;}
            return data; 
        }

        template <typename m, typename G, typename g>
        void vectorize(std::map<m, G*>* ipt, std::vector<g*>* vec){
            typename std::map<m, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec -> push_back(ix -> second);}
        }

        template <typename lx>
        bool match_object(std::map<int, top*>* topx, std::map<std::string, lx*>* objects){
            typename std::map<std::string, lx*>::iterator ite = objects -> begin();
            bool brk = false;
            for (; ite != objects -> end(); ++ite){
                int idx = ite -> second -> top_index; 
                std::string ty = ite -> second -> type; 
                particle_template* tx = (particle_template*)ite -> second; 
                brk += idx == -2; 

                if (ty == "mu" || ty == "el"){this -> Leptons.push_back(tx);}
                else if (ty == "jet"){this -> Jets.push_back(tx);}
                else if (ty == "truth_jet"){this -> TruthJets.push_back(tx);}
                else if (ty == "parton"){this -> TruthChildren.push_back(tx);}
                if (idx < 0 || !topx -> count(idx)){continue;}

                top* tp = (*topx)[idx]; 
                if (ty == "mu" || ty == "el"){tp -> leptons.push_back(tx);}
                else if (ty == "jet"){tp -> jets.push_back(tx);}
                else if (ty == "truth_jet"){tp -> truthjets.push_back(tx);}
                else if (ty == "parton"){
                    tp -> register_child(tx);
                    tx -> register_parent((particle_template*)tp);
                }
            } 
            return brk; 
        }

}; 


#endif
