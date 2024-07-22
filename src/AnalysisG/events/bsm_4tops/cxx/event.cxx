#include "event.h"

bsm_4tops::bsm_4tops(){
    this -> name = "bsm_4tops"; 
    this -> add_leaf("weight", "weight_mc"); 
    this -> add_leaf("event_number", "eventNumber"); 
    this -> add_leaf("mu", "mu"); 
    this -> add_leaf("met", "met_met"); 
    this -> add_leaf("phi", "met_phi"); 
    this -> trees = {"nominal"}; 

    this -> register_particle(&this -> m_Tops);
    this -> register_particle(&this -> m_Children); 

    this -> register_particle(&this -> m_TruthJets);
    this -> register_particle(&this -> m_TruthJetParton); 

    this -> register_particle(&this -> m_Jets); 
    this -> register_particle(&this -> m_JetParton); 

    this -> register_particle(&this -> m_Electrons); 
    this -> register_particle(&this -> m_Muons); 
}

bsm_4tops::~bsm_4tops(){}

event_template* bsm_4tops::clone(){return (event_template*)new bsm_4tops();}

void bsm_4tops::build(element_t* el){
    el -> get("event_number", &this -> event_number); 
    el -> get("weight", (float*)&this -> weight); 
    el -> get("mu", &this -> mu);
    el -> get("met", &this -> met); 
    el -> get("phi", &this -> phi); 
}

void bsm_4tops::CompileEvent(){
    std::map<int, top*>                 _Tops = this -> sort_by_index(&this -> m_Tops); 
    std::map<int, top_children*> _TopChildren = this -> sort_by_index(&this -> m_Children); 
    std::map<int, truthjet*>       _TruthJets = this -> sort_by_index(&this -> m_TruthJets); 
    std::map<int, jet*>                 _Jets = this -> sort_by_index(&this -> m_Jets); 
    std::map<int, truthjetparton*> _TruthJetPartons = this -> sort_by_index(&this -> m_TruthJetParton); 
    std::map<int, jetparton*>           _JetPartons = this -> sort_by_index(&this -> m_JetParton); 

    std::map<int, top_children*>::iterator itc; 
    for (itc = _TopChildren.begin(); itc != _TopChildren.end(); ++itc){
        int index = itc -> second -> top_index; 
        _Tops[index] -> register_child(itc -> second); 
        itc -> second -> register_parent(_Tops[index]); 
        itc -> second -> index = index; 
    }
 
    std::map<int, truthjet*>::iterator itj; 
    for (itj = _TruthJets.begin(); itj != _TruthJets.end(); ++itj){
        truthjet* tj = itj -> second; 
        for (int x : tj -> top_index){
            if (x == -1){continue;}
            tj -> Tops.push_back(_Tops[x]); 
            _Tops[x] -> TruthJets.push_back(tj); 
        }
    }

    std::map<int, truthjetparton*>::iterator itjp = _TruthJetPartons.begin(); 
    for (; itjp != _TruthJetPartons.end(); ++itjp){
        truthjetparton* tjp = itjp -> second; 
        _TruthJets[tjp -> truthjet_index] -> Parton.push_back(tjp); 
        tjp -> register_child(_TruthJets[tjp -> truthjet_index]); 
        for (int x : tjp -> topchild_index){tjp -> register_parent(_TopChildren[x]);}
    }

    std::map<int, jet*>::iterator ij; 
    for (ij = _Jets.begin(); ij != _Jets.end(); ++ij){
        jet* jt = ij -> second; 
        for (int x : jt -> top_index){
            if (x == -1){continue;}
            jt -> Tops.push_back(_Tops[x]); 
            _Tops[x] -> Jets.push_back(jt); 
        }
    }
    
    std::map<int, jetparton*>::iterator ijp = _JetPartons.begin(); 
    for (; ijp != _JetPartons.end(); ++ijp){
        jetparton* jp = ijp -> second; 
        _Jets[jp -> jet_index] -> Parton.push_back(jp); 
        jp -> register_child(_Jets[jp -> jet_index]); 
        for (int x : jp -> topchild_index){jp -> register_parent(_TopChildren[x]);}
    }

    std::map<int, top_children*> maps = {}; 
    for (itc = _TopChildren.begin(); itc != _TopChildren.end(); ++itc){
        if (!(bool)itc -> second -> is_lep){continue;}
        maps[itc -> first] = itc -> second;
    }

    std::vector<particle_template*> lep = {}; 
    std::map<std::string, electron*>::iterator ite = this -> m_Electrons.begin(); 
    for (; ite != this -> m_Electrons.end(); ++ite){lep.push_back(ite -> second);}

    std::map<std::string, muon*>::iterator itm = this -> m_Muons.begin(); 
    for (; itm != this -> m_Muons.end(); ++itm){lep.push_back(itm -> second);}

    std::map<double, std::vector<particle_template*>> dist = {}; 
    for (itc = _TopChildren.begin(); itc != _TopChildren.end(); ++itc){
        for (int x(0); x < lep.size(); ++x){
            double dr = itc -> second -> DeltaR(lep[x]); 
            if (dr > 0.1){continue;}
            dist[dr] = {itc -> second, lep[x]};
        }
    }

    std::map<std::string, bool> accept = {}; 
    std::map<double, std::vector<particle_template*>>::iterator id;
    for (id = dist.begin(); id != dist.end(); ++id){
        particle_template* l = id -> second[1]; 
        std::string    hash_ = l -> hash; 
        if (accept[hash_]){continue;}

        top_children*      c = (top_children*)id -> second[0]; 
        std::map<std::string, particle_template*> ch = c -> children; 
        if (ch.size()){continue;}

        c -> register_child(l); 
        l -> register_parent(c); 
        l -> index = c -> index; 
        accept[hash_] = true;
    }

    this -> vectorize(&_Tops              , &this -> Tops); 
    this -> vectorize(&_TopChildren       , &this -> Children); 
    this -> vectorize(&_TruthJets         , &this -> TruthJets); 
    this -> vectorize(&this -> m_Jets     , &this -> Jets); 
    this -> vectorize(&this -> m_Electrons, &this -> Electrons); 
    this -> vectorize(&this -> m_Muons    , &this -> Muons); 

    this -> vectorize(&this -> m_Jets     , &this -> DetectorObjects); 
    this -> vectorize(&this -> m_Muons    , &this -> DetectorObjects); 
    this -> vectorize(&this -> m_Electrons, &this -> DetectorObjects); 
}
