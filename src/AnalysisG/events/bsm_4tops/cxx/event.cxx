#include <bsm_4tops/event.h>

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

event_template* bsm_4tops::clone(){
    bsm_4tops* ev = new bsm_4tops();
    ev -> reconstruct_nunu = this -> reconstruct_nunu; 
    return (event_template*)ev;
}

void bsm_4tops::build(element_t* el){
    el -> get("event_number", &this -> event_number); 
    el -> get("mu" , &this -> mu);
    el -> get("met", &this -> met); 
    el -> get("phi", &this -> phi);

    float w = 0; 
    el -> get("weight", &w); 
    this -> weight = w; 
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
        if (!_Tops.count(index)){continue;}
        _Tops[index] -> register_child(itc -> second); 
        itc -> second -> register_parent(_Tops[index]); 
        itc -> second -> index = index; 
    }
 
    std::map<int, truthjet*>::iterator itj; 
    for (itj = _TruthJets.begin(); itj != _TruthJets.end(); ++itj){
        truthjet* tj = itj -> second; 
        for (int x : tj -> top_index){
            if (!_Tops.count(x)){continue;}
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
        jt -> pdgid = (jt -> btag_DL1r_85) ? 5 : 0; 
        for (int x : jt -> top_index){
            if (!_Tops.count(x)){continue;}
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
        std::string hash_ = l -> hash; 
        if (accept[hash_]){continue;}

        top_children* c = (top_children*)id -> second[0]; 
        std::map<std::string, particle_template*> ch = c -> children; 
        if (ch.size()){continue;}

        c -> register_child(l); 
        l -> register_parent(c); 

        if (l -> type == "mu"){
            muon* lt = (muon*)l; 
            lt -> from_res = c -> from_res; 
            lt -> top_index = c -> index; 
        }

        if (l -> type == "el"){
            electron* lt = (electron*)l; 
            lt -> from_res = c -> from_res; 
            lt -> top_index = c -> index; 
        }
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

    if (this -> debug_mode){this -> debug_strings();}
    if (!this -> reconstruct_nunu){return;}
  
    std::vector<particle_template*> nux = this -> multi_neutrino(
            &this -> DetectorObjects, this -> phi, this -> met, 
            172.68 * 1000, 80.385 * 1000, 0.0001, 0.1
    ); 

    for (int x(0); x < nux.size(); ++x){
        std::map<std::string, particle_template*> pox = nux[x] -> parents;
        std::map<std::string, particle_template*>::iterator itr = pox.begin(); 
        std::map<int, bool> top_idx_; 
        for (; itr != pox.end(); ++itr){
            std::string type = itr -> second -> type; 
            std::vector<int> _top_idx = {}; 
            if      (type == "jet"){_top_idx = ((jet*)itr -> second) -> top_index;}
            else if (type == "mu"){_top_idx = {((muon*)itr -> second) -> top_index};}
            else if (type == "el"){_top_idx = {((electron*)itr -> second) -> top_index};}
            for (int k(0); k < _top_idx.size(); ++k){top_idx_[_top_idx[k]] = _Tops.count(_top_idx[k]);}
        }
        
        (&nux[x] -> parents) -> clear(); 
        bool is_fake = true; 
        std::map<int, bool>::iterator itx = top_idx_.begin(); 
        for (; itx != top_idx_.end(); ++itx){
            if (!itx -> second){continue;}

            std::map<std::string, particle_template*> tc = _Tops[itx -> first] -> children; 
            std::map<std::string, particle_template*>::iterator itc = tc.begin(); 
            for (; itc != tc.end(); ++itc){
                if (!itc -> second -> is_nu){continue;}
                is_fake = false; break; 
            }
            nux[x] -> register_parent(_Tops[itx -> first]); 
        }
        if (is_fake){nux[x] -> index = -1;}
        this -> DetectorObjects.push_back(nux[x]);
    }
}

void bsm_4tops::debug_strings(){
    std::cout << "-------------- new event ---------------- " << std::endl;
    std::cout << " met: " << double(this -> met) << " phi: " << double(this -> phi) << std::endl;
    std::cout << "---------- Truth ------------------- " << std::endl;
    std::cout << " neutrinos::::: " << std::endl;
    for (size_t x(0); x < this -> Children.size(); ++x){
        if (!this -> Children[x] -> is_nu){continue;}
        particle_template* p = this -> Children[x]; 
        std::cout << "px: "  << double(p -> px) 
                  << " py: " << double(p -> py)
                  << " pz: " << double(p  -> pz) 
                  << " e: "  << double(p -> e) 
                  << " top-index: " << ((top_children*)this -> Children[x]) -> top_index 
                  << std::endl;
    }

    std::cout << "------------ leptons ------------- " << std::endl;
    for (size_t x(0); x < this -> DetectorObjects.size(); ++x){
        if (!this -> DetectorObjects[x] -> is_lep){continue;}    
        particle_template* l = this -> DetectorObjects[x]; 

        int t = -1; 
        if (l -> type == "mu"){t = ((muon*)l) -> top_index;}
        if (l -> type == "el"){t = ((electron*)l) -> top_index;}
        std::cout << "hash of lepton: " << std::string(l -> hash) << " top-index: " << t << std::endl;
    }
    
    std::cout << "------------- b jets ------------- " << std::endl;
    for (size_t x(0); x < this -> Tops.size(); ++x){
        top* tx = (top*)this -> Tops[x]; 
        for (size_t z(0); z < tx -> Jets.size(); ++z){
            std::cout << "hash of b-jet: " << std::string(tx -> Jets[z] -> hash) << " top-index:" << tx -> index << std::endl;
        }
    }

    std::cout << "-------- Detector Objects -------- " << std::endl;
    for (size_t x(0); x < this -> DetectorObjects.size(); ++x){
        particle_template* p = this -> DetectorObjects[x]; 
        std::cout << " px: " << double(p -> px) 
                  << " py: " << double(p -> py) 
                  << " pz: " << double(p  -> pz) 
                  << " e: "  << double(p -> e) 
                  << " h: " << std::string(p -> hash) << std::endl;
    }
}
