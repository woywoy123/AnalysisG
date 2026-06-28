#include <metrics/topefficiency.h>
#include <unordered_map>
#include <stdexcept>
#include <string>

particle::particle(){}
particle::particle(double _pt, double _eta, double _phi, double _mass){
    this -> mass = _mass; this -> eta = _eta; 
    this -> phi = _phi;   this -> pt = _pt; 
}

particle::~particle(){}

Event::Event(){}
Event::~Event(){}

void Event::make_particle(std::map<std::string, std::vector<double>>* vals){
    auto lamb  = [](std::string o) -> std::string {return "nominal." + o + "." + o;}; 
    auto BaseP = [this](
            std::vector<double> pt , std::vector<double> eta, 
            std::vector<double> phi, std::vector<double> mass, 
            std::vector<double> lep, std::vector<double> chi2, 
            std::vector<double> pr
    ) -> std::vector<particle> {
        std::vector<particle> out; 
        for (size_t x(0); x < pt.size(); ++x){
            out.push_back(particle(pt[x], eta[x], phi[x], mass[x]));
            out[x].leptonic = lep[x]; 
            if (chi2.size()){out[x].chi2 = chi2[x];}
            if (pr.size()){out[x].PR = pr[x];}
        }
        return out; 
    }; 

    auto recovery = [this](double pred) -> double {
        return double(this -> top_truth.size()) - pred; 
    }; 

    this -> production = this -> process_sample(this -> dset_name); 
    this -> top_truth = BaseP(
            (*vals)[lamb("top_truth_pt"      )],
            (*vals)[lamb("top_truth_eta"     )],  
            (*vals)[lamb("top_truth_phi"     )], 
            (*vals)[lamb("top_truth_mass"    )], 
            (*vals)[lamb("top_truth_leptonic")], 
            {}, {}
    );  
    
    this -> top_nominal = BaseP(
            (*vals)[lamb("top_nominal_pt"      )],
            (*vals)[lamb("top_nominal_eta"     )],  
            (*vals)[lamb("top_nominal_phi"     )], 
            (*vals)[lamb("top_nominal_mass"    )], 
            (*vals)[lamb("top_nominal_leptonic")], 
            (*vals)[lamb("top_nominal_chi2"    )], 
            {}
    );  

    this -> top_masked_PR = BaseP(
            (*vals)[lamb("top_PR_masked_pt"      )],
            (*vals)[lamb("top_PR_masked_eta"     )],  
            (*vals)[lamb("top_PR_masked_phi"     )], 
            (*vals)[lamb("top_PR_masked_mass"    )], 
            (*vals)[lamb("top_PR_masked_leptonic")], 
            (*vals)[lamb("top_PR_masked_chi2"    )], 
            (*vals)[lamb("top_PR_masked_ranks"   )]
    );  

    this -> top_unmasked_PR = BaseP(
            (*vals)[lamb("top_PR_unmasked_pt"      )],
            (*vals)[lamb("top_PR_unmasked_eta"     )],  
            (*vals)[lamb("top_PR_unmasked_phi"     )], 
            (*vals)[lamb("top_PR_unmasked_mass"    )], 
            (*vals)[lamb("top_PR_unmasked_leptonic")], 
            (*vals)[lamb("top_PR_unmasked_chi2"    )], 
            (*vals)[lamb("top_PR_unmasked_ranks"   )]
    );
    
    this -> nom_recovery  = recovery(this -> top_nominal.size()    ); 
    this -> msk_recovery  = recovery(this -> top_masked_PR.size()  ); 
    this -> umsk_recovery = recovery(this -> top_unmasked_PR.size()); 

    this -> nom_error  = (*vals)[lamb("top_nominal_chi2")    ]; 
    this -> msk_error  = (*vals)[lamb("top_PR_masked_chi2")  ]; 
    this -> umsk_error = (*vals)[lamb("top_PR_unmasked_chi2")]; 

    this -> top_truth_mass    = (*vals)[lamb("top_truth_mass")      ]; 
    this -> top_nominal_mass  = (*vals)[lamb("top_nominal_mass")    ]; 
    this -> top_masked_mass   = (*vals)[lamb("top_PR_masked_mass")  ]; 
    this -> top_unmasked_mass = (*vals)[lamb("top_PR_unmasked_mass")]; 

    this -> top_masked_vPR   = (*vals)[lamb("top_PR_masked_ranks"  )]; 
    this -> top_unmasked_vPR = (*vals)[lamb("top_PR_unmasked_ranks")]; 

}



topefficiency_metric::topefficiency_metric(){this -> name = "topefficiency";}
topefficiency_metric* topefficiency_metric::clone(){return new topefficiency_metric();}
topefficiency_metric::~topefficiency_metric(){this -> vflush(&this -> evnts);}
void topefficiency_metric::add_event(Event* ev){
    this -> proc_evn[ev -> production][ev -> epoch].epoch = ev -> epoch;
    this -> proc_evn[ev -> production][ev -> epoch].evn[ev -> modelname].kfolds.push_back(ev); 

    particle_data_t* px = &this -> proc_evn[ev -> production][ev -> epoch].evn[ev -> modelname]; 

    px -> masked_kfolds_ntop.push_back(  ev ->  msk_recovery); 
    px -> nominal_kfolds_ntop.push_back( ev ->  nom_recovery); 
    px -> unmasked_kfolds_ntop.push_back(ev -> umsk_recovery); 
    
    merge_data(&px -> nominal_kfolds_chi2,  &ev ->  nom_error); 
    merge_data(&px -> masked_kfolds_chi2,   &ev ->  msk_error); 
    merge_data(&px -> unmasked_kfolds_chi2, &ev -> umsk_error); 

    merge_data(&px -> truth_kfolds_top_mass   , &ev ->  top_truth_mass); 
    merge_data(&px -> nominal_kfolds_top_mass , &ev ->  top_nominal_mass); 
    merge_data(&px -> masked_kfolds_top_mass  , &ev ->  top_masked_mass); 
    merge_data(&px -> unmasked_kfolds_top_mass, &ev ->  top_unmasked_mass); 
 
    merge_data(&px -> masked_kfolds_PR  , &ev ->  top_masked_vPR); 
    merge_data(&px -> unmasked_kfolds_PR, &ev ->  top_unmasked_vPR); 



    this -> evnts.push_back(ev); 
}

void topefficiency_metric::finalize(){
    std::map<process_t, std::map<long, epoch_t>>::iterator itr; 
    for (itr = this -> proc_evn.begin(); itr != this -> proc_evn.end(); ++itr){
        std::string prc = this -> to_string(itr -> first); 
        this -> generic_data[prc] = itr -> second; 
    }
}

std::string topefficiency_metric::to_string(process_t p){
    switch (p) {
        case process_t::t_tchan:   return "t_tchan";
        case process_t::t_schan:   return "t_schan";
        case process_t::tW:        return "tW";
        case process_t::ttbar:     return "ttbar";
        case process_t::tt_l:      return "tt_l";
        case process_t::tt_ll:     return "tt_ll";
        case process_t::tttt_SM:   return "tttt_SM";
        case process_t::tttt_m400: return "tttt_m400";
        case process_t::tttt_m500: return "tttt_m500";
        case process_t::tttt_m600: return "tttt_m600";
        case process_t::tttt_m700: return "tttt_m700";
        case process_t::tttt_m800: return "tttt_m800";
        case process_t::tttt_m900: return "tttt_m900";
        case process_t::tttt_m1000:return "tttt_m1000";
        case process_t::Z_ll:      return "Z_ll";
        case process_t::W_lv:      return "W_lv";
        case process_t::ZZ_qqll:   return "ZZ_qqll";
        case process_t::WZ_qqll:   return "WZ_qqll";
        case process_t::ttH:       return "ttH";
        case process_t::ttZ_qq:    return "ttZ_qq";
        case process_t::ttZ_vv:    return "ttZ_vv";
        case process_t::ttW:       return "ttW";
        case process_t::ZH:        return "ZH";
        case process_t::WH:        return "WH";
        case process_t::llll:      return "llll";
        case process_t::lllv:      return "lllv";
        case process_t::llvv:      return "llvv";
        case process_t::lvvv:      return "lvvv";
        default:                   return "invalid";
    }
}



process_t Event::process_sample(std::string name) {

    size_t start_pos = name.find("mc16");
    if (start_pos != std::string::npos){start_pos = name.find('.', start_pos) + 1;}
    else {start_pos = 0;}
    
    size_t end_pos = name.find('.', start_pos);
    if (end_pos == std::string::npos || start_pos >= name.length()) {
        return process_t::invalid; 
    }

    int dsid = -1;
    try {dsid = std::stoi(name.substr(start_pos, end_pos - start_pos));} 
    catch (...) {return process_t::invalid;}

    static const std::unordered_map<int, process_t> dsid_map = {
        // --- 4 Tops ---
        {312440, processtype::tttt::m400},
        {312441, processtype::tttt::m500},
        {312442, processtype::tttt::m600},
        {312443, processtype::tttt::m700},
        {312444, processtype::tttt::m800},
        {312445, processtype::tttt::m900},
        {312446, processtype::tttt::m1000},
        {412043, processtype::tttt::SM},

        // --- Higgs & Boson Associated ---
        {342284, processtype::WH},
        {342285, processtype::ZH},
        {346344, processtype::ttH}, // semilep 
        {346345, processtype::ttH}, // dilep

        // --- Diboson ---
        {363356, processtype::ZZ::qqll},
        {363358, processtype::WZ::qqll},

        // --- Z+Jets (Zll) ---
        {364100, processtype::Z::ll}, {364101, processtype::Z::ll}, {364102, processtype::Z::ll},
        {364103, processtype::Z::ll}, {364104, processtype::Z::ll}, {364105, processtype::Z::ll},
        {364106, processtype::Z::ll}, {364107, processtype::Z::ll}, {364108, processtype::Z::ll},
        {364109, processtype::Z::ll}, {364110, processtype::Z::ll}, {364111, processtype::Z::ll},
        {364112, processtype::Z::ll}, {364113, processtype::Z::ll}, {364114, processtype::Z::ll},
        {364115, processtype::Z::ll}, {364116, processtype::Z::ll}, {364117, processtype::Z::ll},
        {364118, processtype::Z::ll}, {364119, processtype::Z::ll}, {364120, processtype::Z::ll},
        {364121, processtype::Z::ll}, {364122, processtype::Z::ll}, {364123, processtype::Z::ll},
        {364124, processtype::Z::ll}, {364125, processtype::Z::ll}, {364126, processtype::Z::ll},
        {364127, processtype::Z::ll}, {364133, processtype::Z::ll}, {364135, processtype::Z::ll},
        {364136, processtype::Z::ll}, {364137, processtype::Z::ll}, {364138, processtype::Z::ll},
        {364139, processtype::Z::ll}, {364140, processtype::Z::ll}, {364141, processtype::Z::ll},

        // --- W+Jets (Wlnu) ---
        {364165, processtype::W::lv}, {364166, processtype::W::lv}, {364167, processtype::W::lv},
        {364168, processtype::W::lv}, {364169, processtype::W::lv}, {364181, processtype::W::lv},
        {364182, processtype::W::lv}, {364183, processtype::W::lv}, {364197, processtype::W::lv},

        // --- Multi-Lepton ---
        {364250, processtype::llll},
        {364253, processtype::lllv},
        {364254, processtype::llvv},

        // --- Top Pairs (Inclusive / Sliced) ---
        {407342, processtype::ttbar::inclusive}, {407343, processtype::ttbar::inclusive}, 
        {407344, processtype::ttbar::inclusive}, {407348, processtype::ttbar::inclusive}, 
        {407349, processtype::ttbar::inclusive}, {407350, processtype::ttbar::inclusive},
        {410470, processtype::ttbar::inclusive}, {411073, processtype::ttbar::inclusive},
        {411074, processtype::ttbar::inclusive}, {411075, processtype::ttbar::inclusive},
        {411082, processtype::ttbar::inclusive}, {412066, processtype::ttbar::inclusive},
        {412067, processtype::ttbar::inclusive}, {412068, processtype::ttbar::inclusive},

        // --- Top Associated (V) ---
        {410155, processtype::ttW},
        {410156, processtype::ttZ::vv}, // nunu
        {410157, processtype::ttZ::qq},

        // --- Top Pairs (Specific Decays) ---
        {410218, processtype::ttbar::ll}, {410219, processtype::ttbar::ll}, {410220, processtype::ttbar::ll},
        {410464, processtype::ttbar::l},  {410465, processtype::ttbar::ll}, {410472, processtype::ttbar::ll},
        {410480, processtype::ttbar::l},  {410482, processtype::ttbar::ll}, {410557, processtype::ttbar::l},
        {410558, processtype::ttbar::ll}, {411076, processtype::ttbar::ll}, {411077, processtype::ttbar::ll},
        {411078, processtype::ttbar::ll}, {411085, processtype::ttbar::ll}, {411086, processtype::ttbar::ll},
        {411087, processtype::ttbar::ll}, {412069, processtype::ttbar::ll}, {412070, processtype::ttbar::ll},
        {412071, processtype::ttbar::ll},

        // --- Single Top (tchan, schan, tW) ---
        {410560, processtype::t::tchannel}, // lept
        {410658, processtype::t::tchannel},
        {410659, processtype::t::tchannel},
        {411033, processtype::t::tchannel},
        {412004, processtype::t::tchannel},
        
        {410644, processtype::t::schannel}, 
        {410645, processtype::t::schannel},
        {411034, processtype::t::schannel},
        {411035, processtype::t::schannel},
        
        // Note: Grouping inclusive antitop (Wt) into tW 
        {410646, processtype::tW}, {410647, processtype::tW}, 
        {410654, processtype::tW}, {410655, processtype::tW}, 
        {411036, processtype::tW}, {411037, processtype::tW},
        {412002, processtype::tW}
    };
    auto it = dsid_map.find(dsid);
    if (it != dsid_map.end()){return it->second;}
    return process_t::invalid;
}                                     
