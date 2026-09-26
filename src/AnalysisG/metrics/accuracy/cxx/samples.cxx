#include <metrics/samples.h>
#include <tools/tools.h>
#include <unordered_map>
#include <map>

process_t process_sample(std::string* name, int* dsids_){
    int dsid = -1;
    if (name){
        size_t start_pos = name -> find("mc16");
        if (start_pos != std::string::npos){start_pos = name -> find('.', start_pos) + 1;}
        else {start_pos = 0;}
        
        size_t end_pos = name -> find('.', start_pos);
        if (end_pos == std::string::npos || start_pos >= name -> length()){return process_t::invalid;}

        try {dsid = std::stoi(name -> substr(start_pos, end_pos - start_pos));} 
        catch (...) {return process_t::invalid;}
        if (dsids_){*dsids_ = dsid;}
    }
    else if (dsids_){dsid = *dsids_;}
    else {return process_t::invalid;}

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
        {410218, processtype::ttbar::ll}, 
        {410219, processtype::ttbar::ll}, 
        {410220, processtype::ttbar::ll},
        {410464, processtype::ttbar::l},  
        {410465, processtype::ttbar::ll}, 
        {410472, processtype::ttbar::ll},
        {410480, processtype::ttbar::l},  
        {410482, processtype::ttbar::ll}, 
        {410557, processtype::ttbar::l},
        {410558, processtype::ttbar::ll}, 
        {411076, processtype::ttbar::ll}, 
        {411077, processtype::ttbar::ll},
        {411078, processtype::ttbar::ll}, 
        {411085, processtype::ttbar::ll}, 
        {411086, processtype::ttbar::ll},
        {411087, processtype::ttbar::ll}, 
        {412069, processtype::ttbar::ll}, 
        {412070, processtype::ttbar::ll},
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

std::string process_string(process_t name){
    switch (name){
        case process_t::t_tchan:    return "$t_{\\text{t-channel}}$"; 
        case process_t::t_schan:    return "$t_{\\text{s-channel}}$"; 
        case process_t::tW:         return "$tW$";  
        case process_t::ttbar:      return "$t\\bar{t}$";  
        case process_t::tt_l:       return "$t\\bar{t} \\rightarrow \\ell$";  
        case process_t::tt_ll:      return "$t\\bar{t} \\rightarrow \\ell \\bar{\\ell}$";  
        case process_t::ttH:        return "$t\\bar{t}H$";  
        case process_t::ttW:        return "$t\\bar{t}W$";  
        case process_t::ttZ_qq:     return "$t\\bar{t}Z \\rightarrow q \\bar{q} $";  
        case process_t::ttZ_vv:     return "$t\\bar{t}Z \\rightarrow \\nu \\bar{\\nu}$"; 
        case process_t::tttt_SM:    return "$t\\bar{t}t\\bar{t}$ (SM)";  
        case process_t::tttt_m400:  return "$t\\bar{t}t\\bar{t}$ (400)";  
        case process_t::tttt_m500:  return "$t\\bar{t}t\\bar{t}$ (500)";   
        case process_t::tttt_m600:  return "$t\\bar{t}t\\bar{t}$ (600)";  
        case process_t::tttt_m700:  return "$t\\bar{t}t\\bar{t}$ (700)";  
        case process_t::tttt_m800:  return "$t\\bar{t}t\\bar{t}$ (800)";  
        case process_t::tttt_m900:  return "$t\\bar{t}t\\bar{t}$ (900)";  
        case process_t::tttt_m1000: return "$t\\bar{t}t\\bar{t}$ (1000)";  
        case process_t::Z_ll:       return "$Z \\rightarrow \\ell \\bar{\\ell}$"; 
        case process_t::W_lv:       return "$W \\rightarrow \\ell \\bar{\\nu}$";  
        case process_t::ZZ_qqll:    return "$ZZ$";  
        case process_t::WZ_qqll:    return "$WZ$";  
        case process_t::ZH:         return "$ZH$";  
        case process_t::WH:         return "$WH$"; 
        case process_t::llll:       return "$\\ell\\ell\\ell\\ell$";  
        case process_t::lllv:       return "$\\ell\\ell\\ell\\nu$" ; 
        case process_t::llvv:       return "$\\ell\\ell\\nu\\nu$"  ; 
        case process_t::lvvv:       return "$\\ell\\nu\\nu\\nu$"   ; 
        default:                    return "unknown"; 
    }
    return "undefined"; 
}


