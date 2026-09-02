#include <metrics/samples.h>
#include <tools/tools.h>
#include <unordered_map>

process_t process_sample(std::string* name, int* dsids_){
    size_t start_pos = name -> find("mc16");
    if (start_pos != std::string::npos){start_pos = name -> find('.', start_pos) + 1;}
    else {start_pos = 0;}
    
    size_t end_pos = name -> find('.', start_pos);
    if (end_pos == std::string::npos || start_pos >= name -> length()){return process_t::invalid;}

    int dsid = -1;
    try {dsid = std::stoi(name -> substr(start_pos, end_pos - start_pos));} 
    catch (...) {return process_t::invalid;}
    if (dsids_){*dsids_ = dsid;}

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


//std::string mapping(std::string name, collector* cl){
//    if (tools::has_string(&name, "_singletop_" )){return "$t$"                      ;}
//    if (tools::has_string(&name, "_tchan_"     )){return "$t$"                      ;}
//    if (tools::has_string(&name, "_ttbarHT1k_" )){return "$t\\bar{t}$"              ;}
//    if (tools::has_string(&name, "_ttbar_"     )){return "$t\\bar{t}$"              ;}
//    if (tools::has_string(&name, "_ttbarHT1k5_")){return "$t\\bar{t}$"              ;}
//    if (tools::has_string(&name, "_ttbarHT6c_" )){return "$t\\bar{t}$"              ;}
//    if (tools::has_string(&name, "_tt_"        )){return "$t\\bar{t}$"              ;}
//    if (tools::has_string(&name, "_ttee."      )){return "$t\\bar{t}\\ell\\ell$"    ;}
//    if (tools::has_string(&name, "_ttmumu."    )){return "$t\\bar{t}\\ell\\ell$"    ;}
//    if (tools::has_string(&name, "_tttautau."  )){return "$t\\bar{t}\\ell\\ell$"    ;}
//    if (tools::has_string(&name, "_ttW."       )){return "$t\\bar{t}V$"             ;}
//    if (tools::has_string(&name, "_ttZnunu."   )){return "$t\\bar{t}V$"             ;}
//    if (tools::has_string(&name, "_ttZqq."     )){return "$t\\bar{t}V$"             ;}
//    if (tools::has_string(&name, "_ttH125_"    )){return "$t\\bar{t}H$"             ;}
//    if (tools::has_string(&name, "_Wt_"        )){return "$Wt$"                     ;}
//    if (tools::has_string(&name, "_tW."        )){return "$tV$"                     ;}
//    if (tools::has_string(&name, "_tW_"        )){return "$tV$"                     ;}
//    if (tools::has_string(&name, "_tZ."        )){return "$tV$"                     ;}
//    if (tools::has_string(&name, "_SM4topsNLO" )){return "$t\\bar{t}t\\bar{t}$"     ;}
//    if (tools::has_string(&name, "_WlvZqq"     )){return "$WZ$"                     ;}
//    if (tools::has_string(&name, "_WqqZll"     )){return "$WZ$"                     ;}
//    if (tools::has_string(&name, "_WqqZvv"     )){return "$WZ$"                     ;}
//    if (tools::has_string(&name, "_WplvWmqq"   )){return "$WW$"                     ;}
//    if (tools::has_string(&name, "_WpqqWmlv"   )){return "$WW$"                     ;}
//    if (tools::has_string(&name, "_ZqqZll"     )){return "$ZZ$"                     ;}
//    if (tools::has_string(&name, "_ZqqZvv"     )){return "$ZZ$"                     ;}
//    if (tools::has_string(&name, "_WH125."     )){return "$VH$"                     ;}
//    if (tools::has_string(&name, "_ZH125_"     )){return "$VH$"                     ;}
//    if (tools::has_string(&name, "_WH125_"     )){return "$VH$"                     ;}
//    if (tools::has_string(&name, "_Wenu_"      )){return "$V\\ell\\nu$"             ;}
//    if (tools::has_string(&name, "_Wmunu_"     )){return "$V\\ell\\nu$"             ;}
//    if (tools::has_string(&name, "_Wtaunu_"    )){return "$V\\ell\\nu$"             ;}
//    if (tools::has_string(&name, "_Zee_"       )){return "$V\\ell\\ell$"            ;}
//    if (tools::has_string(&name, "_Zmumu_"     )){return "$V\\ell\\ell$"            ;}
//    if (tools::has_string(&name, "_Ztautau_"   )){return "$V\\ell\\ell$"            ;}
//    if (tools::has_string(&name, "_llll"       )){return "$\\ell\\ell\\ell\\ell$"   ;}
//    if (tools::has_string(&name, "_lllv"       )){return "$\\ell\\ell\\ell\\nu$"    ;}
//    if (tools::has_string(&name, "_llvv"       )){return "$\\ell\\ell\\nu\\nu$"     ;}
//    if (tools::has_string(&name, "_lvvv"       )){return "$\\ell\\nu\\nu\\nu$"      ;}
//    return "undef"; 
//}


