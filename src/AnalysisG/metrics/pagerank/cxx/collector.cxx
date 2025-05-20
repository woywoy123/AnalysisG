#include <metrics/pagerank.h>

collector::collector(){}
collector::~collector(){}

void collector::set_index(long idx_){this -> idx = idx_;}

kinematic_t collector::create_kinematic(
        double px, double py,  double pz,  double mass,
        double pt, double eta, double phi, double energy,  
        int num_nodes, double score
){
    return kinematic_t{px, py, pz, mass, pt, eta, phi, energy, num_nodes, score};  
}

void collector::add_truth(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch){
    this -> data[model][epoch][kfold].data[mode][this -> idx].truth.push_back(*p);
}

void collector::add_pagerank(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch){
    this -> data[model][epoch][kfold].data[mode][this -> idx].pageranked.push_back(*p);
}

void collector::add_nominal(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch){
    this -> data[model][epoch][kfold].data[mode][this -> idx].nominal.push_back(*p);
}

void collector::add_process(int prc, std::string mode, std::string model, int kfold, int epoch){
    this -> data[model][epoch][kfold].data[mode][this -> idx].process_mapping = prc;
}

bool collector::add_meta(std::string data, std::string model, int kfold, int epoch){
    if (this -> data[model][epoch][kfold].meta_data.size()){return false;}
    this -> data[model][epoch][kfold].meta_data = data; 
    return true;
}

void collector::add_file_map(std::string fname, long idx_, std::string mode, std::string model, int kfold, int epoch, bool stat){
    if (stat){this -> data[model][epoch][kfold].file_stat[mode][fname] = idx_;}
    else {this -> data[model][epoch][kfold].file_map[mode][fname] = idx_;}
}

std::string mapping(std::string name, collector* cl){
    if (cl -> has_string(&name, "_singletop_" )){return "$t$"                      ;}
    if (cl -> has_string(&name, "_tchan_"     )){return "$t$"                      ;}
    if (cl -> has_string(&name, "_ttbarHT1k_" )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttbar_"     )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttbarHT1k5_")){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttbarHT6c_" )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_tt_"        )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttee."      )){return "$t\\bar{t}\\ell\\ell$"    ;}
    if (cl -> has_string(&name, "_ttmumu."    )){return "$t\\bar{t}\\ell\\ell$"    ;}
    if (cl -> has_string(&name, "_tttautau."  )){return "$t\\bar{t}\\ell\\ell$"    ;}
    if (cl -> has_string(&name, "_ttW."       )){return "$t\\bar{t}V$"             ;}
    if (cl -> has_string(&name, "_ttZnunu."   )){return "$t\\bar{t}V$"             ;}
    if (cl -> has_string(&name, "_ttZqq."     )){return "$t\\bar{t}V$"             ;}
    if (cl -> has_string(&name, "_ttH125_"    )){return "$t\\bar{t}H$"             ;}
    if (cl -> has_string(&name, "_Wt_"        )){return "$Wt$"                     ;}
    if (cl -> has_string(&name, "_tW."        )){return "$tV$"                     ;}
    if (cl -> has_string(&name, "_tW_"        )){return "$tV$"                     ;}
    if (cl -> has_string(&name, "_tZ."        )){return "$tV$"                     ;}
    if (cl -> has_string(&name, "_SM4topsNLO" )){return "$t\\bar{t}t\\bar{t}$"     ;}
    if (cl -> has_string(&name, "_WlvZqq"     )){return "$WZ$"                     ;}
    if (cl -> has_string(&name, "_WqqZll"     )){return "$WZ$"                     ;}
    if (cl -> has_string(&name, "_WqqZvv"     )){return "$WZ$"                     ;}
    if (cl -> has_string(&name, "_WplvWmqq"   )){return "$WW$"                     ;}
    if (cl -> has_string(&name, "_WpqqWmlv"   )){return "$WW$"                     ;}
    if (cl -> has_string(&name, "_ZqqZll"     )){return "$ZZ$"                     ;}
    if (cl -> has_string(&name, "_ZqqZvv"     )){return "$ZZ$"                     ;}
    if (cl -> has_string(&name, "_WH125."     )){return "$VH$"                     ;}
    if (cl -> has_string(&name, "_ZH125_"     )){return "$VH$"                     ;}
    if (cl -> has_string(&name, "_WH125_"     )){return "$VH$"                     ;}
    if (cl -> has_string(&name, "_Wenu_"      )){return "$V\\ell\\nu$"             ;}
    if (cl -> has_string(&name, "_Wmunu_"     )){return "$V\\ell\\nu$"             ;}
    if (cl -> has_string(&name, "_Wtaunu_"    )){return "$V\\ell\\nu$"             ;}
    if (cl -> has_string(&name, "_Zee_"       )){return "$V\\ell\\ell$"            ;}
    if (cl -> has_string(&name, "_Zmumu_"     )){return "$V\\ell\\ell$"            ;}
    if (cl -> has_string(&name, "_Ztautau_"   )){return "$V\\ell\\ell$"            ;}
    if (cl -> has_string(&name, "_llll"       )){return "$\\ell\\ell\\ell\\ell$"   ;}
    if (cl -> has_string(&name, "_lllv"       )){return "$\\ell\\ell\\ell\\nu$"    ;}
    if (cl -> has_string(&name, "_llvv"       )){return "$\\ell\\ell\\nu\\nu$"     ;}
    if (cl -> has_string(&name, "_lvvv"       )){return "$\\ell\\nu\\nu\\nu$"      ;}
    return "undef"; 
}
