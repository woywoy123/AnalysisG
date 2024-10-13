#include <meta.h>
meta::meta(){}
meta::~meta(){
    if (!this -> rpd){return;}
    delete this -> rpd; 
}

void meta::parse_json(std::string inpt){
    if (this -> rpd){return;}
    this -> rpd = new rapidjson::Document(); 
    this -> rpd -> Parse(inpt.c_str());
    if (this -> rpd -> HasParseError()){
        std::string f = inpt.substr(this -> rpd -> GetErrorOffset()-20, 20); 
        this -> replace(&inpt, f, f + ","); 
        this -> rpd -> Parse(inpt.c_str()); 
    }
    if (!this -> rpd -> HasParseError()){this -> compiler();}
    delete this -> rpd; 
    this -> rpd = nullptr; 
}

void meta::compiler(){
    rapidjson::Value* cfg = &(*this -> rpd)["inputConfig"]; 
    this -> meta_data.dsid = (*cfg)["dsid"].GetDouble(); 
    this -> meta_data.isMC = (*cfg)["isMC"].GetBool(); 
    this -> meta_data.AMITag = (*cfg)["amiTag"].GetString(); 
    this -> meta_data.derivationFormat = (*cfg)["derivationFormat"].GetString(); 
    rapidjson::Value* cfg_s = &(*this -> rpd)["configSettings"]; 
    if (cfg_s -> IsArray()){
        for (rapidjson::SizeType i(0); i < cfg_s -> Size(); ++i){
            std::string key = (*cfg_s)[i].GetString(); 
            std::string prm = (*cfg_s)[i].GetString(); 
            this -> meta_data.config[key] = prm; 
        }
    }

    int num_total = 0; 
    rapidjson::Value* files = &(*this -> rpd)["inputFiles"]; 
    for (rapidjson::SizeType i(0); i < files -> Size(); ++i){
        int n_ev = (*files)[i][1].GetInt();
        std::string fname = (*files)[i][0].GetString(); 
        std::vector<std::string> fname_v = this -> split(fname, "/"); 
        this -> meta_data.inputfiles[num_total] = fname_v[fname_v.size()-1]; 
        num_total += n_ev; 
    }
}

void meta::scan_data(TObject* obj){
    std::string obname = std::string(obj -> GetName()); 
    if (obname != "AnalysisTracking"){return;}
    this -> parse_json(this -> parse_string("jsonData", (TTree*)obj)); 
}

float meta::parse_float(std::string key, TTree* tr){
    tr -> GetEntry(0); 
    return tr -> GetLeaf(key.c_str()) -> GetValue();
}

std::string meta::parse_string(std::string key, TTree* tr){
    TBranch* lf = tr -> GetBranch(key.c_str()); 
    tr -> GetEntry(0); 
    std::string data = ""; 
    for (TObject* obj : *lf -> GetListOfLeaves()){
        TLeaf* lx = (TLeaf*)obj; 
        char** datar = reinterpret_cast<char**>(lx -> GetValuePointer()); 
        data += std::string(*datar); 
    }
    return data; 
}

void meta::scan_sow(TObject* obj){
    std::string obname = std::string(obj -> GetName()); 
    if (obj -> InheritsFrom("TTree")){
        TTree* r = (TTree*)obj; 
        weights_t* wg = &this -> meta_data.misc[obname]; 
        wg -> dsid = this -> parse_float("dsid", r); 
        wg -> isAFII = this -> parse_float("isAFII", r); 
        wg -> total_events_weighted = this -> parse_float("totalEventsWeighted", r); 
        wg -> total_events = this -> parse_float("totalEvents", r); 
        wg -> processed_events = this -> parse_float("processedEvents", r); 
        wg -> processed_events_weighted = this -> parse_float("processedEventsWeighted", r); 
        wg -> processed_events_weighted_squared = this -> parse_float("processedEventsWeightedSquared", r); 
        wg -> generator = this -> parse_string("generators", r);  
        wg -> ami_tag = this -> parse_string("AMITag", r);  
        return;
    }
    if (obj -> InheritsFrom("TH1")){
        TH1F* hs = (TH1F*)obj;
        TAxis* xs = hs -> GetXaxis(); 
        weights_t* wg = &this -> meta_data.misc[obname]; 
        for (size_t x(0); x < xs -> GetNbins(); ++x){
            wg -> hist_data[xs -> GetBinLabel(x+1)] = hs -> GetBinContent(x+1);
        }
    }
}

std::string meta::hash(std::string fname){
    std::vector<std::string> spl = this -> split(fname, "/"); 
    size_t x = spl.size(); 
    if (x == 0){return this -> tools::hash(fname);}
    return this -> tools::hash(spl[x-1]);
}
