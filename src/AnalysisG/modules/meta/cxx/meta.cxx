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

    if (!this -> rpd -> HasParseError()){return this -> compiler();}
    delete this -> rpd; 
    this -> rpd = nullptr; 
}

void meta::compiler(){
    rapidjson::Value* cfg = &(*this -> rpd)["inputConfig"]; 
    this -> meta_data.dsid = (*cfg)["dsid"].GetDouble(); 
    this -> meta_data.isMC = (*cfg)["isMC"].GetBool(); 
    this -> meta_data.derivationFormat = (*cfg)["derivationFormat"].GetString(); 

    rapidjson::Value* cfg_s = &(*this -> rpd)["configSettings"]; 
    if (!cfg_s -> IsArray()){return;}
    for (rapidjson::SizeType i(0); i < cfg_s -> Size(); ++i){
        std::string key = (*cfg_s)[i].GetString(); 
        std::string prm = (*cfg_s)[i].GetString(); 
        this -> meta_data.config[key] = prm; 
    }

    rapidjson::Value* files = &(*this -> rpd)["inputFiles"]; 
    for (rapidjson::SizeType i(0); i < files -> Size(); ++i){
        int n_ev = (*files)[i][1].GetInt();
        std::string fname = (*files)[i][0].GetString(); 
        std::vector<std::string> fname_v = this -> split(fname, "/"); 
        this -> meta_data.inputfiles[n_ev] = fname_v[fname_v.size()-1]; 
    }
}



