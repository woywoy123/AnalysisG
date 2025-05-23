#include <meta.h>
#include <TTreeReader.h>

meta::meta(){

    this -> isMC.set_getter(this -> get_isMC);
    this -> found.set_getter(this -> get_found);
    this -> eventNumber.set_getter(this -> get_eventNumber);
    this -> event_index.set_getter(this -> get_event_index);
    this -> totalSize.set_getter(this -> get_totalSize);
    this -> kfactor.set_getter(this -> get_kfactor);
    this -> ecmEnergy.set_getter(this -> get_ecmEnergy);
    this -> genFiltEff.set_getter(this -> get_genFiltEff);
    this -> completion.set_getter(this -> get_completion);
    this -> beam_energy.set_getter(this -> get_beam_energy);

    this -> cross_section_nb.set_getter(this -> get_cross_section_nb);
    this -> cross_section_pb.set_getter(this -> get_cross_section_pb);
    this -> cross_section_fb.set_getter(this -> get_cross_section_fb);

    this -> campaign_luminosity.set_getter(this -> get_campaign_luminosity);
    this -> dsid.set_getter(this -> get_dsid);
    this -> nFiles.set_getter(this -> get_nFiles);
    this -> totalEvents.set_getter(this -> get_totalEvents);
    this -> datasetNumber.set_getter(this -> get_datasetNumber);
    this -> derivationFormat.set_getter(this -> get_derivationFormat);
    this -> AMITag.set_getter(this -> get_AMITag);
    this -> generators.set_getter(this -> get_generators);
    this -> identifier.set_getter(this -> get_identifier);
    this -> DatasetName.set_getter(this -> get_DatasetName);
    this -> prodsysStatus.set_getter(this -> get_prodsysStatus);
    this -> dataType.set_getter(this -> get_dataType);
    this -> version.set_getter(this -> get_version);
    this -> PDF.set_getter(this -> get_PDF);
    this -> AtlasRelease.set_getter(this -> get_AtlasRelease);
    this -> principalPhysicsGroup.set_getter(this -> get_principalPhysicsGroup);
    this -> physicsShort.set_getter(this -> get_physicsShort);
    this -> generatorName.set_getter(this -> get_generatorName);
    this -> geometryVersion.set_getter(this -> get_geometryVersion);
    this -> conditionsTag.set_getter(this -> get_conditionsTag);
    this -> generatorTune.set_getter(this -> get_generatorTune);
    this -> amiStatus.set_getter(this -> get_amiStatus);
    this -> beamType.set_getter(this -> get_beamType);
    this -> productionStep.set_getter(this -> get_productionStep);
    this -> projectName.set_getter(this -> get_projectName);
    this -> statsAlgorithm.set_getter(this -> get_statsAlgorithm);
    this -> genFilterNames.set_getter(this -> get_genFilterNames);
    this -> file_type.set_getter(this -> get_file_type);
    this -> sample_name.set_getter(this -> get_sample_name);
    this -> logicalDatasetName.set_getter(this -> get_logicalDatasetName);
    this -> campaign.set_getter(this -> get_campaign);
    this -> keywords.set_getter(this -> get_keywords);
    this -> weights.set_getter(this -> get_weights);
    this -> keyword.set_getter(this -> get_keyword);
    this -> fileGUID.set_getter(this -> get_fileGUID);
    this -> events.set_getter(this -> get_events);
    this -> run_number.set_getter(this -> get_run_number);
    this -> fileSize.set_getter(this -> get_fileSize);
    this -> inputrange.set_getter(this -> get_inputrange);
    this -> inputfiles.set_getter(this -> get_inputfiles);
    this -> LFN.set_getter(this -> get_LFN);
    this -> misc.set_getter(this -> get_misc);
    this -> config.set_getter(this -> get_config);
    this -> sum_of_weights.set_getter(this -> get_sum_of_weights); 

    this -> isMC.set_object(this); 
    this -> found.set_object(this); 
    this -> eventNumber.set_object(this); 
    this -> event_index.set_object(this); 
    this -> totalSize.set_object(this); 
    this -> kfactor.set_object(this); 
    this -> ecmEnergy.set_object(this); 
    this -> genFiltEff.set_object(this); 
    this -> completion.set_object(this); 
    this -> beam_energy.set_object(this); 

    this -> cross_section_nb.set_object(this); 
    this -> cross_section_pb.set_object(this); 
    this -> cross_section_fb.set_object(this); 
 
    this -> campaign_luminosity.set_object(this); 
    this -> dsid.set_object(this); 
    this -> nFiles.set_object(this); 
    this -> totalEvents.set_object(this); 
    this -> datasetNumber.set_object(this); 
    this -> derivationFormat.set_object(this); 
    this -> AMITag.set_object(this); 
    this -> generators.set_object(this); 
    this -> identifier.set_object(this); 
    this -> DatasetName.set_object(this); 
    this -> prodsysStatus.set_object(this); 
    this -> dataType.set_object(this); 
    this -> version.set_object(this); 
    this -> PDF.set_object(this); 
    this -> AtlasRelease.set_object(this); 
    this -> principalPhysicsGroup.set_object(this); 
    this -> physicsShort.set_object(this); 
    this -> generatorName.set_object(this); 
    this -> geometryVersion.set_object(this); 
    this -> conditionsTag.set_object(this); 
    this -> generatorTune.set_object(this); 
    this -> amiStatus.set_object(this); 
    this -> beamType.set_object(this); 
    this -> productionStep.set_object(this); 
    this -> projectName.set_object(this); 
    this -> statsAlgorithm.set_object(this); 
    this -> genFilterNames.set_object(this); 
    this -> file_type.set_object(this); 
    this -> sample_name.set_object(this); 
    this -> logicalDatasetName.set_object(this); 
    this -> campaign.set_object(this); 
    this -> keywords.set_object(this); 
    this -> weights.set_object(this); 
    this -> keyword.set_object(this); 
    this -> fileGUID.set_object(this); 
    this -> events.set_object(this); 
    this -> run_number.set_object(this); 
    this -> fileSize.set_object(this); 
    this -> inputrange.set_object(this); 
    this -> inputfiles.set_object(this); 
    this -> LFN.set_object(this); 
    this -> misc.set_object(this); 
    this -> config.set_object(this); 
    this -> sum_of_weights.set_object(this); 
    this -> prefix = "meta"; 
}

meta::~meta(){
    if (!this -> rpd){return;}
    delete this -> rpd; 
}

void meta::parse_json(std::string inpt){
    if (this -> rpd){return;}
    this -> rpd = new rapidjson::Document(); 
    this -> rpd -> Parse(inpt.c_str());
    if (this -> rpd -> HasParseError()){
        int inx = this -> rpd -> GetErrorOffset(); 
        if (inx > 20){inx -= 20;}
        std::string f  = inpt.substr(inx, 20); 
        std::string fn = f; 
        if (!this -> ends_with(&f, ",\n")){this -> replace(&fn, "\n", ",\n");}
        this -> replace(&inpt, f, fn); 

        delete this -> rpd; 
        this -> rpd = new rapidjson::Document(); 
        this -> rpd -> Parse(inpt.c_str()); 
    }
    this -> compiler();
    delete this -> rpd; 
    this -> rpd = nullptr; 
}

void meta::compiler(){
    rapidjson::Value* cfg = &(*this -> rpd)["inputConfig"]; 
    if (!cfg){return;}
    if (cfg -> HasMember("dsid")){this -> meta_data.dsid = (*cfg)["dsid"].GetDouble();}
    if (cfg -> HasMember("isMC")){this -> meta_data.isMC = (*cfg)["isMC"].GetBool();}
    if (cfg -> HasMember("derivationFormat")){this -> meta_data.derivationFormat = (*cfg)["derivationFormat"].GetString();}

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

    if (cfg -> HasMember("amiTag")){this -> meta_data.AMITag = (*cfg)["amiTag"].GetString();}
    else if (files){
        for (rapidjson::SizeType i(0); i < files -> Size(); ++i){
            std::string fname = (*files)[i][0].GetString(); 
            std::vector<std::string> fname_v = this -> split(fname, "/"); 
            fname = fname_v[fname_v.size()-2]; 
            fname_v = this -> split(fname, "."); 
            this -> meta_data.AMITag = fname_v[fname_v.size()-1]; 
            break; 
        }
        if (!this -> has_string(&this -> meta_data.AMITag, "e")){this -> meta_data.AMITag = "";}
    }
    else {
        std::vector<std::string> spl = this -> split(this -> meta_data.sample_name, "/"); 
        size_t x = spl.size(); 
        if (x >= 2){this -> meta_data.AMITag = spl[x-2];}
    }

}

void meta::scan_data(TObject* obj){
    gErrorIgnoreLevel = 6001;
    std::string obname = std::string(obj -> GetName()); 
    if (obname == "AnalysisTracking"){return this -> parse_json(this -> parse_string("jsonData", (TTree*)obj));}
    else if (obname == "MetaData"){((TTree*)obj) -> SetBranchAddress("MetaData", &this -> meta_data);}
    else {return this -> scan_sow(obj);}
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
        TTree* _t = (TTree*)obj; 
        weights_t* wg = &this -> meta_data.misc[obname]; 
        if (obname == "AnalysisTracking"){
            wg -> dsid = this -> parse_float("dsid", _t); 
            wg -> isAFII = this -> parse_float("isAFII", _t); 
            wg -> total_events_weighted = this -> parse_float("totalEventsWeighted", _t); 
            wg -> total_events = this -> parse_float("totalEvents", _t); 
            wg -> processed_events = this -> parse_float("processedEvents", _t); 
            wg -> processed_events_weighted = this -> parse_float("processedEventsWeighted", _t); 
            wg -> processed_events_weighted_squared = this -> parse_float("processedEventsWeightedSquared", _t); 
            wg -> generator = this -> parse_string("generators", _t);  
            wg -> ami_tag = this -> parse_string("AMITag", _t);  
            return; 
        }
        if (obname == "EventLoop_FileExecuted"){
            TTreeReader _r = TTreeReader(obj -> GetName()); 
            TTreeReaderValue<TString> dr(_r, "file");  
            std::vector<std::string> data = {}; 
            while(_r.Next()){data.push_back(std::string(*dr));}
            for (size_t x(0); x < data.size(); ++x){this -> meta_data.inputfiles[x] = data[x];}
        }
        return;
    }

    if (obj -> InheritsFrom("TH1")){
        TH1F* hs = (TH1F*)obj;
        TAxis* xs = hs -> GetXaxis(); 
        weights_t* wg = &this -> meta_data.misc[obname]; 
        
        bool capture = false; 
        for (int x(0); x < xs -> GetNbins(); ++x){
            std::string lbl = xs -> GetBinLabel(x+1); 
            float val = hs -> GetBinContent(x+1); 
            if (lbl == "Initial events"){wg -> processed_events = val;}
            if (lbl == "Initial sum of weights"){wg -> processed_events_weighted = val;}
            if (lbl == "Initial sum of weights squared"){wg -> processed_events_weighted_squared = val;}
            if (this -> has_string(&lbl, "mc")){this -> meta_data.campaign = lbl; capture = true;}
            wg -> hist_data[xs -> GetBinLabel(x+1)] = hs -> GetBinContent(x+1);
        }
        if (!capture){return;}
        this -> meta_data.dsid = std::atoi(xs -> GetBinLabel(3)); 
        this -> meta_data.AMITag = xs -> GetBinLabel(4);
    }
}

std::string meta::hash(std::string fname){
    std::vector<std::string> spl = this -> split(fname, "/"); 
    size_t x = spl.size(); 
    if (x == 0){return this -> tools::hash(fname);}
    return this -> tools::hash(spl[x-1]);
}

void meta::get_sum_of_weights(double* val, meta* m){
    std::map<std::string, weights_t> data = m -> misc; 
    std::map<std::string, weights_t>::iterator itr = data.begin(); 
    for (; itr != data.end(); ++itr){
        if (itr -> second.processed_events_weighted == -1){continue;}
        *val = itr -> second.processed_events_weighted; 
        return; 
    }
}

const folds_t* meta::get_tags(std::string hash_){
    if (!this -> folds){return nullptr;}
    for (size_t x(0); x < this -> folds -> size(); ++x){
        if (hash_ != std::string((*this -> folds)[x].hash)){continue;}
        return &(*this -> folds)[x]; 
    } 
    return nullptr; 
}

void meta::get_cross_section_fb(double* val, meta* m){
    *val = m -> cross_section_nb*1000000;
}

void meta::get_cross_section_pb(double* val, meta* m){
    *val = m -> cross_section_nb*1000;
}

void meta::get_cross_section_nb(double* val, meta* m){
    double tml = m -> meta_data.crossSection_mean; 
    if (tml == 0){tml = m -> meta_data.crossSection_mean;}
    *val = tml;
}

void meta::get_campaign(std::string* val, meta* m){
    m -> replace(&m -> meta_data.campaign, " ", ""); 
    *val = m -> meta_data.campaign; 
}

void meta::get_isMC(bool* val, meta* m){*val = m -> meta_data.isMC;}
void meta::get_found(bool* val, meta* m){*val = m -> meta_data.found;}
void meta::get_eventNumber(double* val, meta* m){*val = m -> meta_data.eventNumber;}
void meta::get_event_index(double* val, meta* m){*val = m -> meta_data.event_index;}
void meta::get_totalSize(double* val, meta* m){*val = m -> meta_data.totalSize;}
void meta::get_kfactor(double* val, meta* m){*val = m -> meta_data.kfactor;}
void meta::get_ecmEnergy(double* val, meta* m){*val = m -> meta_data.ecmEnergy;}
void meta::get_genFiltEff(double* val, meta* m){*val = m -> meta_data.genFiltEff;}
void meta::get_completion(double* val, meta* m){*val = m -> meta_data.completion;}
void meta::get_beam_energy(double* val, meta* m){*val = m -> meta_data.beam_energy;}

void meta::get_campaign_luminosity(double* val, meta* m){*val = m -> meta_data.campaign_luminosity;}
void meta::get_dsid(unsigned int* val, meta* m){*val = m -> meta_data.dsid;}
void meta::get_nFiles(unsigned int* val, meta* m){*val = m -> meta_data.nFiles;}
void meta::get_totalEvents(unsigned int* val, meta* m){*val = m -> meta_data.totalEvents;}
void meta::get_datasetNumber(unsigned int* val, meta* m){*val = m -> meta_data.datasetNumber;}
void meta::get_derivationFormat(std::string* val, meta* m){*val = m -> meta_data.derivationFormat;}
void meta::get_AMITag(std::string* val, meta* m){*val = m -> meta_data.AMITag;}
void meta::get_generators(std::string* val, meta* m){*val = m -> meta_data.generators;}
void meta::get_identifier(std::string* val, meta* m){*val = m -> meta_data.identifier;}
void meta::get_DatasetName(std::string* val, meta* m){*val = m -> meta_data.DatasetName;}
void meta::get_prodsysStatus(std::string* val, meta* m){*val = m -> meta_data.prodsysStatus;}
void meta::get_dataType(std::string* val, meta* m){*val = m -> meta_data.dataType;}
void meta::get_version(std::string* val, meta* m){*val = m -> meta_data.version;}
void meta::get_PDF(std::string* val, meta* m){*val = m -> meta_data.PDF;}
void meta::get_AtlasRelease(std::string* val, meta* m){*val = m -> meta_data.AtlasRelease;}
void meta::get_principalPhysicsGroup(std::string* val, meta* m){*val = m -> meta_data.principalPhysicsGroup;}
void meta::get_physicsShort(std::string* val, meta* m){*val = m -> meta_data.physicsShort;}
void meta::get_generatorName(std::string* val, meta* m){*val = m -> meta_data.generatorName;}
void meta::get_geometryVersion(std::string* val, meta* m){*val = m -> meta_data.geometryVersion;}
void meta::get_conditionsTag(std::string* val, meta* m){*val = m -> meta_data.conditionsTag;}
void meta::get_generatorTune(std::string* val, meta* m){*val = m -> meta_data.generatorTune;}
void meta::get_amiStatus(std::string* val, meta* m){*val = m -> meta_data.amiStatus;}
void meta::get_beamType(std::string* val, meta* m){*val = m -> meta_data.beamType;}
void meta::get_productionStep(std::string* val, meta* m){*val = m -> meta_data.productionStep;}
void meta::get_projectName(std::string* val, meta* m){*val = m -> meta_data.projectName;}
void meta::get_statsAlgorithm(std::string* val, meta* m){*val = m -> meta_data.statsAlgorithm;}
void meta::get_genFilterNames(std::string* val, meta* m){*val = m -> meta_data.genFilterNames;}
void meta::get_file_type(std::string* val, meta* m){*val = m -> meta_data.file_type;}
void meta::get_sample_name(std::string* val, meta* m){*val = m -> meta_data.sample_name;}
void meta::get_logicalDatasetName(std::string* val, meta* m){*val = m -> meta_data.logicalDatasetName;}

void meta::get_keywords(std::vector<std::string>* val, meta* m){*val = m -> meta_data.keywords;}
void meta::get_weights(std::vector<std::string>* val, meta* m){*val = m -> meta_data.weights;}
void meta::get_keyword(std::vector<std::string>* val, meta* m){*val = m -> meta_data.keyword;}
void meta::get_fileGUID(std::vector<std::string>* val, meta* m){*val = m -> meta_data.fileGUID;}
void meta::get_events(std::vector<int>* val, meta* m){*val = m -> meta_data.events;}
void meta::get_run_number(std::vector<int>* val, meta* m){*val = m -> meta_data.run_number;}
void meta::get_fileSize(std::vector<double>* val, meta* m){*val = m -> meta_data.fileSize;}
void meta::get_inputrange(std::map<int, int>* val, meta* m){*val = m -> meta_data.inputrange;}
void meta::get_inputfiles(std::map<int, std::string>* val, meta* m){*val = m -> meta_data.inputfiles;}
void meta::get_LFN(std::map<std::string, int>* val, meta* m){*val = m -> meta_data.LFN;}
void meta::get_misc(std::map<std::string, weights_t>* val, meta* m){*val = m -> meta_data.misc;}
void meta::get_config(std::map<std::string, std::string>* val, meta* m){*val = m -> meta_data.config;}
