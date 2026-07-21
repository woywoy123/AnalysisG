#include <AnalysisG/analysis.h>
#include <ROOT/RDataFrame.hxx>

void analysis::build_events(){
    auto lamb = [this](
            event_template* evnt_comp, 
            std::vector<std::string>* trees, std::vector<std::string>* branches, 
            std::vector<std::string>* leaves, std::vector<std::string>* files, tracing_t* tr
    ){
        if (!files -> size()){return;}
        io* rdr = new io(); 
        rdr -> shush = true; 
        rdr -> import_settings(&this -> m_settings); 
        rdr -> trees    = *trees; 
        rdr -> branches = *branches; 
        rdr -> leaves   = *leaves; 
        for (size_t x(0); x < files -> size(); ++x){rdr -> root_files[files -> at(x)] = true;}
        rdr -> check_root_file_paths();
        event_template* evn = evnt_comp -> clone(); 

        size_t num_evn = (*tr -> maxlength);  
        std::map<std::string, event_template*> evnts; 
        std::map<std::string, event_template*>::iterator tx; 
        std::map<std::string, data_t*>* io_handle = rdr -> get_data(); 

        while (tr -> index() < num_evn){
            evnts = evn -> build_event(io_handle);
            if (!evnts.size()){continue;}
            tr -> next(); 

            tx = evnts.begin(); 
            std::string fname = tx -> second -> filename; 
            std::string label = this -> file_labels[fname]; 

            meta* meta_ = nullptr; 
            bool ext = this -> meta_data.count(fname); 
            if (ext){meta_ = this -> meta_data[fname];}
            else {meta_ = rdr -> meta_data[fname];}

            if (this -> tracer -> add_meta_data(meta_, fname) && !ext){
                meta_ -> folds = this -> tags; 
                this -> meta_data[fname] = meta_; 
                rdr -> meta_data[fname] = nullptr;
            }

            std::vector<std::string> tmp = this -> split(fname, "/"); 
            tr -> message(tmp[tmp.size()-1]); 
            for (; tx != evnts.end(); ++tx){
                if (!this -> tracer -> add_event(tx -> second, label)){continue;} 
                this -> pflush(&tx -> second); 
            }
        }
        this -> pflush(&rdr); 
        this -> pflush(&evn); 
        tr -> finished(); 
    };  
            
    std::vector<std::string>* trees_    = &this -> reader -> trees; 
    std::vector<std::string>* branches_ = &this -> reader -> branches; 
    std::vector<std::string>* leaves_   = &this -> reader -> leaves; 
    this -> reader -> import_settings(&this -> m_settings); 

    event_template* event_f = nullptr; 
    std::map<std::string, std::string>::iterator itf = this -> file_labels.begin(); 
    for (; itf != this -> file_labels.end(); ++itf){
        if (!this -> event_labels.count(itf -> second)){continue;}
        if (this -> skip_event_build[itf -> first]){continue;}
        this -> reader -> root_files[this -> absolute_path(itf -> first)] = true;

        if (event_f){continue;}
        event_f = this -> event_labels[itf -> second]; 
        std::vector<std::string> dt = event_f -> trees; 
        trees_ -> insert(trees_ -> end(), dt.begin(), dt.end()); 

        dt = event_f -> branches; 
        branches_ -> insert(branches_ -> end(), dt.begin(), dt.end()); 

        dt = event_f -> leaves; 
        leaves_ -> insert(leaves_ -> end(), dt.begin(), dt.end()); 
    } 
    this -> reader -> scan_keys();
    if (!this -> reader -> root_files.size()){return this -> info("Skipping event building.");}
    if (!event_f){return this -> warning("Missing Event Implementation for specified samples!");}
    this -> success("+============================+"); 
    this -> success("|   Starting Event Builder   |");
    this -> success("+============================+"); 
    this -> reader -> check_root_file_paths(); 

    size_t nevents = 0;
    std::map<std::string, long> len = this -> reader -> root_size(); 
    std::map<std::string, long>::iterator ity = len.begin(); 
    for (; ity != len.end(); ++ity){nevents += ity -> second;}
    if (nevents == 0){return;}

    std::vector<std::vector<std::string>> thsmpl = {}; 
    thsmpl.push_back({}); 

    std::vector<size_t> thevnt = {};
    thevnt.push_back(0); 

    int idx = 0; 
    size_t avg = nevents / this -> m_settings.threads;
    if (avg == 0){avg = nevents;}
    std::string test_tree = trees_ -> at(0); 
    std::map<std::string, std::map<std::string, long>>::iterator ite = this -> reader -> tree_entries.begin(); 
    for (; ite != this -> reader -> tree_entries.end(); ++ite){
        long num_ev = ite -> second[test_tree];
        if (!num_ev){continue;}
        if (thevnt[idx] < avg){
            thevnt[idx] += num_ev;
            thsmpl[idx].push_back(ite -> first);   
            continue;
        }
        ++idx; --ite; 
        thevnt.push_back(0);
        thsmpl.push_back({}); 
    }

    std::vector<std::string> _trees    = *trees_; 
    std::vector<std::string> _branches = *branches_; 
    std::vector<std::string> _leaves   = *leaves_; 
    this -> reader -> root_end(); 
    this -> pflush(&this -> reader); 

    ROOT::EnableImplicitMT(thsmpl.size()); 
    multithreaded_t* thr = this -> make_threads(thsmpl.size(), this -> m_settings.threads); 
    for (size_t x(0); x < thsmpl.size(); ++x){
        tracing_t* tr = thr -> traces -> at(x); 
        tr -> register_thread(new std::thread(lamb, event_f, &_trees, &_branches, &_leaves, &thsmpl[x], tr), thevnt[x]); 
        while (this -> await_threads(thr, false)){} 
    }
    while (this -> await_threads(thr, true)){}
    this -> success("Finished Building Events"); 
    delete thr; 
    this -> reader = new io(); 
}
