#include <generators/analysis.h>
#include <ROOT/RDataFrame.hxx>

void analysis::build_events(){
    auto lamb = [this](
            std::vector<std::string>* trees, 
            std::vector<std::string>* branches, 
            std::vector<std::string>* leaves,
            std::vector<std::string>* files,
            std::string* title,
            size_t* num_events, 
            size_t* idx,
            event_template* evnt_comp
    ){
        if (!files -> size()){return;}
        event_template* evn = evnt_comp -> clone(); 
        ROOT::EnableImplicitMT(); 
        io* rdr = new io(); 
        rdr -> shush = true; 
        rdr -> import_settings(&this -> m_settings); 
        rdr -> trees = *trees; 
        rdr -> branches = *branches; 
        rdr -> leaves = *leaves; 
        for (size_t x(0); x < files -> size(); ++x){
            rdr -> root_files[files -> at(x)] = true;
        }
        rdr -> check_root_file_paths();

        size_t num_evn = *num_events;  
        std::map<std::string, event_template*> evnts; 
        std::map<std::string, event_template*>::iterator tx; 
        std::map<std::string, data_t*>* io_handle = rdr -> get_data(); 
        while (*idx < num_evn){
            evnts = evn -> build_event(io_handle);
            if (!evnts.size()){continue;}
            *idx += 1; 

            tx = evnts.begin(); 
            std::string fname = tx -> second -> filename; 
            std::string label = this -> file_labels[fname]; 

            meta* meta_ = nullptr; 
            bool ext = this -> meta_data.count(fname); 
            if (ext){meta_ = this -> meta_data[fname];}
            else {meta_ = rdr -> meta_data[fname];}

            if (this -> tracer -> add_meta_data(meta_, fname) && !ext){
                rdr -> meta_data[fname] = nullptr;
                this -> meta_data[fname] = meta_; 
            }

            std::vector<std::string> tmp = this -> split(fname, "/"); 
            *title = tmp[tmp.size()-1]; 
            for (; tx != evnts.end(); ++tx){
                if (!this -> tracer -> add_event(tx -> second, label)){continue;} 
                delete tx -> second; tx -> second = nullptr; 
            }
        }
        delete rdr; 
        delete evn;  
    };  
            
    event_template* event_f = nullptr; 
    std::vector<std::string>* trees_    = &this -> reader -> trees; 
    std::vector<std::string>* branches_ = &this -> reader -> branches; 
    std::vector<std::string>* leaves_   = &this -> reader -> leaves; 
    this -> reader -> import_settings(&this -> m_settings); 

    std::map<std::string, std::string>::iterator itf = this -> file_labels.begin(); 
    for (; itf != this -> file_labels.end(); ++itf){
        if (!this -> event_labels.count(itf -> second)){continue;}
        if (this -> skip_event_build[itf -> first]){continue;}
        this -> reader -> root_files[itf -> first] = true;

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

    std::vector<size_t> thevnt = {};
    std::vector<std::vector<std::string>> thsmpl = {}; 
    thsmpl.push_back({}); 
    thevnt.push_back(0); 

    int idx = 0; 
    size_t avg = nevents / this -> m_settings.threads;
    std::string test_tree = trees_ -> at(0); 
    std::map<std::string, std::map<std::string, long>>::iterator ite; 
    for (ite = this -> reader -> tree_entries.begin(); ite != this -> reader -> tree_entries.end(); ++ite){
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

    std::vector<size_t> th_prg(thsmpl.size(), 0);
    std::vector<std::thread*> thrs(thsmpl.size(), nullptr); 
    std::vector<std::string*> title(thsmpl.size(), nullptr); 
    for (size_t x(0); x < thsmpl.size(); ++x){
        title[x] = new std::string(""); 
        thrs[x] = new std::thread(lamb, trees_, branches_, leaves_, &thsmpl[x], title[x], &thevnt[x], &th_prg[x], event_f); 
    }

    std::thread* th_ = new std::thread(this -> progressbar3, &th_prg, &thevnt, &title); 
    this -> monitor(&thrs); 
    this -> reader -> root_end(); 
    delete this -> reader; 
    this -> reader = new io(); 
    th_ -> join(); 

    delete th_; 
    th_ = nullptr; 
    this -> success("Finished Building Events"); 
}
