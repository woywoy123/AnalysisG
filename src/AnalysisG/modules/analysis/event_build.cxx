#include <generators/analysis.h>

void analysis::build_events(){
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
    if (!this -> reader -> root_files.size()){return this -> info("Skipping event building.");}
    if (!event_f){return this -> warning("Missing Event Implementation for specified samples!");}
    this -> success("+============================+"); 
    this -> success("|   Starting Event Builder   |");
    this -> success("+============================+"); 
    this -> reader -> check_root_file_paths(); 

    long index = 0; 
    size_t nevents = 0;
    std::map<std::string, long> len = this -> reader -> root_size(); 
    std::map<std::string, long>::iterator ity = len.begin(); 
    for (; ity != len.end(); ++ity){nevents += ity -> second;}
    if (nevents == 0){return;}
    std::map<std::string, std::map<std::string, long>> root_entries = this -> reader -> tree_entries; 

    std::string title = ""; 
    std::vector<size_t> th_prg(1, 0); 
    std::thread* th_ = new std::thread(this -> progressbar2, &th_prg, &nevents, &title); 
    std::map<std::string, data_t*>* io_handle = this -> reader -> get_data(); 

    while (index < nevents){
        std::map<std::string, event_template*> evnts = event_f -> build_event(io_handle);
        if (!evnts.size()){continue;}
        th_prg[0]+=1; 
        ++index; 

        std::map<std::string, event_template*>::iterator tx = evnts.begin(); 
        std::string fname = tx -> second -> filename; 
        std::string label = this -> file_labels[fname]; 

        meta* meta_ = nullptr; 
        bool ext = this -> meta_data.count(fname); 
        if (ext){meta_ = this -> meta_data[fname];}
        else {meta_ = this -> reader -> meta_data[fname];}

        bool detach = this -> tracer -> add_meta_data(meta_, fname); 
        if (detach && !ext){
            this -> reader -> meta_data[fname] = nullptr;
            this -> meta_data[fname] = meta_; 
        }

        std::vector<std::string> tmp = this -> split(fname, "/"); 
        title = tmp[tmp.size()-1]; 
        for (; tx != evnts.end(); ++tx){
            if (!this -> tracer -> add_event(tx -> second, label)){continue;} 
            delete tx -> second; 
            tx -> second = nullptr; 
        }
    }
    th_ -> join(); 
    delete th_; 
    th_ = nullptr; 

    this -> reader -> root_end(); 
    delete this -> reader; 
    this -> reader = new io(); 
    std::cout << std::endl;

    this -> success("Finished Building Events"); 
}
