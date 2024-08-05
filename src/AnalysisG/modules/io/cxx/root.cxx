#include <io.h>
#include <iostream>

bool element_t::next(){
    bool stop = false; 
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin();
    for (; itr != this -> handle.end(); ++itr){
        switch (itr -> second -> type){
            case data_enum::vvf: stop = itr -> second -> next(&this -> r_vvf[itr -> first]); break; 
            case data_enum::vvd: stop = itr -> second -> next(&this -> r_vvd[itr -> first]); break; 
            case data_enum::vvl: stop = itr -> second -> next(&this -> r_vvl[itr -> first]); break; 
            case data_enum::vvi: stop = itr -> second -> next(&this -> r_vvi[itr -> first]); break; 

            case data_enum::vf:  stop = itr -> second -> next(&this -> r_vf[itr -> first]);  break; 
            case data_enum::vl:  stop = itr -> second -> next(&this -> r_vl[itr -> first]);  break; 
            case data_enum::vi:  stop = itr -> second -> next(&this -> r_vi[itr -> first]);  break; 
            case data_enum::vc:  stop = itr -> second -> next(&this -> r_vc[itr -> first]);  break; 

            case data_enum::f:   stop = itr -> second -> next(&this -> r_f[itr -> first]);   break; 
            case data_enum::l:   stop = itr -> second -> next(&this -> r_l[itr -> first]);   break; 
            case data_enum::i:   stop = itr -> second -> next(&this -> r_i[itr -> first]);   break; 
            case data_enum::ull: stop = itr -> second -> next(&this -> r_ull[itr -> first]); break; 
            default: break; 
        }
    }
    return stop; 
}

void element_t::set_meta(){
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin();
    bool sk = itr -> second -> file_index >= (int)itr -> second -> files_i -> size(); 
    if (sk){return;}
    this -> event_index = itr -> second -> index; 
    this -> filename = itr -> second -> files_s -> at(itr -> second -> file_index);
}

void data_t::flush_buffer(){
    switch (this -> type){
        case data_enum::vvf: this -> flush_buffer(&this -> r_vvf); break; 
        case data_enum::vvd: this -> flush_buffer(&this -> r_vvd); break; 
        case data_enum::vvl: this -> flush_buffer(&this -> r_vvl); break; 
        case data_enum::vvi: this -> flush_buffer(&this -> r_vvi); break; 

        case data_enum::vf:  this -> flush_buffer(&this -> r_vf ); break; 
        case data_enum::vl:  this -> flush_buffer(&this -> r_vl ); break; 
        case data_enum::vi:  this -> flush_buffer(&this -> r_vi ); break; 
        case data_enum::vc:  this -> flush_buffer(&this -> r_vc ); break; 

        case data_enum::f:   this -> flush_buffer(&this -> r_f  ); break; 
        case data_enum::l:   this -> flush_buffer(&this -> r_l  ); break; 
        case data_enum::i:   this -> flush_buffer(&this -> r_i  ); break; 
        case data_enum::ull: this -> flush_buffer(&this -> r_ull); break; 
        default: return; 
    }
}

void data_t::fetch_buffer(){
    switch (this -> type){
        case data_enum::vvf: return this -> fetch_buffer(&this -> r_vvf);
        case data_enum::vvd: return this -> fetch_buffer(&this -> r_vvd);
        case data_enum::vvl: return this -> fetch_buffer(&this -> r_vvl);
        case data_enum::vvi: return this -> fetch_buffer(&this -> r_vvi);

        case data_enum::vf:  return this -> fetch_buffer(&this -> r_vf );
        case data_enum::vl:  return this -> fetch_buffer(&this -> r_vl );
        case data_enum::vi:  return this -> fetch_buffer(&this -> r_vi );
        case data_enum::vc:  return this -> fetch_buffer(&this -> r_vc );

        case data_enum::f:   return this -> fetch_buffer(&this -> r_f  );
        case data_enum::l:   return this -> fetch_buffer(&this -> r_l  );
        case data_enum::i:   return this -> fetch_buffer(&this -> r_i  );
        case data_enum::ull: return this -> fetch_buffer(&this -> r_ull);
        default: return; 
    }
}

void data_t::flush(){
    this -> flush_buffer();
    if (this -> files_s){delete this -> files_s;}
    if (this -> files_i){delete this -> files_i;}
    if (this -> files_t){delete this -> files_t;}
}

void data_t::initialize(){
    TFile* c = this -> files_t -> at(this -> file_index); 
    c = c -> Open(c -> GetTitle()); 

    this -> tree        = (TTree*)c -> Get(this -> tree_name.c_str()); 
    this -> leaf        = this -> tree -> FindLeaf(this -> leaf_name.c_str());
    this -> branch      = this -> leaf -> GetBranch();  
    
    this -> tree_name   = this -> tree -> GetName();
    this -> leaf_name   = this -> leaf -> GetName();
    this -> branch_name = this -> branch -> GetName(); 

    this -> string_type(); 
    this -> flush_buffer(); 
    this -> fetch_buffer(); 
    this -> index = 0; 
    c -> Close(); 
    delete c; 
}; 

void data_t::string_type(){
    if (this -> leaf_type == "Float_t"){
        this -> type = data_enum::f; return; 
    }

    if (this -> leaf_type == "Int_t"){
        this -> type = data_enum::i; return; 
    }

    if (this -> leaf_type == "ULong64_t"){
        this -> type = data_enum::ull; return; 
    }

    if (this -> leaf_type == "vector<vector<int> >"){
        this -> type = data_enum::vvi; return; 
    }

    if (this -> leaf_type == "vector<vector<float> >"){
        this -> type = data_enum::vvf; return; 
    }

    if (this -> leaf_type == "vector<vector<double> >"){
        this -> type = data_enum::vvd; return; 
    }

    if (this -> leaf_type == "vector<float>"){
        this -> type = data_enum::vf; return; 
    }

    if (this -> leaf_type == "vector<int>"){
        this -> type = data_enum::vi; return; 
    }

    if (this -> leaf_type == "vector<char>"){
        this -> type = data_enum::vc; return; 
    }

    std::cout << "UNKNOWN TYPE: " << this -> leaf_type << " " << path << std::endl; 
    std::cout << "Add the type under modules/io/cxx/root.cxx" << std::endl;
    abort(); 
}

bool data_t::element(std::vector<std::vector<float>>* el){
    if (!this -> r_vvf){return false;}
    (*el) = (*this -> r_vvf)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<std::vector<double>>* el){
    if (!this -> r_vvd){return false;}
    (*el) = (*this -> r_vvd)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<std::vector<long>>* el){
    if (!this -> r_vvl){return false;}
    (*el) = (*this -> r_vvl)[this -> index];
    return true; 
}

bool data_t::element(std::vector<std::vector<int>>* el){
    if (!this -> r_vvi){return false;} 
    (*el) = (*this -> r_vvi)[this -> index];
    return true; 
}

bool data_t::element(std::vector<float>* el){
    if (!this -> r_vf){return false;}
    (*el) = (*this -> r_vf)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<int>* el){
    if (!this -> r_vi){return false;}
    (*el) = (*this -> r_vi)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<long>* el){
    if (!this -> r_vl){return false;}
    (*el) = (*this -> r_vl)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<char>* el){
    if (!this -> r_vc){return false;}
    (*el) = (*this -> r_vc)[this -> index]; 
    return true; 
}

bool data_t::element(float* el){
    if (!this -> r_f){return false;}
    (*el) = (*this -> r_f)[this -> index];
    return true; 
}

bool data_t::element(int* el){
    if (!this -> r_i){return false;}
    (*el) = (*this -> r_i)[this -> index];
    return true; 
}

bool data_t::element(long* el){
    if (!this -> r_l){return false;}
    (*el) = (*this -> r_l)[this -> index];
    return true; 
}

bool data_t::element(unsigned long long* el){
    if (!this -> r_ull){return false;}
    (*el) = (*this -> r_ull)[this -> index];
    return true; 
}


void io::check_root_file_paths(){
    std::map<std::string, bool> tmp = {}; 
    std::map<std::string, bool>::iterator itr = this -> root_files.begin(); 

    this -> success("Checking File Path:"); 
    for (; itr != this -> root_files.end(); ++itr){
        int l = itr -> first.size(); 
        std::string last = itr -> first.substr(l - 1); 
        if (last == "*"){
            std::vector<std::string> files = this -> ls(itr -> first.substr(0, l-1), ".root"); 
            for (std::string x : files){tmp[x] = true;}
            continue; 
        }
        last = itr -> first; 
        if (!this -> ends_with(&last, ".root")){
            std::vector<std::string> f = this -> ls(last, ".root"); 
            for (int x(0); x < f.size(); ++x){
                this -> success(f[x]);
                tmp[f[x]] = true; 
            }
            continue;
        }
        this -> success(itr -> first); 
        tmp[this -> absolute_path(itr -> first)] = true; 
    }
    this -> root_files = tmp; 
}

void io::root_key_paths(std::string path, TTree* t){
    if (!this -> file_root){return;}
    TTree* tr = this -> file_root -> Get<TTree>(path.c_str()); 
    if (!tr){return;}
    bool found = false; 
    std::string file_name = this -> file_root -> GetTitle(); 
    for (unsigned int x(0); x < this -> trees.size(); ++x){
        std::string name = this -> trees[x];
        if (std::string(tr -> GetName()) != name){continue;}
        this -> tree_data[file_name][name] = tr; 
        this -> tree_entries[file_name][std::string(tr -> GetName())] = tr -> GetEntries(); 
        found = true; 
        break; 
    }
    if (!found){return;}

    for (unsigned int x(0); x < this -> branches.size(); ++x){
        std::string name = this -> branches[x]; 
        TBranch* br = tr -> FindBranch(name.c_str()); 
        if (!br){continue;}
        name = std::string(tr -> GetName()) + "." + std::string(br -> GetName()); 
        this -> branch_data[file_name][name] = br; 
    }

    for (unsigned int x(0); x < this -> leaves.size(); ++x){
        std::string name = this -> leaves[x]; 
        TLeaf* lf = tr -> FindLeaf(name.c_str());
        if (!lf){continue;}
        TBranch* br = lf -> GetBranch();

        std::string name_s = std::string(tr -> GetName()) + "."; 
        if (!br){
            this -> leaf_data[file_name][name_s] = lf; 
            this -> leaf_typed[file_name][name_s] = std::string(lf -> GetTypeName()); 
            continue;
        }

        name_s += std::string(br -> GetName()) + "." + std::string(lf -> GetName());
        if (!br -> IsFolder()){
            this -> leaf_data[file_name][name_s]  = lf; 
            this -> leaf_typed[file_name][name_s] = lf -> GetTypeName(); 
            continue;
        }

        for (TObject* obj : *br -> GetListOfLeaves()){
            TLeaf* lf = (TLeaf*)obj; 
            if (std::string(lf -> GetName()) != name){continue;}
            std::string lx = name_s + "." + lf -> GetName(); 
            this -> leaf_typed[file_name][lx] = std::string(lf -> GetTypeName()); 
            this -> leaf_data[file_name][lx] = lf;
        }
    }
}

void io::root_key_paths(std::string path){
    TDirectory* dir = gDirectory; 
    std::vector<std::string> tmp = {}; 
    for (TObject* key : *dir -> GetListOfKeys()){tmp.push_back(key -> GetName());}
    for (unsigned int x(0); x < tmp.size(); ++x){
        std::string updated = path + tmp[x]; 
        TObject* obj = gDirectory -> Get(updated.c_str()); 
        if (!obj){continue;}

        std::string fname = this -> file_root -> GetTitle(); 
        if (std::string(obj -> GetName()) == "AnalysisTracking"){
            if (this -> meta_data.count(fname)){continue;}
            TTreeReader r = TTreeReader("AnalysisTracking"); 
            TTreeReaderValue<std::string> dr(r, "jsonData"); 
            std::string data = ""; 
            while (r.Next()){data += (*dr);}
            this -> meta_data[fname] = new meta(); 
            this -> meta_data[fname] -> parse_json(data); 
            continue; 
        }

        if (obj -> InheritsFrom("TTree")){
            this -> root_key_paths(updated, (TTree*)obj);
            continue;
        }
        if (obj -> InheritsFrom("TH1")){continue;}

        dir -> cd(updated.c_str()); 
        this -> root_key_paths(updated + "/"); 
        dir -> cd(path.c_str()); 
    }
}

bool io::scan_keys(){
    std::map<std::string, bool>::iterator itr = this -> root_files.begin();
    for (; itr != this -> root_files.end(); ++itr){
        if (!this -> files_open.count(itr -> first)){
            this -> file_root = new TFile(itr -> first.c_str(), "READ");
            this -> file_root -> SetTitle(itr -> first.c_str()); 
            this -> files_open[itr -> first] = this -> file_root; 
        }
        else {this -> file_root = this -> files_open[itr -> first];}
        if (!this -> file_root -> IsOpen()){this -> file_root -> ReOpen("READ");}
        this -> root_key_paths(""); 
        this -> file_root -> Close(); 
        this -> file_root = nullptr; 
    }

    std::map<std::string, TFile*>::iterator tf = this -> files_open.begin(); 
    for (; tf != this -> files_open.end(); ++tf){
        std::string fname = tf -> first; 
        if (!this -> tree_data.count(fname)){
            this -> keys[fname]["missed"]["Trees"]    = this -> trees; 
            this -> keys[fname]["missed"]["Branches"] = this -> branches; 
            this -> keys[fname]["missed"]["Leaves"]   = this -> leaves; 
            continue;
        }

        std::map<std::string, TTree*>* tr = &this -> tree_data[fname]; 
        for (int x(0); x < this -> trees.size(); ++x){
            if (tr -> count(this -> trees[x])){continue;}
            bool has = this -> has_value(&this -> keys[fname]["missed"]["Trees"], this -> trees[x]); 
            if (has){continue;}
            this -> keys[fname]["missed"]["Trees"].push_back(this -> trees[x]); 
        }
    
        std::map<std::string, TBranch*>* br = &this -> branch_data[fname]; 
        for (int x(0); x < this -> branches.size(); ++x){
            bool found = false; 

            std::map<std::string, TBranch*>::iterator itb = br -> begin();
            for (; itb != br -> end(); ++itb){
                std::string br_name = itb -> first; 
                if (!this -> ends_with(&br_name, this -> branches[x])){continue;}
                found = true;
                break;
            }
            if (found){continue;}
            bool has = this -> has_value(&this -> keys[fname]["missed"]["Branches"], this -> branches[x]); 
            if (has){continue;}
            this -> keys[fname]["missed"]["Branches"].push_back(this -> branches[x]); 
        }

        std::map<std::string, TLeaf*>* lf = &this -> leaf_data[fname]; 
        for (int x(0); x < this -> leaves.size(); ++x){
            bool found = false; 

            std::map<std::string, TLeaf*>::iterator itf = lf -> begin();
            for (; itf != lf -> end(); ++itf){
                std::string lf_name = itf -> first; 
                if (!this -> ends_with(&lf_name, this -> leaves[x])){continue;}
                found = true;
                break;
            }
            if (found){continue;}
            bool has = this -> has_value(&this -> keys[fname]["missed"]["Leaves"], this -> leaves[x]); 
            if (has){continue;}
            this -> keys[fname]["missed"]["Leaves"].push_back(this -> leaves[x]); 
        }

       std::vector<std::string> leaves_m   = this -> keys[fname]["missed"]["Leaves"];
       std::vector<std::string> branches_m = this -> keys[fname]["missed"]["Branches"];
       std::vector<std::string> trees_m    = this -> keys[fname]["missed"]["Trees"];
       if (leaves_m.size() || branches_m.size() || trees_m.size()){this -> failure("-> " + fname);}
       else if (this -> success_trigger[fname]){continue;}
       else {
           this -> success("-> " + fname + " OK!"); 
           this -> success_trigger[fname] = true; 
           continue;
       }
       for (int x(0); x < trees_m.size(); ++x){
           if (this -> missing_trigger[trees_m[x]]){continue;}
           this -> missing_trigger[trees_m[x]] = true; 
           this -> warning("Missing Tree: " + trees_m[x]); 
       }

       for (int x(0); x < branches_m.size(); ++x){
           if (this -> missing_trigger[branches_m[x]]){continue;}
           this -> missing_trigger[branches_m[x]] = true; 
           this -> warning("Missing Branch: " + branches_m[x]); 
       }

       for (int x(0); x < leaves_m.size(); ++x){
           if (this -> missing_trigger[leaves_m[x]]){continue;}
           this -> missing_trigger[leaves_m[x]] = true; 
           this -> warning("Missing Leaves: " + leaves_m[x]); 
       }
       this -> failure("aborting due to missing data!"); 
       return false; 
    }
    return true; 
}

std::map<std::string, long> io::root_size(){
    this -> scan_keys(); 
    std::map<std::string, long> output = {}; 
    std::map<std::string, std::map<std::string, long>>::iterator itr; 
    for (itr = this -> tree_entries.begin(); itr != this -> tree_entries.end(); ++itr){
        std::map<std::string, long>::iterator itx = itr -> second.begin(); 
        for (; itx != itr -> second.end(); ++itx){output[itx -> first] += itx -> second;}
    }
    return output; 
}

void io::root_begin(){
    this -> scan_keys();
    if (this -> iters){this -> root_end();}
    this -> iters = new std::map<std::string, data_t*>(); 

    std::map<std::string, TLeaf*>::iterator lfii; 
    std::map<std::string, std::map<std::string, TLeaf*>>::iterator lfi; 

    for (lfi = this -> leaf_data.begin(); lfi != this -> leaf_data.end(); ++lfi){
        std::string fname = lfi -> first; 
        std::map<std::string, TLeaf*>* lf_map = &lfi -> second; 

        for (lfii = lf_map -> begin(); lfii != lf_map -> end(); ++lfii){
            std::string lf_name = lfii -> first; 
            TLeaf* leaf_pnt = lfii -> second; 
            if (!this -> iters -> count(lf_name)){
                std::vector<std::string> pth_ = this -> split(lf_name, "."); 
                data_t* dt      = new data_t(); 
                dt -> path      = lf_name; 
                dt -> tree_name = pth_[0];
                dt -> leaf_name = pth_[pth_.size()-1]; 
                dt -> leaf_type = this -> leaf_typed[fname][lf_name]; 
                dt -> files_s   = new std::vector<std::string>();
                dt -> files_i   = new std::vector<long>(); 
                dt -> files_t   = new std::vector<TFile*>();  
                (*this -> iters)[lf_name] = dt; 
            }

            data_t* v = (*this -> iters)[lf_name]; 
            v -> files_s -> push_back(fname); 
            v -> files_t -> push_back(this -> files_open[fname]); 
            v -> files_i -> push_back(this -> tree_entries[fname][v -> tree_name]); 
            if (!v -> leaf){v -> initialize();}
        }
    }
}

std::map<std::string, data_t*>* io::get_data(){
    if (!this -> iters){this -> root_begin();}
    return this -> iters;  
}

void io::root_end(){
    if (!this -> iters){return;}
    std::map<std::string, data_t*>::iterator it = this -> iters -> begin(); 
    for (; it != this -> iters -> end(); ++it){
        data_t* v = it -> second;
        v -> flush();
        delete v; 
    }
    this -> iters -> clear(); 
    delete this -> iters; 
    this -> iters = nullptr; 
}


