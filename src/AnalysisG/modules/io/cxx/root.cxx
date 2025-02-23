#include "io.h"
#include <TH1.h>

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
            for (size_t x(0); x < f.size(); ++x){
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
    for (size_t x(0); x < this -> trees.size(); ++x){
        std::string name = this -> trees[x];
        if (std::string(tr -> GetName()) != name){continue;}
        this -> tree_data[file_name][name] = tr; 
        this -> tree_entries[file_name][std::string(tr -> GetName())] = tr -> GetEntries(); 
        found = true; 
        break; 
    }
    if (!found){return;}

    for (size_t x(0); x < this -> branches.size(); ++x){
        std::string name = this -> branches[x]; 
        TBranch* br = tr -> FindBranch(name.c_str()); 
        if (!br){continue;}
        name = std::string(tr -> GetName()) + "." + std::string(br -> GetName()); 
        this -> branch_data[file_name][name] = br; 
    }

    for (size_t x(0); x < this -> leaves.size(); ++x){
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

    if (this -> ends_with(&this -> metacache_path, ".h5")){}
    else if (!this -> ends_with(&this -> metacache_path, "/")){this -> metacache_path += "/meta.h5";}
    else {this -> metacache_path += "meta.h5";}
    if (!this -> is_file(this -> metacache_path)){
        std::vector<std::string> tmpo = this -> split(this -> metacache_path, "/"); 
        std::string meta_s = tmpo[tmpo.size()-1]; 
        this -> replace(&this -> metacache_path, meta_s, ""); 
        this -> create_path(this -> metacache_path); 
        this -> metacache_path += meta_s; 
    }

    for (size_t x(0); x < tmp.size(); ++x){
        std::string updated = path + tmp[x]; 
        TObject* obj = gDirectory -> Get(updated.c_str()); 
        if (!obj){continue;}

        std::string fname = this -> file_root -> GetTitle(); 
        std::string obname = std::string(obj -> GetName()); 

        // ------------ meta data scraping ---------------------- //
        if (!this -> meta_data.count(fname)){this -> meta_data[fname] = new meta();}
        meta* mtx = this -> meta_data[fname];  
        mtx -> metacache_path = this -> metacache_path; 
        mtx -> meta_data.sample_name = fname; 
        if (obname == "AnalysisTracking"){mtx -> scan_data(obj); continue;}
        if (this -> sow_name == obname){mtx -> scan_sow(obj); continue;}
        else if (this -> has_string(&this -> sow_name, "*")){
            std::vector<std::string> spl = this -> split(this -> sow_name, "*");
            bool found = spl.size(); 
            for (std::string& v : spl){found *= this -> has_string(&obname, v);}
            if (found){mtx -> scan_data(obj); continue;}
        }
        // ------------ meta data scraping ---------------------- //

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
            if (this -> file_root -> IsZombie()){
                delete this -> file_root; 
                this -> file_root = nullptr; 
                continue;
            }
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
        for (size_t x(0); x < this -> trees.size(); ++x){
            if (tr -> count(this -> trees[x])){continue;}
            bool has = this -> has_value(&this -> keys[fname]["missed"]["Trees"], this -> trees[x]); 
            if (has){continue;}
            this -> keys[fname]["missed"]["Trees"].push_back(this -> trees[x]); 
        }
    
        std::map<std::string, TBranch*>* br = &this -> branch_data[fname]; 
        for (size_t x(0); x < this -> branches.size(); ++x){
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
        for (size_t x(0); x < this -> leaves.size(); ++x){
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
           long nev_ = 0; 
           std::string msg = ""; 
           std::map<std::string, long> tree_ent = this -> tree_entries[fname]; 
           std::map<std::string, long>::iterator itl = tree_ent.begin();
           for (; itl != tree_ent.end(); ++itl){msg += itl -> first + " - " + itl -> second + " | "; nev_ +=itl -> second;}
           if (!nev_){this -> warning("(-)> " + fname + " (" + msg.substr(0, msg.size()-3) + ") Skipping..."); }
           else {this -> success("(+)> " + fname + " (" + msg.substr(0, msg.size()-3) + ") OK! ");}
           this -> success_trigger[fname] = true; 
           continue;
       }
       for (size_t x(0); x < trees_m.size(); ++x){
           if (this -> missing_trigger[trees_m[x]]){continue;}
           this -> missing_trigger[trees_m[x]] = true; 
           this -> warning("Missing Tree: " + trees_m[x]); 
       }

       for (size_t x(0); x < branches_m.size(); ++x){
           if (this -> missing_trigger[branches_m[x]]){continue;}
           this -> missing_trigger[branches_m[x]] = true; 
           this -> warning("Missing Branch: " + branches_m[x]); 
       }

       for (size_t x(0); x < leaves_m.size(); ++x){
           if (this -> missing_trigger[leaves_m[x]]){continue;}
           this -> missing_trigger[leaves_m[x]] = true; 
           this -> warning("Missing Leaves: " + leaves_m[x]); 
       }
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
            std::vector<std::string> pth_ = this -> split(lf_name, "."); 
            if (!this -> tree_entries[fname][pth_[0]]){continue;}
            if (!this -> iters -> count(lf_name)){
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
    for (; it != this -> iters -> end(); ++it){it -> second -> flush(); delete it -> second;}
    this -> iters -> clear(); 
    delete this -> iters; this -> iters = nullptr; 
}


