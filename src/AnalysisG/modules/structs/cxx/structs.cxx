#include <element.h>
#include <report.h>
#include <folds.h>

void element_t::set_meta(){
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin();
    bool sk = itr -> second -> file_index >= (int)itr -> second -> files_i -> size(); 
    if (sk){return;}
    this -> event_index = itr -> second -> index; 
    this -> filename = itr -> second -> files_s -> at(itr -> second -> file_index);
}

bool element_t::next(){
    bool stop = true; 
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin(); 
    for (; itr != this -> handle.end(); ++itr){stop *= itr -> second -> next();}
    return stop; 
}

bool element_t::boundary(){
    long idx = -1; 
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin(); 
    for (; itr != this -> handle.end(); ++itr){idx = (*itr -> second -> files_i)[itr -> second -> file_index];}
    return idx > 0; 
}

// -------------------------- If you were directed here, simply add the data type within this section ----------------- //
// also make sure to checkout the structs/include/structs/element.h file! 

void data_t::flush_buffer(){
    // ------------ (5.) Add the buffer flush -------------------- //
    switch (this -> type){
        case data_enum::vvf: this -> flush_buffer(&this -> r_vvf); return;  
        case data_enum::vvd: this -> flush_buffer(&this -> r_vvd); return; 
        case data_enum::vvl: this -> flush_buffer(&this -> r_vvl); return; 
        case data_enum::vvi: this -> flush_buffer(&this -> r_vvi); return; 
        case data_enum::vvb: this -> flush_buffer(&this -> r_vvb); return; 

        case data_enum::vl:  this -> flush_buffer(&this -> r_vl ); return; 
        case data_enum::vd:  this -> flush_buffer(&this -> r_vd ); return; 
        case data_enum::vf:  this -> flush_buffer(&this -> r_vf ); return; 
        case data_enum::vi:  this -> flush_buffer(&this -> r_vi ); return; 
        case data_enum::vc:  this -> flush_buffer(&this -> r_vc ); return; 
        case data_enum::vb:  this -> flush_buffer(&this -> r_vb ); return; 
       
        case data_enum::ull: this -> flush_buffer(&this -> r_ull); return; 
        case data_enum::ui:  this -> flush_buffer(&this -> r_ui);  return; 
        case data_enum::d:   this -> flush_buffer(&this -> r_d  ); return; 
        case data_enum::l:   this -> flush_buffer(&this -> r_l  ); return; 
        case data_enum::f:   this -> flush_buffer(&this -> r_f  ); return; 
        case data_enum::i:   this -> flush_buffer(&this -> r_i  ); return; 
        case data_enum::b:   this -> flush_buffer(&this -> r_b  ); return; 
        case data_enum::c:   this -> flush_buffer(&this -> r_c  ); return; 
        default: return; 
    }
}

void data_t::fetch_buffer(){
    // ------------ (6.) Add the fetch buffer -------------------- //
    switch (this -> type){
        case data_enum::vvf: return this -> fetch_buffer(&this -> r_vvf);
        case data_enum::vvd: return this -> fetch_buffer(&this -> r_vvd);
        case data_enum::vvl: return this -> fetch_buffer(&this -> r_vvl);
        case data_enum::vvi: return this -> fetch_buffer(&this -> r_vvi);
        case data_enum::vvb: return this -> fetch_buffer(&this -> r_vvb);

        case data_enum::vl:  return this -> fetch_buffer(&this -> r_vl );
        case data_enum::vd:  return this -> fetch_buffer(&this -> r_vd );
        case data_enum::vf:  return this -> fetch_buffer(&this -> r_vf );
        case data_enum::vi:  return this -> fetch_buffer(&this -> r_vi );
        case data_enum::vc:  return this -> fetch_buffer(&this -> r_vc );
        case data_enum::vb:  return this -> fetch_buffer(&this -> r_vb );

        case data_enum::ull: return this -> fetch_buffer(&this -> r_ull);
        case data_enum::ui:  return this -> fetch_buffer(&this -> r_ui );
        case data_enum::l:   return this -> fetch_buffer(&this -> r_l  );
        case data_enum::d:   return this -> fetch_buffer(&this -> r_d  );
        case data_enum::f:   return this -> fetch_buffer(&this -> r_f  );
        case data_enum::i:   return this -> fetch_buffer(&this -> r_i  );
        case data_enum::b:   return this -> fetch_buffer(&this -> r_b  );
        case data_enum::c:   return this -> fetch_buffer(&this -> r_c  );
        default: return; 
    }
    // -> go to core/structs.pxd
}

void data_t::string_type(){

    // -------------------- (0). add the routing -------------- //
    if (this -> leaf_type == "vector<vector<float> >"){ this -> type = data_enum::vvf; return;}
    if (this -> leaf_type == "vector<vector<double> >"){this -> type = data_enum::vvd; return;}
    if (this -> leaf_type == "vector<vector<long> >"){  this -> type = data_enum::vvl; return;}
    if (this -> leaf_type == "vector<vector<int> >"){   this -> type = data_enum::vvi; return;}
    if (this -> leaf_type == "vector<vector<bool> >"){  this -> type = data_enum::vvb; return;}

    if (this -> leaf_type == "vector<float>"){ this -> type = data_enum::vf; return;}
    if (this -> leaf_type == "vector<long>"){  this -> type = data_enum::vl; return;}
    if (this -> leaf_type == "vector<int>"){   this -> type = data_enum::vi; return;}
    if (this -> leaf_type == "vector<char>"){  this -> type = data_enum::vc; return;}
    if (this -> leaf_type == "vector<bool>"){  this -> type = data_enum::vb; return;}
    if (this -> leaf_type == "vector<double>"){this -> type = data_enum::vd; return;}

    if (this -> leaf_type == "double"){   this -> type = data_enum::d;   return;}
    if (this -> leaf_type == "Float_t"){  this -> type = data_enum::f;   return;}
    if (this -> leaf_type == "Int_t"){    this -> type = data_enum::i;   return;}
    if (this -> leaf_type == "ULong64_t"){this -> type = data_enum::ull; return;}
    if (this -> leaf_type == "UInt_t"){   this -> type = data_enum::ui;  return;}
    if (this -> leaf_type == "Char_t"){   this -> type = data_enum::c;   return;}

    std::cout << "UNKNOWN TYPE: " << this -> leaf_type << " " << path << std::endl; 
    std::cout << "Add the type under modules/structs/cxx/structs.cxx" << std::endl;
    abort(); 
    // open -> /modules/structs/include/structs/element.h
}


// -------------- (4). add the data type interace ---------- //
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

bool data_t::element(std::vector<std::vector<bool>>* el){
    if (!this -> r_vvb){return false;} 
    (*el) = (*this -> r_vvb)[this -> index];
    return true; 
}

bool data_t::element(std::vector<long>* el){
    if (!this -> r_vl){return false;}
    (*el) = (*this -> r_vl)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<double>* el){
    if (!this -> r_vd){return false;}
    (*el) = (*this -> r_vd)[this -> index]; 
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

bool data_t::element(std::vector<bool>* el){
    if (!this -> r_vb){return false;}
    (*el) = (*this -> r_vb)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<char>* el){
    if (!this -> r_vc){return false;}
    (*el) = (*this -> r_vc)[this -> index]; 
    return true; 
}

bool data_t::element(bool* el){
    if (!this -> r_b){return false;}
    (*el) = (*this -> r_b)[this -> index];
    return true; 
}

bool data_t::element(double* el){
    if (!this -> r_d){return false;}
    (*el) = (*this -> r_d)[this -> index];
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

bool data_t::element(unsigned int* el){
    if (!this -> r_ui){return false;}
    (*el) = (*this -> r_ui)[this -> index];
    return true; 
}

bool data_t::element(char* el){
    if (!this -> r_c){return false;}
    (*el) = (*this -> r_c)[this -> index];
    return true; 
}

// ******************************************************************************************* //

void data_t::flush(){
    this -> flush_buffer();
    for (size_t x(0); x < this -> files_t -> size(); ++x){
        if (!(*this -> files_t)[x]){continue;}
        (*this -> files_t)[x] -> Close(); 
        (*this -> files_t)[x] -> Delete(); 
        delete (*this -> files_t)[x]; 
        (*this -> files_t)[x] = nullptr; 
    }
    this -> leaf = nullptr; 
    this -> branch = nullptr; 
    this -> tree = nullptr; 
    if (this -> files_s){delete this -> files_s; this -> files_s = nullptr;}
    if (this -> files_i){delete this -> files_i; this -> files_i = nullptr;}
    if (this -> files_t){delete this -> files_t; this -> files_t = nullptr;}
}

void data_t::initialize(){
    TFile* c = (*this -> files_t)[this -> file_index]; 
    c = (c -> Open(c -> GetTitle())); 
    this -> tree        = (TTree*)c -> Get(this -> tree_name.c_str()); 
    this -> tree -> SetCacheSize(10000000U); 
    this -> tree -> AddBranchToCache("*", true);
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
    c -> Delete(); 
    delete c; 
    (*this -> files_t)[this -> file_index] = nullptr; 
} 

bool data_t::next(){
    if (this -> file_index >= (int)this -> files_i -> size()){return true;} 
    long idx = (*this -> files_i)[this -> file_index];
    this -> fname = &(*this -> files_s)[this -> file_index];
    if (this -> index+1 < idx){this -> index++; return false;}

    this -> file_index++; 
    if (this -> file_index >= (int)this -> files_i -> size()){return true;}
    this -> initialize();
    return false; 
}


std::string model_report::print(){
    std::string msg = "Run Name: " + this -> run_name; 
    msg += " Epoch: " + std::to_string(this -> epoch); 
    msg += " K-Fold: " + std::to_string(this -> k+1); 
    msg += "\n"; 
    msg += "__________ LOSS FEATURES ___________ \n"; 
    msg += this -> prx(&this -> loss_graph, "Graph Loss");
    msg += this -> prx(&this -> loss_node, "Node Loss"); 
    msg += this -> prx(&this -> loss_edge, "Edge Loss"); 

    msg += "__________ ACCURACY FEATURES ___________ \n"; 
    msg += this -> prx(&this -> accuracy_graph, "Graph Accuracy");
    msg += this -> prx(&this -> accuracy_node, "Node Accuracy"); 
    msg += this -> prx(&this -> accuracy_edge, "Edge Accuracy"); 
    return msg; 
}

std::string model_report::prx(std::map<mode_enum, std::map<std::string, float>>* data, std::string title){
    bool trig = false; 
    std::string out = ""; 
    std::map<std::string, float>::iterator itf; 
    std::map<mode_enum, std::map<std::string, float>>::iterator itx; 
    for (itx = data -> begin(); itx != data -> end(); ++itx){
        if (!itx -> second.size()){return "";}
        if (!trig){out += title + ": \n"; trig = true;}
        switch (itx -> first){
            case mode_enum::training:   out += "Training -> "; break;
            case mode_enum::validation: out += "Validation -> "; break;
            case mode_enum::evaluation: out += "Evaluation -> "; break; 
        }
        for (itf = itx -> second.begin(); itf != itx -> second.end(); ++itf){
            out += itf -> first + ": " + std::to_string(itf -> second) + " | "; 
        }
        out += "\n"; 
    }
    return out; 
}

void write_t::write(){
    this -> tree -> Fill(); 
    std::map<std::string, variable_t>::iterator itx = this -> data.begin(); 
    for (; itx != this -> data.end(); ++itx){itx -> second.flush();}
}

void write_t::create(std::string tr_name, std::string path){
    if (this -> file){return;}
    this -> file = new TFile(path.c_str(), "RECREATE"); 
    this -> tree = new TTree(tr_name.c_str(), "data"); 
}

void write_t::close(){
    this -> tree -> ResetBranchAddresses(); 
    this -> tree -> Write("", TObject::kOverwrite); 
    this -> file -> Close();
    this -> file -> Delete(); 
}

void graph_hdf5_w::flush_data(){
    free(this -> hash); 
    free(this -> filename); 
    free(this -> edge_index); 

    free(this -> data_map_graph); 
    free(this -> data_map_node); 
    free(this -> data_map_edge); 

    free(this -> truth_map_graph); 
    free(this -> truth_map_node);
    free(this -> truth_map_edge); 

    free(this -> data_graph); 
    free(this -> data_node); 
    free(this -> data_edge); 

    free(this -> truth_graph); 
    free(this -> truth_node); 
    free(this -> truth_edge); 

    this -> hash = nullptr; 
    this -> filename = nullptr; 
    this -> edge_index = nullptr; 

    this -> data_map_graph = nullptr; 
    this -> data_map_node = nullptr; 
    this -> data_map_edge = nullptr; 

    this -> truth_map_graph = nullptr; 
    this -> truth_map_node = nullptr;
    this -> truth_map_edge = nullptr; 

    this -> data_graph = nullptr; 
    this -> data_node = nullptr; 
    this -> data_edge = nullptr; 

    this -> truth_graph = nullptr; 
    this -> truth_node = nullptr; 
    this -> truth_edge = nullptr; 
}



