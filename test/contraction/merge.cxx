#include <vector>
#include <TTree.h>
#include <TFile.h>
#include <TBranch.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <iostream>
#include <string>
#include <TInterpreter.h>

void add_to_dict(std::vector<std::vector<int>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<int>>", "vector");}
void add_to_dict(std::vector<std::vector<float>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<float>>", "vector");}
void add_to_dict(std::vector<std::vector<double>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<double>>", "vector");}
void add_to_dict(std::vector<float>* dummy){gInterpreter -> GenerateDictionary("vector<float>", "vector");}
void add_to_dict(std::vector<double>* dummy){gInterpreter -> GenerateDictionary("vector<double>", "vector");}

template <typename T>
void fetch_buffer(std::vector<T>** data, TFile* tf, std::string br){
    TTree* tr    = (TTree*)tf -> Get("nominal"); 
    TTreeReader r = TTreeReader(tr); 
    TTreeReaderValue<T> dr(r, br.c_str()); 
    (*data) = new std::vector<T>();  
    while (r.Next()){(*data) -> push_back(*dr);}
}

void fill_buffer(std::vector<std::vector<int>>* out, std::vector<std::vector<std::vector<int>>>* inpt, int index){
    for (size_t i(0); i < (*inpt)[index].size(); ++i){out -> push_back((*inpt)[index][i]);}
}

void fill_buffer(std::vector<std::vector<float>>* out, std::vector<std::vector<std::vector<float>>>* inpt, int index){
    for (size_t i(0); i < (*inpt)[index].size(); ++i){out -> push_back((*inpt)[index][i]);}
}

void fill_buffer(std::vector<std::vector<double>>* out, std::vector<std::vector<std::vector<double>>>* inpt, int index){
    for (size_t i(0); i < (*inpt)[index].size(); ++i){out -> push_back((*inpt)[index][i]);}
}

void fill_buffer(std::vector<double>* out, std::vector<std::vector<double>>* inpt, int index){
    for (size_t i(0); i < (*inpt)[index].size(); ++i){out -> push_back((*inpt)[index][i]);}
}


int main(){
    gErrorIgnoreLevel = 6001; 

    std::vector<std::vector<std::vector<int>>>* edge_index = nullptr; 
    std::vector<std::vector<std::vector<float>>>* edge_top = nullptr; 
    std::vector<std::vector<std::vector<double>>>* top_kine = nullptr; 

    std::vector<std::vector<std::vector<float>>>* graph_ntops = nullptr; 
    std::vector<std::vector<std::vector<float>>>* graph_signal = nullptr;

    std::vector<std::vector<double>>* res_mass = nullptr;  
    std::vector<std::vector<double>>* top_mass = nullptr; 
    
    TFile* tf2 = new TFile("gnn.root", "READ"); 
    fetch_buffer(&edge_index  , tf2, "edge_index"); 
    fetch_buffer(&edge_top    , tf2, "extra_top_edge_score"); 
    fetch_buffer(&top_kine    , tf2, "extra_top_kinematic"); 
    fetch_buffer(&graph_ntops , tf2, "extra_ntops_score"); 
    fetch_buffer(&graph_signal, tf2, "extra_is_res_score"); 
    fetch_buffer(&res_mass    , tf2, "extra_res_mass"); 
    fetch_buffer(&top_mass    , tf2, "extra_top_mass"); 
    tf2 -> Close();
    delete tf2; 

    std::cout << edge_index -> size() << std::endl; 
    std::cout << edge_top -> size() << std::endl; 
    std::cout << top_kine -> size() << std::endl; 
    std::cout << graph_ntops -> size() << std::endl; 
    std::cout << graph_signal -> size() << std::endl;
    std::cout << res_mass -> size() << std::endl;  
    std::cout << top_mass -> size() << std::endl; 
 
    std::vector<std::vector<int>> edge_index_out = {}; 
    std::vector<std::vector<float>> edge_top_out = {}; 
    std::vector<std::vector<double>> top_kine_out = {}; 
    std::vector<std::vector<float>> graph_signal_out = {}; 
    std::vector<std::vector<float>> graph_ntops_out = {}; 

    std::vector<double> res_mass_out; 
    std::vector<double> top_mass_out; 

    add_to_dict(&edge_index_out);
    add_to_dict(&edge_top_out); 
    add_to_dict(&graph_signal_out); 
    add_to_dict(&graph_ntops_out); 
    add_to_dict(&res_mass_out); 
    add_to_dict(&top_mass_out); 

    TFile* tf1 = new TFile("dst.root", "UPDATE"); 
    TTree* tr1 = (TTree*)tf1 -> Get("reco;1");

    tr1 -> Branch("edge_index"     , &edge_index_out); 
    tr1 -> Branch("edge_top"       , &edge_top_out); 
    tr1 -> Branch("top_pxpypze"    , &top_kine_out); 
    tr1 -> Branch("graph_signal"   , &graph_signal_out); 
    tr1 -> Branch("graph_ntops"    , &graph_ntops_out);
    tr1 -> Branch("resonance_mass" , &res_mass_out); 
    tr1 -> Branch("top_mass"       , &top_mass_out); 

    size_t lx = edge_index -> size(); 
    for (size_t x(0); x < edge_index -> size(); ++x){
        fill_buffer(&edge_index_out  , edge_index  , x); 
        fill_buffer(&edge_top_out    , edge_top    , x); 
        fill_buffer(&top_kine_out    , top_kine    , x); 
        fill_buffer(&graph_signal_out, graph_signal, x); 
        fill_buffer(&graph_ntops_out , graph_ntops , x);
        fill_buffer(&res_mass_out    , res_mass    , x);
        fill_buffer(&top_mass_out    , top_mass    , x); 
        tr1 -> Fill();

        edge_index_out.clear(); 
        edge_top_out.clear();    
        top_kine_out.clear();    
        graph_signal_out.clear();
        graph_ntops_out.clear(); 
        res_mass_out.clear();    
        top_mass_out.clear();    
        if (x % lx){continue;}
        std::cout << x / lx << std::endl;
    }
    tr1 -> ResetBranchAddresses(); 
    tr1 -> Write(0, TObject::kOverwrite); 
    tf1 -> Close(); 
    delete tf1; 

    delete edge_index; 
    delete edge_top; 
    delete top_kine; 
    delete graph_ntops; 
    delete graph_signal;
    delete res_mass;  
    delete top_mass; 
 
    return 0;
}
