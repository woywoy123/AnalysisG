#include <io.h>
#include <iostream>
#include <TTree.h>
#include <TFile.h>
#include <TString.h>

void io::check_root_file_paths(){
    std::map<std::string, bool> tmp = {}; 
    std::map<std::string, bool>::iterator itr = this -> root_files.begin(); 
    for (; itr != this -> root_files.end(); ++itr){
        int l = itr -> first.size(); 
        std::string last = itr -> first.substr(l - 1); 
        if (last == "*"){
            std::vector<std::string> files = this -> ls(itr -> first.substr(0, l-1), ".root"); 
            for (std::string x : files){tmp[x] = true;}
            continue; 
        }
        last = itr -> first; 
        if (!this -> is_file(last) && !this -> ends_with(&last, ".root")){continue;}
        tmp[itr -> first] = true; 
    }
    this -> root_files = tmp; 
}

std::vector<std::string> io::root_key_paths(){
    std::vector<std::string> output = {}; 

    TDirectory* dir = gDirectory; 
    for (TObject* key : *dir -> GetListOfKeys()){
        std::string pth = key -> GetName(); 
        std::cout << pth << std::endl; 
    }
    return output; 
}

void io::scan_keys(){
    std::map<std::string, bool>::iterator itr = this -> root_files.begin();
    for (; itr != this -> root_files.end(); ++itr){
        TFile* F = new TFile(itr -> first.c_str(), "READ"); 
        std::vector<std::string> data = this -> root_key_paths(); 
        delete F; 
    }
}
