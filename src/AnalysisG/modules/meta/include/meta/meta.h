#ifndef META_META_H
#define META_META_H

#include <tools/tools.h>
#include <structs/meta.h>
#include <rapidjson/document.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TH1F.h>

class meta: public tools
{
    public:
        meta(); 
        ~meta(); 

        void scan_data(TObject* obj); 
        void scan_sow(TObject* obj); 
        void parse_json(std::string inpt); 
        std::string hash(std::string fname); 

        rapidjson::Document* rpd = nullptr;
        std::string metacache_path; 
        meta_t meta_data; 
 
    private:
        void compiler(); 

        float parse_float(std::string key, TTree* tr);
        std::string parse_string(std::string key, TTree* tr); 
}; 

#endif
