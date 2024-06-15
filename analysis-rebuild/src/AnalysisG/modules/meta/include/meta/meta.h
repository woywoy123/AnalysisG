#ifndef META_META_H
#define META_META_H

#include <tools/tools.h>
#include <structs/meta.h>
#include <rapidjson/document.h>

class meta: public tools
{
    public:
        meta(); 
        ~meta(); 

        void parse_json(std::string inpt); 
        rapidjson::Document* rpd = nullptr;
        meta_t meta_data; 

    private:
        void compiler(); 
}; 

#endif
