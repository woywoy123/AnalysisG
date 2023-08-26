#include "../code/code.h"

namespace Code
{
    CyCode::CyCode(){}
    CyCode::~CyCode(){}

    void CyCode::Hash()
    {
        std::hash<std::string> hasher; 
        std::stringstream ss; 

        std::string obj_c = this -> container.object_code; 
        ss << "0x" << std::hex << hasher(obj_c); 
        std::string out = ss.str(); 
        int diff = out.size() - 18; 
        if (diff) { out += std::string(std::abs(diff), '0'); }
        this -> hash = out; 
    }

    bool CyCode::operator==(CyCode& inpt)
    {
        inpt.Hash(); 
        this -> Hash(); 
        return this -> hash == inpt.hash;
    }

    code_t CyCode::ExportCode()
    {
        this -> Hash();
        this -> container.hash = this -> hash;
        return this -> container; 
    }

    void CyCode::ImportCode(code_t code)
    {
        this -> container = code;
    }
}
