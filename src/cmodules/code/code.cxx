#include "../code/code.h"

namespace Code
{
    CyCode::CyCode(){}
    CyCode::~CyCode(){}

    void CyCode::Hash()
    {
        std::string obj_c = this -> container.object_code; 
        this -> CyBase::Hash(obj_c); 
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
