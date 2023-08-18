#include "code.h"
#include "../tools/tools.h"

namespace Tools
{
    CyCode::CyCode(){}
    CyCode::~CyCode(){}
    void CyCode::Hash()
    {
        this -> hash = Hashing(this -> object_code);
    }

    bool CyCode::operator==(CyCode* inpt)
    {
        this -> Hash();
        inpt -> Hash();
        return this -> hash == inpt -> hash;
    }
}
