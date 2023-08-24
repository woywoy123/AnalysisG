#include "../abstractions/cytypes.h"
#include "../abstractions/abstractions.h"

#ifndef CODE_H
#define CODE_H

namespace Code
{
    class CyCode : public Abstraction::CyBase
    {
        public:
            CyCode();
            ~CyCode();

            void Hash(); 
            void ImportCode(code_t code);
            code_t ExportCode(); 
            
            code_t container;
            bool operator==(CyCode* code);
    };
}

#endif
