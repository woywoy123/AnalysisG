#include "../abstractions/cytypes.h"
#include <sstream> 

#ifndef CODE_H
#define CODE_H

namespace Code
{
    class CyCode
    {
        public:
            CyCode();
            ~CyCode();

            void Hash(); 
            void ImportCode(code_t code);
            code_t ExportCode(); 
            
            code_t container;
            bool operator == (CyCode& code);
            std::string hash = ""; 
    };
}

#endif
