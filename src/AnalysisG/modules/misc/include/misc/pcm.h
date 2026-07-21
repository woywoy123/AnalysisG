#ifndef ANALYSISG_MISC_PCM_H
#define ANALYSISG_MISC_PCM_H

#include <structs/enums.h>
#include <vector>
#include <string>

namespace AnalysisG {
    namespace misc {
        void buildPCM(AnalysisG::enums::data, std::string incl, bool exl); 
        void registerInclude(std::string name, bool is_abs = false); 
        void buildDict(std::string name, std::string incl);
        void buildAll();
    }
}

#endif
