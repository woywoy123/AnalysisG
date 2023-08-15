#ifndef METADATA_H
#define METADATA_H

#include <iostream>
#include <map>

namespace SampleTracer
{
    class CyMetaData
    {
        public:
            CyMetaData(); 
            ~CyMetaData(); 

            unsigned int dsid = 0; 
            std::string AMITag = ""; 
            std::string generators = ""; 

            void addsamples(int index, std::string sample);
            void addconfig(std::string key, std::string val); 

            bool isMC = true; 
            std::string derivationFormat = ""; 
            std::map<int, std::string> inputfiles = {}; 
            std::map<std::string, std::string> config = {}; 

           unsigned int eventNumber = 0; 


    }; 
}

#endif
