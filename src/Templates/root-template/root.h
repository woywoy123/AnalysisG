#ifndef ROOT_H
#define ROOT_H

#include <iostream>

class CyROOT
{
    public: 
        CyROOT(); 
        ~CyROOT(); 

        std::string Filename = ""; 
        std::string SourcePath = ""; 
        std::string CachePath = ""; 

}
#endif
