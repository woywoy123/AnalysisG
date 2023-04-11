#include <iostream>

#ifndef SAMPLE_H
#define SAMPLE_H

namespace Sample
{
    class Event
    {
        public:
            Event(); 
            ~Event(); 
            
            void MakeHash();
            std::string Hash = "";
            signed int EventIndex = -1;
            bool Compiled = false; 
            bool Train = false; 
    };
}
#endif
