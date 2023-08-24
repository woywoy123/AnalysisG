#include "../sampletracer/root.h"
#include "../metadata/metadata.h"
#include "../event/event.h"
#include "../code/code.h"

#ifndef SAMPLETRACER_H
#define SAMPLETRACER_H

namespace SampleTracer
{
    class CySampleTracer
    {
        public:
            CySampleTracer(); 
            ~CySampleTracer(); 
            void AddEvent(event_t event, meta_t meta, std::vector<code_t> code);

            tracer_t Export(); 
            void Import(tracer_t inpt);
            
            std::map<std::string, CyROOT*> root_map; 
            std::map<std::string, Code::CyCode*> code_hashes; 
    }; 
}

#endif
