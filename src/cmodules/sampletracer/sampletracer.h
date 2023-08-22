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
            void AddEvent(ExportEventTemplate event, ExportMetaData meta, ExportCode code);
            std::map<std::string, Container> Search(std::vector<std::string> get); 
            std::map<std::string, unsigned int> Length(); 
           
            std::map<std::string, CyROOT*> ROOT_map; 
            std::map<std::string, Code::CyCode*> event_code; 
    }; 
}

#endif
