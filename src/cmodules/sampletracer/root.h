#include "../metadata/metadata.h"
#include "../event/event.h"

#ifndef ROOT_H
#define ROOT_H

struct Container
{
    std::string hash; 
    std::vector<std::string> event_name; 
    std::vector<ExportEventTemplate> events; 
    ExportMetaData meta; 
};

namespace SampleTracer
{
    class CyROOT : public CyMetaData
    {
        public:
            CyROOT(ExportMetaData); 
            ~CyROOT(); 
            void AddEvent(ExportEventTemplate event); 
            std::vector<Container> Scan(std::string get); 

            std::vector<Container> Export(std::vector<std::string> exp); 
            std::vector<Container> Export(std::map<std::string, std::map<std::string, CyTemplate::CyEventTemplate*>> exp); 
 
            std::map<std::string, std::map<std::string, CyTemplate::CyEventTemplate*>> events;
            std::map<std::string, unsigned int> n_events; 
    };
}
#endif
