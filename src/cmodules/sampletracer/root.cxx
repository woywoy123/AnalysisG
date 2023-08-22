#include "../sampletracer/root.h"
#include "../tools/tools.h"

typedef std::map<std::string, CyTemplate::CyEventTemplate*> ev_names; 


namespace SampleTracer
{
    CyROOT::CyROOT(ExportMetaData meta)
    {
        this -> ImportMetaData(meta);
        this -> n_events = {}; 
    }
    CyROOT::~CyROOT()
    {
        std::map<std::string, ev_names>::iterator it;  
        ev_names::iterator evn; 
        it = this -> events.begin(); 
        for (; it != this -> events.end(); ++it)
        {
            evn = it -> second.begin(); 
            for (; evn != it -> second.end(); ++evn){ delete evn -> second; }
        }
    }

    void CyROOT::AddEvent(ExportEventTemplate event)
    {
        ev_names* this_event; 
        unsigned int contains = this -> events.count(event.event_hash);
        if (contains)
        {
            this_event = &(this -> events[event.event_hash]);
            if (this_event -> count(event.event_name)){ return; }
        }
        else
        {
            this -> events[event.event_hash] = {}; 
            this_event = &(this -> events[event.event_hash]); 
        }
        
        CyTemplate::CyEventTemplate* ev = new CyTemplate::CyEventTemplate();    
        ev -> ImportEventData(event);
        this_event -> insert({ev -> event_name, ev});
        n_events[ev -> event_name] += 1; 
    }

    std::vector<Container> CyROOT::Export(std::vector<std::string> exp)
    { 
        std::vector<Container> output; 
        std::map<std::string, CyTemplate::CyEventTemplate*>* ev;
        std::map<std::string, CyTemplate::CyEventTemplate*>::iterator it; 
        for (unsigned int x(0); x < exp.size(); ++x)
        {
            Container con; 
            con.meta = this -> MakeMapping(); 
            
            std::string hash = exp[x]; 
            ev = &(this -> events[hash]);
            it = ev -> begin();  
            for (; it != ev -> end(); ++it)
            {
                con.event_name.push_back( it -> first );         
                con.events.push_back( it -> second -> MakeMapping() ); 
            }

            con.hash = hash; 
            output.push_back(con);
        }
        return output; 
    }

    std::vector<Container> CyROOT::Export(
                    std::map<std::string, std::map<std::string, CyTemplate::CyEventTemplate*>> exp)
    {
        std::map<std::string, Container> output;
        std::map<std::string, std::map<std::string, CyTemplate::CyEventTemplate*>>::iterator it_h;
        std::vector<std::string> get = {}; 
        for (it_h = exp.begin(); it_h != exp.end(); ++it_h){get.push_back( it_h -> first );}
        return this -> Export(get); 
    }

    std::vector<Container> CyROOT::Scan(std::string get)
    {

        if (count(get, ".root"))
        {
            if (count(get, this -> original_input)){return this -> Export(this -> events);}
            if (count(get, this -> DatasetName)){return this -> Export(this -> events);}
        }

        std::vector<std::string> getter; 
        std::map<std::string, CyTemplate::CyEventTemplate*> event; 
        std::map<std::string, CyTemplate::CyEventTemplate*>::iterator ev_it; 
        std::map<std::string, std::map<std::string, CyTemplate::CyEventTemplate*>>::iterator it_h;
        for (it_h = this -> events.begin(); it_h != this -> events.end(); ++it_h)
        {
            std::string hash = it_h -> first; 

            if (count(get, hash)){ getter.push_back(hash); break; }
            event = it_h -> second; 
            for (ev_it = event.begin(); ev_it != event.end(); ++ev_it)            
            {
                std::string event_name = ev_it -> first;
                CyTemplate::CyEventTemplate* ev = ev_it -> second; 

                // Collect this event if it has the requested implementation
                if (count(get, event_name)){ getter.push_back(hash); break; }

                // Get this particular event tag
                if (count(get, ev -> event_tagging)){ getter.push_back(hash); break; }
               
                // Check if the event has this tree
                if (count(get, ev -> event_tree)){ getter.push_back(hash); break; }

                // Check if the event is part of a ROOT file (see event.ROOT)
                if (count(get, ev -> ROOT)){ getter.push_back(hash); break; }

                // Check whether the DAOD is requested. 
                if (count(get, this -> IndexToSample(ev -> event_index)))
                {
                    getter.push_back(hash); 
                    break; 
                }
            }
        }
        return this -> Export(getter); 
    }

}
