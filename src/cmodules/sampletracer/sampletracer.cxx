#include "../sampletracer/sampletracer.h"
#include "../tools/tools.h"

namespace SampleTracer
{
    CySampleTracer::CySampleTracer(){}
    CySampleTracer::~CySampleTracer()
    {
        std::map<std::string, CyROOT*>::iterator it;
        it = this -> ROOT_map.begin();  
        for (; it != this -> ROOT_map.end(); ++it){delete it -> second;}
        std::map<std::string, Code::CyCode*>::iterator c_it = this -> event_code.begin(); 
        for (; c_it != this -> event_code.end(); ++c_it){delete c_it -> second;}
    }

    void CySampleTracer::AddEvent(ExportEventTemplate event, ExportMetaData meta, ExportCode code)
    {
        CyROOT* root; 
        unsigned int contains = this -> ROOT_map.count(event.ROOT);
        if (contains){root = ROOT_map[event.ROOT];}
        else 
        {
            root = new CyROOT(meta);
            this -> ROOT_map.insert({event.ROOT, root});
        }

        root -> AddEvent(event);
        contains = this -> event_code.count(event.event_name); 
        if (contains){ return; }
        Code::CyCode* code_ = new Code::CyCode(); 
        code_ -> ImportCode(code);
        this -> event_code.insert({event.event_name, code_});  
    }

    std::map<std::string, unsigned int> CySampleTracer::Length()
    {
        std::map<std::string, CyROOT*>::iterator itr; 
        std::map<std::string, unsigned int> output; 
        std::map<std::string, unsigned int>::iterator itn;

        for (itr = this -> ROOT_map.begin(); itr != this -> ROOT_map.end(); ++itr)
        {
            std::map<std::string, unsigned int> tmp = itr -> second -> n_events; 
            for (itn = tmp.begin(); itn != tmp.end(); ++itn)
            {
                output[itn -> first] += itn -> second; 
            }
        }
        return output; 
    }

    std::map<std::string, Container> CySampleTracer::Search(std::vector<std::string> get)
    {
        auto scanthis = [](std::vector<Container>* out, const std::string get, CyROOT* root)
        {
            std::vector<Container> output = root -> Scan(get); 
            for (unsigned int x(0); x < output.size(); ++x){out -> push_back(output[x]);}
        }; 

        std::map<std::string, CyROOT*> check = {}; 
        std::map<std::string, Container> found = {}; 
        std::map<std::string, std::thread*> threads = {}; 
        std::map<std::string, std::vector<Container>*> output; 

        std::map<std::string, CyROOT*>::iterator itr; 
        for (std::string key : get)
        {
            itr = this -> ROOT_map.begin(); 
            for (; itr != this -> ROOT_map.end(); ++itr)
            {
                std::string key_ = key + "-" + itr -> first; 
                CyROOT* root = itr -> second; 

                Container x; 
                std::vector<Container>* ptr = new std::vector<Container>(); 
                threads[key_] = new std::thread(scanthis, ptr, key, root); 
                output[key_] = ptr; 
            }
        }

        std::map<std::string, std::vector<Container>*>::iterator it; 
        for (it = output.begin(); it != output.end(); ++it)
        {
            std::string key = it -> first; 
            std::thread* th = threads[key]; 
            th -> join(); 
            std::vector<Container>* o = output[key]; 
            for (unsigned int x(0); x < o -> size(); ++x)
            {
                Container* con = &(o -> at(x)); 
                found[con -> hash] = o -> at(x);          
            }
            delete o; 
            delete th; 
        }
        return found; 
    }
}
