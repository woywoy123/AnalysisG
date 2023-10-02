#include "../metadata/metadata.h"
#include "../root/root.h"
#include <algorithm>

#ifndef SAMPLETRACER_H
#define SAMPLETRACER_H

namespace SampleTracer
{
    namespace CyHelpers
    {
        static void Make(
                CyROOT* root, 
                settings_t* apply, 
                std::vector<CyBatch*>* out, 
                std::map<std::string, Code::CyCode*>* code_hashes)
        {
            std::map<std::string, CyBatch*>::iterator itb; 
            itb = root -> batches.begin(); 
            for (; itb != root -> batches.end(); ++itb)
            {
                CyBatch* this_b = itb -> second; 
                this_b -> ApplySettings(apply);  

                if (!this_b -> valid){continue;}
                this_b -> ApplyCodeHash(code_hashes); 
                out -> push_back(this_b); 
            }
        }; 

        static std::vector<CyBatch*> ReleaseVector(
                std::vector<std::vector<CyBatch*>*> output, 
                std::vector<std::thread*> jobs, 
                const unsigned int threads)
        {
            std::vector<CyBatch*> release = {}; 
            for (unsigned int x(0); x < output.size(); ++x)
            {
                if (threads != 1){ jobs[x] -> join(); }
                release.insert(release.end(), output[x] -> begin(), output[x] -> end()); 
                output[x] -> clear(); 
                delete output[x]; 
                if (threads != 1){ delete jobs[x]; }
            }
            return release; 
        };  

        static void ExportCode(
                std::map<std::string, code_t>* output, 
                std::map<std::string, Code::CyCode*> code_hashes)
        {
            std::map<std::string, Code::CyCode*>::iterator itc;
            itc = code_hashes.begin();
            for (; itc != code_hashes.end(); ++itc){
                Code::CyCode* code = itc -> second; 
                if (!code -> container.object_code.size()){continue;}
                (*output)[itc -> first] = code -> ExportCode();
            }
        }; 

        static void ImportCode(
                std::map<std::string, Code::CyCode*>* output,
                std::map<std::string, code_t>* hashed_code) 
        {
            std::map<std::string, code_t>::iterator itc; 
            itc = hashed_code -> begin(); 
            for (; itc != hashed_code -> end(); ++itc){
                code_t* co = &(itc -> second); 
                if (!co -> object_code.size()){continue;}
                if (output -> count(co -> hash)){ continue; }
                Code::CyCode* co_ = new Code::CyCode();
                co_ -> ImportCode(*co, *hashed_code); 
                (*output)[co -> hash] = co_;
            }
        }; 
    }

    class CySampleTracer
    {
        public:
            CySampleTracer(); 
            ~CySampleTracer(); 

            template<typename G>
            CyROOT* AddContent(G* type, meta_t* meta, std::map<std::string, CyROOT*>* root)
            {
                std::string event_root = type -> event_root; 
                if (!root -> count(event_root)){
                    (*root)[event_root] = new CyROOT(*meta); 
                }
                return root -> at(event_root); 
            }; 
            
            template <typename G>
            void ReleaseObjects(std::map<std::string, std::vector<G*>>* out)
            {
                std::map<std::string, CyROOT*>* roots = &(this -> root_map);
                std::map<std::string, CyROOT*>::iterator itr = roots -> begin();
                typename std::map<std::string, std::vector<G*>>::iterator itG;
                typename std::map<std::string, std::vector<G*>> tmp;
                typename std::vector<G*> app; 
                for (; itr != roots -> end(); ++itr){
                    tmp = {}; 
                    std::string r_name = itr -> first; 
                    itr -> second -> ReleaseObjects(&tmp); 
                    itG = tmp.begin(); 
                    for(; itG != tmp.end(); ++itG){
                        app = itG -> second; 
                        std::string name = r_name + ":" + itG -> first; 
                        (*out)[name].insert((*out)[name].end(), app.begin(), app.end());
                    }
                }
            }; 

            std::string make_flush_string(std::string root_name, std::string subkey)
            {
                std::string get = ""; 
                get += this -> settings.outputdirectory; 
                get += this -> settings.projectname + "/" + subkey + "/"; 
                get += root_name; 
                if (get.rfind(".root.1") == std::string::npos){}
                else {get.erase(get.rfind(".root.1"), get.size()-1);}
                get += ".hdf5";
                return get;
            }; 

            void AddMeta(meta_t, std::string);
            void AddEvent(event_t event, meta_t meta);
            void AddGraph(graph_t graph, meta_t meta); 
            void AddSelection(selection_t selection, meta_t meta); 
            void AddCode(code_t code); 
            CyBatch* RegisterHash(std::string hash, std::string event_root); 

            std::map<std::string, std::vector<CyEventTemplate*>> DumpEvents(); 
            std::map<std::string, std::vector<CyGraphTemplate*>> DumpGraphs(); 
            std::map<std::string, std::vector<CySelectionTemplate*>> DumpSelections();

            void FlushEvents(std::vector<std::string> hashes); 
            void FlushGraphs(std::vector<std::string> hashes); 
            void FlushSelections(std::vector<std::string> hashes); 

            void DumpTracer();

            tracer_t Export(); 
            void Import(tracer_t inpt);

            settings_t ExportSettings();
            void ImportSettings(settings_t inpt); 

            std::vector<CyBatch*> MakeIterable(); 
            std::map<std::string, int> length();  

            CySampleTracer* operator + (CySampleTracer*); 
            void operator += (CySampleTracer*);
            void iadd(CySampleTracer*); 

            std::map<std::string, std::string> link_event_code = {}; 
            std::map<std::string, std::string> link_graph_code = {};
            std::map<std::string, std::string> link_selection_code = {}; 
            std::map<std::string, Code::CyCode*> code_hashes = {};
            std::map<std::string, CyROOT*> root_map = {}; 

            export_t state;
            settings_t settings; 
            std::string caller = ""; 
            std::map<std::string, int> event_trees = {}; 

    }; 
}

#endif
