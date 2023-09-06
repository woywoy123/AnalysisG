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

            void AddMeta(meta_t, std::string);
            void AddEvent(event_t event, meta_t meta);
            void AddGraph(graph_t graph, meta_t meta); 
            void AddSelection(selection_t selection, meta_t meta); 
            void AddCode(code_t code); 

            std::map<std::string, std::vector<event_t*>> DumpEvents(); 

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

            settings_t settings; 
            export_t state;
            std::string caller = ""; 
            std::map<std::string, int> event_trees = {}; 

    }; 
}

#endif
