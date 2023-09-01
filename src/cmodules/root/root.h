#include "../abstractions/cytypes.h"
#include "../selection/selection.h"
#include "../event/event.h"
#include "../graph/graph.h"

#ifndef ROOT_H
#define ROOT_H

using namespace CyTemplate; 

namespace SampleTracer
{
    class CyBatch
    {
        public:
            CyBatch(std::string hash); 
            ~CyBatch(); 
            std::string Hash(); 
            void Import(const meta_t*); 
            void Import(const event_t*);
            void Import(const graph_t*); 
            void Import(const selection_t*); 
            void Import(const batch_t*); 

            batch_t ExportPickled(); 
            void ImportPickled(const batch_t*); 

            void Export(batch_t* exp); 
            batch_t Export(); 


            void Contextualize(); 
            void ApplySettings(const settings_t*); 
            void LinkCode(
                    std::map<std::string, std::string>* inpt, 
                    std::map<std::string, Code::CyCode*>* link, 
                    const std::map<std::string, Code::CyCode*>* code_h); 

            void ApplyCodeHash(const std::map<std::string, Code::CyCode*>*);

            std::map<std::string, CyEventTemplate*> events; 
            std::map<std::string, CyGraphTemplate*> graphs; 
            std::map<std::string, CySelectionTemplate*> selections; 
           
            std::string hash = ""; 

            const meta_t* meta;
            bool lock_meta = false;
            
            // Object Context
            bool get_event = false; 
            bool get_graph = false; 
            bool get_selection = false;
            bool valid = false; 

            CyEventTemplate* this_ev = nullptr; 
            CyGraphTemplate* this_gr = nullptr; 
            CySelectionTemplate* this_sel = nullptr; 

            std::string this_event_name = ""; 
            std::string this_graph_name = "";
            std::string this_selection_name = ""; 
            std::string this_tree = "";

            std::map<std::string, Code::CyCode*> code_hashes = {}; 

        private: 
            template <typename G> 
            void destroy(std::map<std::string, G*>* input)
            {
                typename std::map<std::string, G*>::iterator it; 
                it = input -> begin(); 
                for (; it != input -> end(); ++it){delete it -> second;}
                input -> clear(); 
            };

            template <typename G, typename T>
            void export_this(
                    std::map<std::string, G*> inpt, 
                    std::map<std::string, T>* out)
            {
                typename std::map<std::string, G*>::iterator it; 
                it = inpt.begin();
                for (; it != inpt.end(); ++it){
                    (*out)[it -> first] = it -> second -> Export(); 
                }
            }; 

            template <typename G>
            void code_link(
                    std::map<std::string, G*> event, 
                    const std::map<std::string, Code::CyCode*>* code_h)
            {
                typename std::map<std::string, G*>::iterator it; 
                it = event.begin();
                for (; it != event.end(); ++it){
                    std::string code_hash; 
                    G* entry = it -> second; 
                    if (entry -> is_event){
                        code_hash = entry -> event.code_hash;
                    }
                    else if (entry -> is_graph){
                        code_hash = entry -> graph.code_hash; 
                    }
                    else if (entry -> is_selection){
                        code_hash = entry -> selection.code_hash; 
                    }
                    if (!code_h -> count(code_hash)){ continue; }
                    entry -> code_link = code_h -> at(code_hash);  
                } 
            }; 

            template <typename G>
            void export_code(
                    std::map<std::string, G*> entry, 
                    std::map<std::string, code_t>* out)
            {
                std::map<std::string, Code::CyCode*>::iterator itc; 
                typename std::map<std::string, G*>::iterator it; 
                it = entry.begin(); 
                for (; it != entry.end(); ++it){
                    Code::CyCode* link = it -> second -> code_link; 
                    itc = link -> dependency.begin(); 
                    for (; itc != link -> dependency.end(); ++itc){
                        (*out)[itc -> first] = itc -> second -> ExportCode();
                    }
                    (*out)[link -> hash] = link -> ExportCode(); 
                }
            }; 

            template<typename G>
            void not_code_owner(std::map<std::string, G*>* type)
            {
                typename std::map<std::string, G*>::iterator it;
                it = type -> begin(); 
                for (; it != type -> end(); ++it){ 
                    it -> second -> code_owner = false; 
                }
            };


    }; 

    class CyROOT
    {
        public:
            CyROOT(meta_t meta);  
            ~CyROOT(); 

            static void Make(CyBatch* smpls, batch_t* exp_container){
                smpls -> Export(exp_container);     
            }; 

            root_t Export(); 
            void Import(const root_t* inpt); 

            void AddEvent(const event_t* event);
            void AddGraph(const graph_t* graph);
            void UpdateSampleStats(); 
            
            meta_t meta;
            std::map<std::string, CyBatch*> batches;
            std::map<std::string, int> n_events = {};
            std::map<std::string, int> n_graphs = {}; 
            std::map<std::string, int> n_selections = {}; 
            int total_hashes = 0; 
        
        private: 
            template<typename G> 
            void ImportBatch(
                    const std::map<std::string, G>* to_import, 
                    std::map<std::string, CyBatch*>* batches, 
                    const meta_t* meta)
            {
                typename std::map<std::string, G>::const_iterator it; 
                it = to_import -> begin();
                for (; it != to_import -> end(); ++it){
                    const std::string event_h = it -> second.event_hash; 
                    CyBatch* bt; 
                    if (batches -> count(event_h)){bt = batches -> at(event_h);}
                    else {bt = new CyBatch(event_h); (*batches)[event_h] = bt;}
                    bt -> Import(meta); 
                    bt -> Import(&it -> second); 
                    std::string name = it -> second.event_name; 
                    std::string tree = it -> second.event_tree;
                    std::string key = tree + "/" + name;  
                    if (it -> second.event){ this -> n_events[key] += 1; }
                    else if (it -> second.graph){ this -> n_graphs[key] += 1; }
                    else if (it -> second.selection){ this -> n_selections[key] += 1; }
                }
            };
    };
}
#endif
