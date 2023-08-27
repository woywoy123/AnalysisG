#include "../metadata/metadata.h"
#include "../root/root.h"

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

            static void Make(
                    CyROOT* root, settings_t* apply, std::vector<CyBatch*>* out, 
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

            std::vector<CyBatch*> ReleaseVector(
                    std::vector<std::vector<CyBatch*>*> output, 
                    std::vector<std::thread*> jobs); 


            std::vector<CyBatch*> MakeIterable(); 
            std::map<std::string, int> length();  


            CySampleTracer* operator + (CySampleTracer*); 
            void operator += (CySampleTracer*);
            void iadd(CySampleTracer*); 

            std::map<std::string, Code::CyCode*> code_hashes;
            std::map<std::string, CyROOT*> root_map; 
            settings_t settings; 

    }; 
}

#endif
