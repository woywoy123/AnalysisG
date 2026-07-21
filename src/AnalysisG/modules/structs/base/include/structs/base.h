#ifndef ANALYSISG_STRUCTS_BASE_H
#define ANALYSISG_STRUCTS_BASE_H
#include <structs/enums.h> // <---- go here first
#include <tools/tools.h>

#include <vector>
#include <string>

namespace AnalysisG {
    namespace core {
        struct element_t {
            public: 
                element_t();
                ~element_t();  

                long index = 0; 
                bool clear = false; 

                AnalysisG::enums::data type = AnalysisG::enums::data::unset;  

            private: 
                void _flush_buffer();

                template <typename g>
                bool _getalt(g* v, g* out){
                    if (!v){return false;}
                    *out = *v; 
                    return true;
                }
    
                template <typename g>
                bool _getalt(std::vector<g>* vv, g* v, g* out){
                    if (!vv && !v){return false;}
                    else if (vv  && vv -> size() > this -> index){*out = (*vv)[this -> index];}
                    else if (!vv && v){*out = *v;}
                    else {return false;}
                    return true;
                }

                template <typename T>
                bool _flush_buffer(std::vector<T>** _data){
                    if (!(*_data)){return false;}
                    if (!this -> clear){(*_data) -> clear(); return true;}
                    AnalysisG::tooling::pflush(_data);  
                    return true; 
                } 

                template <typename T>
                bool _flush_buffer(T** _data){
                    if (!(*_data)){return false;}
                    if (!this -> clear){(**_data) = 0; return true;}
                    AnalysisG::tooling::pflush(_data);  
                    return true; 
                }     
            }; 
    }
}


#endif
