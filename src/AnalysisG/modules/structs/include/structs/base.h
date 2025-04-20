#ifndef STRUCTS_BASE_H
#define STRUCTS_BASE_H
#include <vector>
#include <string>
#include <structs/enums.h> // <---- go here first

void buildPCM(std::string name, std::string incl, bool exl); 
void registerInclude(std::string name, bool is_abs = false); 
void buildDict(std::string name, std::string incl);
void buildAll();

struct bsc_t {

    public:

        bsc_t();  
        virtual ~bsc_t(); 
        void flush_buffer(); 

        std::string as_string();
        std::string scan_buffer(); 
        data_enum root_type_translate(std::string*); 

        // ------------- (1). add the data type here -------------- //
        bool element(std::vector<std::vector<std::vector<float>>>*  el);
        bool element(std::vector<std::vector<std::vector<double>>>* el);
        bool element(std::vector<std::vector<std::vector<long>>>*   el);
        bool element(std::vector<std::vector<std::vector<int>>>*    el);
        bool element(std::vector<std::vector<std::vector<bool>>>*   el);

        bool element(std::vector<std::vector<float>>*  el);
        bool element(std::vector<std::vector<double>>* el);
        bool element(std::vector<std::vector<long>>*   el);
        bool element(std::vector<std::vector<int>>*    el);
        bool element(std::vector<std::vector<bool>>*   el);

        bool element(std::vector<float>* el); 
        bool element(std::vector<double>* el); 
        bool element(std::vector<long>* el); 
        bool element(std::vector<int>* el); 
        bool element(std::vector<char>* el); 
        bool element(std::vector<bool>* el); 

        bool element(double* el); 
        bool element(float* el);
        bool element(long* el);
        bool element(int* el); 
        bool element(bool* el); 
        bool element(unsigned long long* el); 
        bool element(unsigned int* el); 
        bool element(char* el); 

        std::vector<std::vector<std::vector<unsigned long long>>>* vvv_ull = nullptr;
        std::vector<std::vector<std::vector<unsigned int>>>*       vvv_ui  = nullptr;
        std::vector<std::vector<std::vector<double>>>*             vvv_d   = nullptr;
        std::vector<std::vector<std::vector<long>>>*               vvv_l   = nullptr;
        std::vector<std::vector<std::vector<float>>>*              vvv_f   = nullptr;
        std::vector<std::vector<std::vector<int>>>*                vvv_i   = nullptr;
        std::vector<std::vector<std::vector<bool>>>*               vvv_b   = nullptr;
        std::vector<std::vector<std::vector<char>>>*               vvv_c   = nullptr;

        std::vector<std::vector<unsigned long long>>*               vv_ull = nullptr;
        std::vector<std::vector<unsigned int>>*                     vv_ui  = nullptr;
        std::vector<std::vector<double>>*                           vv_d   = nullptr;
        std::vector<std::vector<long>>*                             vv_l   = nullptr;
        std::vector<std::vector<float>>*                            vv_f   = nullptr;
        std::vector<std::vector<int>>*                              vv_i   = nullptr;
        std::vector<std::vector<bool>>*                             vv_b   = nullptr;
        std::vector<std::vector<char>>*                             vv_c   = nullptr;

        std::vector<unsigned long long>*                             v_ull = nullptr;
        std::vector<unsigned int>*                                   v_ui  = nullptr;
        std::vector<double>*                                         v_d   = nullptr;
        std::vector<long>*                                           v_l   = nullptr;
        std::vector<float>*                                          v_f   = nullptr;
        std::vector<int>*                                            v_i   = nullptr;
        std::vector<bool>*                                           v_b   = nullptr;
        std::vector<char>*                                           v_c   = nullptr;

        unsigned long long*                                            ull = nullptr;
        unsigned int*                                                  ui  = nullptr;
        double*                                                        d   = nullptr;
        long*                                                          l   = nullptr;
        float*                                                         f   = nullptr;
        int*                                                           i   = nullptr;
        bool*                                                          b   = nullptr;
        char*                                                          c   = nullptr;
        // ========================================================================= //
        
        long index = 0; 
        bool clear = false; 
        data_enum type = data_enum::unset;  

        template <typename T>
        bool flush_buffer(std::vector<T>** data){
            if (!(*data)){return false;}
            if (!this -> clear){(*data) -> clear(); return true;}
            delete (*data); (*data) = nullptr;
            return true; 
        } 

        template <typename T>
        bool flush_buffer(T** data){
            if (!(*data)){return false;}
            if (!this -> clear){(**data) = 0; return true;}
            else {delete (*data); (*data) = nullptr;}
            return true; 
        } 

    private: 
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
};






#endif
