#include "../code/code.h"

namespace Code
{
    CyCode::CyCode(){}
    CyCode::~CyCode(){}
    void CyCode::Hash()
    {
        this -> hash = Hashing(this -> object_code);
    }

    bool CyCode::operator==(CyCode* inpt)
    {
        this -> Hash();
        inpt -> Hash();
        return this -> hash == inpt -> hash;
    }

    ExportCode CyCode::MakeMapping()
    {
        ExportCode tmp; 
        tmp.input_params          = this -> input_params;     
        tmp.co_vars               = this -> co_vars;      
        tmp.param_space           = this -> param_space;
        
        tmp.function_name         = this -> function_name;
        tmp.class_name            = this -> class_name;    
        tmp.hash                  = this -> hash;       
        tmp.source_code           = this -> source_code;
        tmp.object_code           = this -> object_code;       
        
        tmp.is_class              = this -> is_class;          
        tmp.is_function           = this -> is_function;       
        tmp.is_callable           = this -> is_callable;       
        tmp.is_initialized        = this -> is_initialized;    
        tmp.has_param_variable    = this -> has_param_variable;
        return tmp; 
    }

    void CyCode::ImportCode(ExportCode code)
    {
        this -> input_params          = code.input_params;     
        this -> co_vars               = code.co_vars;      
        this -> param_space           = code.param_space;
        
        this -> function_name         = code.function_name;
        this -> class_name            = code.class_name;    
        this -> hash                  = code.hash;       
        this -> source_code           = code.source_code;
        this -> object_code           = code.object_code;       
        
        this -> is_class              = code.is_class;          
        this -> is_function           = code.is_function;       
        this -> is_callable           = code.is_callable;       
        this -> is_initialized        = code.is_initialized;    
        this -> has_param_variable    = code.has_param_variable;
    }
}
