#include <structs/base.h>
#include <TInterpreter.h>
#include <iostream>

void buildDict(std::string _name, std::string _shrt){
    std::string name = std::string(_name);
    gInterpreter -> GenerateDictionary(name.c_str(), _shrt.c_str()); 
}

void registerInclude(std::string pth, bool is_abs){
    std::string shrt = "#include ";
    if (is_abs){shrt += " \"" + pth + "\"";}
    else {shrt += "<" + pth + ">";}
    gInterpreter -> ProcessLine(shrt.c_str());
}

void buildPCM(std::string name, std::string incl, bool exl){
    if (exl){return;}
    buildDict(name, incl); 
}

// ============================= Add your type (2) =================================== //

void buildAll(){
    buildPCM("vector<vector<vector<unsigned long long>>>", "vector", false); 
    buildPCM("vector<vector<vector<unsigned int>>>"      , "vector", false); 
    buildPCM("vector<vector<vector<double>>>"            , "vector", false); 
    buildPCM("vector<vector<vector<long>>>"              , "vector", false); 
    buildPCM("vector<vector<vector<float>>>"             , "vector", false); 
    buildPCM("vector<vector<vector<int>>>"               , "vector", false); 
    buildPCM("vector<vector<vector<bool>>>"              , "vector", false); 
    buildPCM("vector<vector<vector<char>>>"              , "vector", false); 
    buildPCM("vector<vector<unsigned long long>>"        , "vector", false); 
    buildPCM("vector<vector<unsigned int>>"              , "vector", false); 
    buildPCM("vector<vector<double>>"                    , "vector", false); 
    buildPCM("vector<vector<long>>"                      , "vector", false); 
    buildPCM("vector<vector<float>>"                     , "vector", false); 
    buildPCM("vector<vector<int>>"                       , "vector", false); 
    buildPCM("vector<vector<bool>>"                      , "vector", false); 
    buildPCM("vector<vector<char>>"                      , "vector", false); 
    buildPCM("vector<unsigned long long>"                , "vector", false); 
    buildPCM("vector<unsigned int>"                      , "vector", false); 
    buildPCM("vector<double>"                            , "vector", false); 
    buildPCM("vector<long>"                              , "vector", false); 
    buildPCM("vector<float>"                             , "vector", false); 
    buildPCM("vector<int>"                               , "vector", false); 
    buildPCM("vector<bool>"                              , "vector", false); 
    buildPCM("vector<char>"                              , "vector", false); 
    buildPCM("unsigned long long"                        , ""      , true ); 
    buildPCM("unsigned int"                              , ""      , true ); 
    buildPCM("double"                                    , ""      , true );      
    buildPCM("long"                                      , ""      , true );        
    buildPCM("float"                                     , ""      , true );       
    buildPCM("int"                                       , ""      , true );        
    buildPCM("bool"                                      , ""      , true );        
    buildPCM("char"                                      , ""      , true );        
}

bsc_t::bsc_t(){}
bsc_t::~bsc_t(){}

int count(const std::string* str, const std::string sub){
    int count = 0;
    std::string::size_type pos = 0;
    while ((pos = str -> find(sub, pos)) != std::string::npos){++count; ++pos;}
    return count;
}

// -------------------- (3). add the routing -------------- //
data_enum bsc_t::root_type_translate(std::string* root_str){
    int vec = count(root_str, "vector"); 
    if (vec == 0 && (*root_str) ==   "Float_t"){return data_enum::v_f  ;}
    if (vec == 0 && (*root_str) ==  "Double_t"){return data_enum::v_d  ;}
    if (vec == 0 && (*root_str) ==    "UInt_t"){return data_enum::v_ui ;}
    if (vec == 0 && (*root_str) ==     "Int_t"){return data_enum::v_i  ;}
    if (vec == 0 && (*root_str) ==    "Char_t"){return data_enum::v_c  ;}
    if (vec == 0 && (*root_str) ==      "char"){return data_enum::v_c  ;}
    if (vec == 0 && (*root_str) == "ULong64_t"){return data_enum::v_ull;}
  
    if (vec == 0 && count(root_str, "float")){return   data_enum::v_f;}
    if (vec == 1 && count(root_str, "float")){return  data_enum::vv_f;}
    if (vec == 2 && count(root_str, "float")){return data_enum::vvv_f;}

    if (vec == 0 && count(root_str, "double")){return   data_enum::v_d;}
    if (vec == 1 && count(root_str, "double")){return  data_enum::vv_d;}
    if (vec == 2 && count(root_str, "double")){return data_enum::vvv_d;}

    if (vec == 0 && count(root_str, "int")){return   data_enum::v_i;}
    if (vec == 1 && count(root_str, "int")){return  data_enum::vv_i;}
    if (vec == 2 && count(root_str, "int")){return data_enum::vvv_i;}

    if (vec == 0 && count(root_str, "long")){return   data_enum::v_l;}
    if (vec == 1 && count(root_str, "long")){return  data_enum::vv_l;}
    if (vec == 2 && count(root_str, "long")){return data_enum::vvv_l;}

    if (vec == 0 && count(root_str, "bool")){return   data_enum::v_b;}
    if (vec == 1 && count(root_str, "bool")){return  data_enum::vv_b;}
    if (vec == 2 && count(root_str, "bool")){return data_enum::vvv_b;}

    if (vec == 1 && count(root_str, "char")){return  data_enum::vv_c;}
    return data_enum::undef; 
}

std::string bsc_t::as_string(){
    switch (this -> type){
        case data_enum::vvv_ull: return "vector<vector<vector<unsigned long long>>>";  
        case data_enum::vvv_ui:  return "vector<vector<vector<unsigned int>>>";        
        case data_enum::vvv_d:   return "vector<vector<vector<double>>>";              
        case data_enum::vvv_l:   return "vector<vector<vector<long>>>";                
        case data_enum::vvv_f:   return "vector<vector<vector<float>>>";               
        case data_enum::vvv_i:   return "vector<vector<vector<int>>>" ;                
        case data_enum::vvv_b:   return "vector<vector<vector<bool>>>";                
        case data_enum::vvv_c:   return "vector<vector<vector<char>>>";                
        case data_enum::vv_ull:  return "vector<vector<unsigned long long>>";          
        case data_enum::vv_ui:   return "vector<vector<unsigned int>>";                
        case data_enum::vv_d:    return "vector<vector<double>>";                      
        case data_enum::vv_l:    return "vector<vector<long>>";                        
        case data_enum::vv_f:    return "vector<vector<float>>";                       
        case data_enum::vv_i:    return "vector<vector<int>>" ;                        
        case data_enum::vv_b:    return "vector<vector<bool>>";                        
        case data_enum::vv_c:    return "vector<vector<char>>";                        
        case data_enum::v_ull:   return "vector<unsigned long long>";                  
        case data_enum::v_ui:    return "vector<unsigned int>";                        
        case data_enum::v_d:     return "vector<double>";                              
        case data_enum::v_l:     return "vector<long>";                                
        case data_enum::v_f:     return "vector<float>";                               
        case data_enum::v_i:     return "vector<int>" ;                                
        case data_enum::v_b:     return "vector<bool>";                                
        case data_enum::v_c:     return "vector<char>";                                
        case data_enum::ull:     return "unsigned long long";                          
        case data_enum::ui:      return "unsigned int";                                
        case data_enum::d:       return "double";                                      
        case data_enum::l:       return "long";                                        
        case data_enum::f:       return "float";                                       
        case data_enum::i:       return "int" ;                                        
        case data_enum::b:       return "bool";                                        
        case data_enum::c:       return "char";                                        
        case data_enum::unset:   return "unset";                                       
        default:                 return "undef";                                                   
    }
}

std::string bsc_t::scan_buffer(){
    std::string x = ""; 
    if (this -> vvv_ull){x += " | vector<vector<vector<unsigned long long>>>";}
    if (this -> vvv_ui ){x += " | vector<vector<vector<unsigned int>>>";      }
    if (this -> vvv_d  ){x += " | vector<vector<vector<double>>>";            }
    if (this -> vvv_l  ){x += " | vector<vector<vector<long>>>";              }
    if (this -> vvv_f  ){x += " | vector<vector<vector<float>>>";             }
    if (this -> vvv_i  ){x += " | vector<vector<vector<int>>>" ;              }
    if (this -> vvv_b  ){x += " | vector<vector<vector<bool>>>";              }
    if (this -> vvv_c  ){x += " | vector<vector<vector<char>>>";              }
    if (this -> vv_ull ){x += " | vector<vector<unsigned long long>>";        }
    if (this -> vv_ui  ){x += " | vector<vector<unsigned int>>";              }
    if (this -> vv_d   ){x += " | vector<vector<double>>";                    }
    if (this -> vv_l   ){x += " | vector<vector<long>>";                      }
    if (this -> vv_f   ){x += " | vector<vector<float>>";                     }
    if (this -> vv_i   ){x += " | vector<vector<int>>" ;                      }
    if (this -> vv_b   ){x += " | vector<vector<bool>>";                      }
    if (this -> vv_c   ){x += " | vector<vector<char>>";                      }
    if (this -> v_ull  ){x += " | vector<unsigned long long>";                }
    if (this -> v_ui   ){x += " | vector<unsigned int>";                      }
    if (this -> v_d    ){x += " | vector<double>";                            }
    if (this -> v_l    ){x += " | vector<long>";                              }
    if (this -> v_f    ){x += " | vector<float>";                             }
    if (this -> v_i    ){x += " | vector<int>" ;                              }
    if (this -> v_b    ){x += " | vector<bool>";                              }
    if (this -> v_c    ){x += " | vector<char>";                              }
    if (this -> ull    ){x += " | unsigned long long";                        }
    if (this -> ui     ){x += " | unsigned int";                              }
    if (this -> d      ){x += " | double";                                    }
    if (this -> l      ){x += " | long";                                      }
    if (this -> f      ){x += " | float";                                     }
    if (this -> i      ){x += " | int" ;                                      }
    if (this -> b      ){x += " | bool";                                      }
    if (this -> c      ){x += " | char";                                      }
    return (x.size()) ? x : "undefined/unset"; 
}                                                         
 
// ------------ (4.) Add the buffer flush -------------------- //
void bsc_t::flush_buffer(){
    switch (this -> type){
        case data_enum::vvv_ull:  this -> flush_buffer(&this -> vvv_ull); return;  
        case data_enum::vvv_ui:   this -> flush_buffer(&this -> vvv_ui ); return;  
        case data_enum::vvv_d:    this -> flush_buffer(&this -> vvv_d  ); return;  
        case data_enum::vvv_l:    this -> flush_buffer(&this -> vvv_l  ); return;  
        case data_enum::vvv_f:    this -> flush_buffer(&this -> vvv_f  ); return;  
        case data_enum::vvv_i:    this -> flush_buffer(&this -> vvv_i  ); return;  
        case data_enum::vvv_b:    this -> flush_buffer(&this -> vvv_b  ); return;  
        case data_enum::vvv_c:    this -> flush_buffer(&this -> vvv_c  ); return;  
        
        case data_enum::vv_ull:   this -> flush_buffer(&this -> vv_ull ); return;  
        case data_enum::vv_ui:    this -> flush_buffer(&this -> vv_ui  ); return;  
        case data_enum::vv_d:     this -> flush_buffer(&this -> vv_d   ); return;  
        case data_enum::vv_l:     this -> flush_buffer(&this -> vv_l   ); return;  
        case data_enum::vv_f:     this -> flush_buffer(&this -> vv_f   ); return;  
        case data_enum::vv_i:     this -> flush_buffer(&this -> vv_i   ); return;  
        case data_enum::vv_b:     this -> flush_buffer(&this -> vv_b   ); return;  
        case data_enum::vv_c:     this -> flush_buffer(&this -> vv_c   ); return;  
        
        case data_enum::v_ull:    this -> flush_buffer(&this -> v_ull  ); return;  
        case data_enum::v_ui:     this -> flush_buffer(&this -> v_ui   ); return;  
        case data_enum::v_d:      this -> flush_buffer(&this -> v_d    ); return;  
        case data_enum::v_l:      this -> flush_buffer(&this -> v_l    ); return;  
        case data_enum::v_f:      this -> flush_buffer(&this -> v_f    ); return;  
        case data_enum::v_i:      this -> flush_buffer(&this -> v_i    ); return;  
        case data_enum::v_b:      this -> flush_buffer(&this -> v_b    ); return;  
        case data_enum::v_c:      this -> flush_buffer(&this -> v_c    ); return;  

        case data_enum::ull:      this -> flush_buffer(&this -> ull    ); return; 
        case data_enum::ui:       this -> flush_buffer(&this -> ui     ); return; 
        case data_enum::d:        this -> flush_buffer(&this -> d      ); return; 
        case data_enum::l:        this -> flush_buffer(&this -> l      ); return; 
        case data_enum::f:        this -> flush_buffer(&this -> f      ); return; 
        case data_enum::i:        this -> flush_buffer(&this -> i      ); return; 
        case data_enum::b:        this -> flush_buffer(&this -> b      ); return; 
        case data_enum::c:        this -> flush_buffer(&this -> c      ); return; 
        default: break; 
    }

    if      (this -> vvv_ull){this -> type = data_enum::vvv_ull;}
    else if (this -> vvv_ui ){this -> type = data_enum::vvv_ui; }
    else if (this -> vvv_d  ){this -> type = data_enum::vvv_d;  }
    else if (this -> vvv_l  ){this -> type = data_enum::vvv_l;  }
    else if (this -> vvv_f  ){this -> type = data_enum::vvv_f;  }
    else if (this -> vvv_i  ){this -> type = data_enum::vvv_i;  }
    else if (this -> vvv_b  ){this -> type = data_enum::vvv_b;  }
    else if (this -> vvv_c  ){this -> type = data_enum::vvv_c;  }

    else if (this -> vv_ull ){this -> type = data_enum::vv_ull; }
    else if (this -> vv_ui  ){this -> type = data_enum::vv_ui;  }
    else if (this -> vv_d   ){this -> type = data_enum::vv_d;   }
    else if (this -> vv_l   ){this -> type = data_enum::vv_l;   }
    else if (this -> vv_f   ){this -> type = data_enum::vv_f;   }
    else if (this -> vv_i   ){this -> type = data_enum::vv_i;   }
    else if (this -> vv_b   ){this -> type = data_enum::vv_b;   }
    else if (this -> vv_c   ){this -> type = data_enum::vv_c;   }

    else if (this -> v_ull  ){this -> type = data_enum::v_ull;  }
    else if (this -> v_ui   ){this -> type = data_enum::v_ui;   }
    else if (this -> v_d    ){this -> type = data_enum::v_d;    }
    else if (this -> v_l    ){this -> type = data_enum::v_l;    }
    else if (this -> v_f    ){this -> type = data_enum::v_f;    }
    else if (this -> v_i    ){this -> type = data_enum::v_i;    }
    else if (this -> v_b    ){this -> type = data_enum::v_b;    }
    else if (this -> v_c    ){this -> type = data_enum::v_c;    }

    else if (this -> ull    ){this -> type = data_enum::ull;    }
    else if (this -> ui     ){this -> type = data_enum::ui;     }
    else if (this -> d      ){this -> type = data_enum::d;      }
    else if (this -> l      ){this -> type = data_enum::l;      }
    else if (this -> f      ){this -> type = data_enum::f;      }
    else if (this -> i      ){this -> type = data_enum::i;      }
    else if (this -> b      ){this -> type = data_enum::b;      }
    else if (this -> c      ){this -> type = data_enum::c;      }
    else if (this -> type == data_enum::unset){return;}
    else {this -> type = data_enum::undef;}
    // =================================================================== //

    if (this -> type != data_enum::undef && this -> type != data_enum::unset){return;}
    std::cout << "UNDEFINED DATA TYPE! SEE modules/structs/cxx/base.cxx" << std::endl;
    abort();
}

// -------------- (4). add the data type interace ---------- //

bool bsc_t::element(std::vector<std::vector<std::vector<float>>>* el){
    return this -> _getalt(this -> vvv_f, el); 
}

bool bsc_t::element(std::vector<std::vector<std::vector<double>>>* el){
    return this -> _getalt(this -> vvv_d, el); 
}

bool bsc_t::element(std::vector<std::vector<std::vector<long>>>* el){
    return this -> _getalt(this -> vvv_l, el); 
}

bool bsc_t::element(std::vector<std::vector<std::vector<int>>>* el){
    return this -> _getalt(this -> vvv_i, el); 
}

bool bsc_t::element(std::vector<std::vector<std::vector<bool>>>* el){
    return this -> _getalt(this -> vvv_b, el); 
}


bool bsc_t::element(std::vector<std::vector<float>>* el){
    return this -> _getalt(this -> vvv_f, this -> vv_f, el); 
}

bool bsc_t::element(std::vector<std::vector<double>>* el){
    return this -> _getalt(this -> vvv_d, this -> vv_d, el); 
}

bool bsc_t::element(std::vector<std::vector<long>>* el){
    return this -> _getalt(this -> vvv_l, this -> vv_l, el); 
}

bool bsc_t::element(std::vector<std::vector<int>>* el){
    return this -> _getalt(this -> vvv_i, this -> vv_i, el); 
}

bool bsc_t::element(std::vector<std::vector<bool>>* el){
    return this -> _getalt(this -> vvv_b, this -> vv_b, el); 
}



bool bsc_t::element(std::vector<float>* el){
    return this -> _getalt(this -> vv_f, this -> v_f, el); 
}

bool bsc_t::element(std::vector<double>* el){
    return this -> _getalt(this -> vv_d, this -> v_d, el); 
}

bool bsc_t::element(std::vector<int>* el){
    return this -> _getalt(this -> vv_i, this -> v_i, el); 
}

bool bsc_t::element(std::vector<bool>* el){
    return this -> _getalt(this -> vv_b, this -> v_b, el); 
}

bool bsc_t::element(std::vector<long>* el){
    return this -> _getalt(this -> vv_l, this -> v_l, el); 
}

bool bsc_t::element(std::vector<char>* el){
    return this -> _getalt(this -> vv_c, this -> v_c, el); 
}

bool bsc_t::element(bool* el){
    return this -> _getalt(this -> v_b, this -> b, el); 
}

bool bsc_t::element(double* el){
    return this -> _getalt(this -> v_d, this -> d, el); 
}

bool bsc_t::element(float* el){
    return this -> _getalt(this -> v_f, this -> f, el); 
}

bool bsc_t::element(int* el){
    return this -> _getalt(this -> v_i, this -> i, el); 
}

bool bsc_t::element(long* el){
    return this -> _getalt(this -> v_l, this -> l, el); 
}

bool bsc_t::element(unsigned long long* el){
    return this -> _getalt(this -> v_ull, this -> ull, el); 
}

bool bsc_t::element(unsigned int* el){
    return this -> _getalt(this -> v_ui, this -> ui, el); 
}

bool bsc_t::element(char* el){
    return this -> _getalt(this -> v_c, this -> c, el); 
}

// ******************************************************************************************* //


