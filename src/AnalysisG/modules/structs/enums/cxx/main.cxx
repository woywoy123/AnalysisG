#include <structs/enums.h>
#include <tools/tools.h>

std::string AnalysisG::enums::as_string(AnalysisG::enums::graph gr){
    switch(gr){
        case AnalysisG::enums::graph::truth_graph:  return "truth::graph::"; 
        case AnalysisG::enums::graph::truth_node :  return "truth::node::"; 
        case AnalysisG::enums::graph::truth_edge :  return "truth::edge::"; 
        case AnalysisG::enums::graph::edge_index  : return "data::edge::"; 

        case AnalysisG::enums::graph::pred_graph: return "prediction::graph::"; 
        case AnalysisG::enums::graph::pred_node : return "prediction::node::" ; 
        case AnalysisG::enums::graph::pred_edge : return "prediction::edge::" ; 
        case AnalysisG::enums::graph::pred_extra: return "prediction::extra::"; 

        case AnalysisG::enums::graph::data_graph: return "data::graph::"; 
        case AnalysisG::enums::graph::data_node : return "data::node::" ; 
        case AnalysisG::enums::graph::data_edge : return "data::edge::" ; 

        case AnalysisG::enums::graph::batch_index : return "data::graph::"; 
        case AnalysisG::enums::graph::batch_events: return "data::graph::"; 
        case AnalysisG::enums::graph::weight      : return "data::graph::"; 

        case AnalysisG::enums::graph::extra_graph: return "extra::graph::"; 
        case AnalysisG::enums::graph::extra_node : return "extra::node::" ; 
        case AnalysisG::enums::graph::extra_edge : return "extra::edge::" ; 

        default: return "invalid"; 
    }
}

std::string AnalysisG::enums::as_string(AnalysisG::enums::mode st){
    switch (st){
        case AnalysisG::enums::mode::training:   return "Training";
        case AnalysisG::enums::mode::validation: return "Validation";
        case AnalysisG::enums::mode::evaluation: return "Evaluation";
        default: return "Invalid";
    }
    return "Invalid"; 
}

AnalysisG::enums::mode AnalysisG::enums::as_mode(const std::string* val){
    std::string vl = AnalysisG::tooling::lower(val); 
    if (vl == "training"  ){return AnalysisG::enums::mode::training;}
    if (vl == "validation"){return AnalysisG::enums::mode::validation;}
    if (vl == "evaluation"){return AnalysisG::enums::mode::evaluation;} 
    return AnalysisG::enums::mode::invalid; 
}

std::map<AnalysisG::enums::mode, std::string> AnalysisG::enums::as_mode(const std::map<std::string, std::string>* val){
    std::map<AnalysisG::enums::mode, std::string> out;
    std::map<std::string, std::string>::const_iterator it = val -> begin(); 
    for (; it != val -> end(); ++it){out[AnalysisG::enums::as_mode(&it -> first)] = it -> second;}
    return out; 
}

AnalysisG::enums::graph as_graph(const std::string* vl){
    auto lambda =[](bool trg1, bool trg2, bool t1, bool t2, bool t3) -> bool {
        return (trg1 && trg2) && !( t1 + t2 + t3 ); 
    }; 

    std::string lw = AnalysisG::tooling::lower(vl);  
    bool edx = AnalysisG::tooling::has_string(&lw, "edge" ); 
    bool idx = AnalysisG::tooling::has_string(&lw, "index"); 

    bool dta = AnalysisG::tooling::has_string(&lw, "data"      );
    bool tru = AnalysisG::tooling::has_string(&lw, "truth"     );
    bool etc = AnalysisG::tooling::has_string(&lw, "extra"     ); 
    bool prd = AnalysisG::tooling::has_string(&lw, "prediction");
    bool btx = AnalysisG::tooling::has_string(&lw, "batch"     );

    if (lambda(edx, idx, !prd, !(etc + dta), !tru)){return AnalysisG::enums::graph::edge_index;}
    if (btx && AnalysisG::tooling::has_string(&lw, "event"    )){return AnalysisG::enums::graph::batch_events;}
    if (btx && idx){return AnalysisG::enums::graph::batch_index;}
    if (AnalysisG::tooling::has_string(&lw, "weight"          )){return AnalysisG::enums::graph::weight;}

 
    bool gdx = AnalysisG::tooling::has_string(&lw, "graph"     ); 
    bool ndx = AnalysisG::tooling::has_string(&lw, "node"      ); 

    if (lambda(tru, gdx,  prd,  etc, dta)){ return AnalysisG::enums::graph::truth_graph; }
    if (lambda(tru, ndx,  prd,  etc, dta)){ return AnalysisG::enums::graph::truth_node;  }
    if (lambda(tru, edx,  prd,  etc, dta)){ return AnalysisG::enums::graph::truth_edge;  }

    if (lambda(prd, gdx, etc, dta, tru)){ return AnalysisG::enums::graph::pred_graph; }
    if (lambda(prd, ndx, etc, dta, tru)){ return AnalysisG::enums::graph::pred_node;  }
    if (lambda(prd, edx, etc, dta, tru)){ return AnalysisG::enums::graph::pred_edge;  }

    if (lambda(dta, gdx, tru, prd, etc)){ return AnalysisG::enums::graph::data_graph; }
    if (lambda(dta, ndx, tru, prd, etc)){ return AnalysisG::enums::graph::data_node;  }
    if (lambda(dta, edx, tru, prd, etc)){ return AnalysisG::enums::graph::data_edge;  }

    // covers literal: extra + <graph, node, edge> cases.
    if (lambda(etc, gdx, dta, tru, prd)){ return AnalysisG::enums::graph::extra_graph; }
    if (lambda(etc, ndx, dta, tru, prd)){ return AnalysisG::enums::graph::extra_node;  }
    if (lambda(etc, edx, dta, tru, prd)){ return AnalysisG::enums::graph::extra_edge;  }

    // covers literal: extra + prediction <graph, node, edge> cases.
    if (lambda((etc && prd),  gdx, dta, tru, !prd)){ return AnalysisG::enums::graph::extra_graph; }
    if (lambda((etc && prd),  ndx, dta, tru, !prd)){ return AnalysisG::enums::graph::extra_node;  }
    if (lambda((etc && prd),  edx, dta, tru, !prd)){ return AnalysisG::enums::graph::extra_edge;  }
    if (lambda((etc && prd), !tru, gdx, ndx, edx )){ return AnalysisG::enums::graph::pred_extra; }
    return AnalysisG::enums::graph::invalid;
}

// -------------------- (3). add the routing -------------- //
AnalysisG::enums::data AnalysisG::enums::as_data(const std::string* root_str){
    auto lambda = [](int vx, const std::string* r_st, std::string trg) -> bool {
        return vx && AnalysisG::tooling::count(r_st, trg); 
    }; 

    std::string r_str = *root_str; 
    int vec = AnalysisG::tooling::count(&r_str, "vector"); 
    if (vec == 0 && r_str ==   "Float_t"){return AnalysisG::enums::data::v_f  ;}
    if (vec == 0 && r_str ==  "Double_t"){return AnalysisG::enums::data::v_d  ;}
    if (vec == 0 && r_str ==    "UInt_t"){return AnalysisG::enums::data::v_ui ;}
    if (vec == 0 && r_str ==     "Int_t"){return AnalysisG::enums::data::v_i  ;}
    if (vec == 0 && r_str ==    "Char_t"){return AnalysisG::enums::data::v_c  ;}
    if (vec == 0 && r_str ==      "char"){return AnalysisG::enums::data::v_c  ;}
    if (vec == 0 && r_str == "ULong64_t"){return AnalysisG::enums::data::v_ull;}
  
    if (lambda(0, root_str, "float")){return AnalysisG::enums::data::v_f;}
    if (lambda(1, root_str, "float")){return AnalysisG::enums::data::vv_f;}
    if (lambda(2, root_str, "float")){return AnalysisG::enums::data::vvv_f;}

    if (lambda(0, root_str, "double")){return AnalysisG::enums::data::v_d;}
    if (lambda(1, root_str, "double")){return AnalysisG::enums::data::vv_d;}
    if (lambda(2, root_str, "double")){return AnalysisG::enums::data::vvv_d;}

    if (lambda(0, root_str, "int")){return AnalysisG::enums::data::v_i;}
    if (lambda(1, root_str, "int")){return AnalysisG::enums::data::vv_i;}
    if (lambda(2, root_str, "int")){return AnalysisG::enums::data::vvv_i;}

    if (lambda(0, root_str, "long")){return AnalysisG::enums::data::v_l;}
    if (lambda(1, root_str, "long")){return AnalysisG::enums::data::vv_l;}
    if (lambda(2, root_str, "long")){return AnalysisG::enums::data::vvv_l;}

    if (lambda(0, root_str, "bool")){return AnalysisG::enums::data::v_b;}
    if (lambda(1, root_str, "bool")){return AnalysisG::enums::data::vv_b;}
    if (lambda(2, root_str, "bool")){return AnalysisG::enums::data::vvv_b;}

    if (lambda(1, root_str, "char")){return AnalysisG::enums::data::vv_c;}
    return AnalysisG::enums::data::undefined; 
}

std::string AnalysisG::enums::as_string(AnalysisG::enums::data rt){
    switch (rt){
        case AnalysisG::enums::data::vvv_ull:       return "vector<vector<vector<unsigned long long>>>";  
        case AnalysisG::enums::data::vvv_ui:        return "vector<vector<vector<unsigned int>>>";        
        case AnalysisG::enums::data::vvv_d:         return "vector<vector<vector<double>>>";              
        case AnalysisG::enums::data::vvv_l:         return "vector<vector<vector<long>>>";                
        case AnalysisG::enums::data::vvv_f:         return "vector<vector<vector<float>>>";               
        case AnalysisG::enums::data::vvv_i:         return "vector<vector<vector<int>>>" ;                
        case AnalysisG::enums::data::vvv_b:         return "vector<vector<vector<bool>>>";                
        case AnalysisG::enums::data::vvv_c:         return "vector<vector<vector<char>>>";                
        case AnalysisG::enums::data::vv_ull:        return "vector<vector<unsigned long long>>";          
        case AnalysisG::enums::data::vv_ui:         return "vector<vector<unsigned int>>";                
        case AnalysisG::enums::data::vv_d:          return "vector<vector<double>>";                      
        case AnalysisG::enums::data::vv_l:          return "vector<vector<long>>";                        
        case AnalysisG::enums::data::vv_f:          return "vector<vector<float>>";                       
        case AnalysisG::enums::data::vv_i:          return "vector<vector<int>>" ;                        
        case AnalysisG::enums::data::vv_b:          return "vector<vector<bool>>";                        
        case AnalysisG::enums::data::vv_c:          return "vector<vector<char>>";                        
        case AnalysisG::enums::data::v_ull:         return "vector<unsigned long long>";                  
        case AnalysisG::enums::data::v_ui:          return "vector<unsigned int>";                        
        case AnalysisG::enums::data::v_d:           return "vector<double>";                              
        case AnalysisG::enums::data::v_l:           return "vector<long>";                                
        case AnalysisG::enums::data::v_f:           return "vector<float>";                               
        case AnalysisG::enums::data::v_i:           return "vector<int>" ;                                
        case AnalysisG::enums::data::v_b:           return "vector<bool>";                                
        case AnalysisG::enums::data::v_c:           return "vector<char>";                                
        case AnalysisG::enums::data::ull:           return "unsigned long long";                          
        case AnalysisG::enums::data::ui:            return "unsigned int";                                
        case AnalysisG::enums::data::d:             return "double";                                      
        case AnalysisG::enums::data::l:             return "long";                                        
        case AnalysisG::enums::data::f:             return "float";                                       
        case AnalysisG::enums::data::i:             return "int" ;                                        
        case AnalysisG::enums::data::b:             return "bool";                                        
        case AnalysisG::enums::data::c:             return "char";                                        
        case AnalysisG::enums::data::unset:         return "unset";                                       
        case AnalysisG::enums::data::undefined:     return "undefined"; 
        default:                                    return "undefined";                                                   
    }
    return "undefined"; 
}




//std::string bsc_t::scan_buffer(){
//    std::string x = ""; 
//    if (this -> vvv_ull){x += " | vector<vector<vector<unsigned long long>>>";}
//    if (this -> vvv_ui ){x += " | vector<vector<vector<unsigned int>>>";      }
//    if (this -> vvv_d  ){x += " | vector<vector<vector<double>>>";            }
//    if (this -> vvv_l  ){x += " | vector<vector<vector<long>>>";              }
//    if (this -> vvv_f  ){x += " | vector<vector<vector<float>>>";             }
//    if (this -> vvv_i  ){x += " | vector<vector<vector<int>>>" ;              }
//    if (this -> vvv_b  ){x += " | vector<vector<vector<bool>>>";              }
//    if (this -> vvv_c  ){x += " | vector<vector<vector<char>>>";              }
//    if (this -> vv_ull ){x += " | vector<vector<unsigned long long>>";        }
//    if (this -> vv_ui  ){x += " | vector<vector<unsigned int>>";              }
//    if (this -> vv_d   ){x += " | vector<vector<double>>";                    }
//    if (this -> vv_l   ){x += " | vector<vector<long>>";                      }
//    if (this -> vv_f   ){x += " | vector<vector<float>>";                     }
//    if (this -> vv_i   ){x += " | vector<vector<int>>" ;                      }
//    if (this -> vv_b   ){x += " | vector<vector<bool>>";                      }
//    if (this -> vv_c   ){x += " | vector<vector<char>>";                      }
//    if (this -> v_ull  ){x += " | vector<unsigned long long>";                }
//    if (this -> v_ui   ){x += " | vector<unsigned int>";                      }
//    if (this -> v_d    ){x += " | vector<double>";                            }
//    if (this -> v_l    ){x += " | vector<long>";                              }
//    if (this -> v_f    ){x += " | vector<float>";                             }
//    if (this -> v_i    ){x += " | vector<int>" ;                              }
//    if (this -> v_b    ){x += " | vector<bool>";                              }
//    if (this -> v_c    ){x += " | vector<char>";                              }
//    if (this -> ull    ){x += " | unsigned long long";                        }
//    if (this -> ui     ){x += " | unsigned int";                              }
//    if (this -> d      ){x += " | double";                                    }
//    if (this -> l      ){x += " | long";                                      }
//    if (this -> f      ){x += " | float";                                     }
//    if (this -> i      ){x += " | int" ;                                      }
//    if (this -> b      ){x += " | bool";                                      }
//    if (this -> c      ){x += " | char";                                      }
//    return (x.size()) ? x : "undefined/unset"; 
//}                                                         
// 
