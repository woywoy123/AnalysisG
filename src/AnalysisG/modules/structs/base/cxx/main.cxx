#include <structs/base.h>
#include <tools/tools.h>

AnalysisG::core::element_t::element_t(){}
AnalysisG::core::element_t::~element_t(){this -> _flush_buffer();}

// ------------ (4.) Add the buffer flush -------------------- //
void AnalysisG::core::element_t::_flush_buffer(){
//    switch (this -> type){
//        case AnalysisG::enums::data::vvv_ull:  this -> _flush_buffer(&this -> vvv_ull); return;  
//        case AnalysisG::enums::data::vvv_ui:   this -> _flush_buffer(&this -> vvv_ui ); return;  
//        case AnalysisG::enums::data::vvv_d:    this -> _flush_buffer(&this -> vvv_d  ); return;  
//        case AnalysisG::enums::data::vvv_l:    this -> _flush_buffer(&this -> vvv_l  ); return;  
//        case AnalysisG::enums::data::vvv_f:    this -> _flush_buffer(&this -> vvv_f  ); return;  
//        case AnalysisG::enums::data::vvv_i:    this -> _flush_buffer(&this -> vvv_i  ); return;  
//        case AnalysisG::enums::data::vvv_b:    this -> _flush_buffer(&this -> vvv_b  ); return;  
//        case AnalysisG::enums::data::vvv_c:    this -> _flush_buffer(&this -> vvv_c  ); return;  
//        
//        case AnalysisG::enums::data::vv_ull:   this -> _flush_buffer(&this -> vv_ull ); return;  
//        case AnalysisG::enums::data::vv_ui:    this -> _flush_buffer(&this -> vv_ui  ); return;  
//        case AnalysisG::enums::data::vv_d:     this -> _flush_buffer(&this -> vv_d   ); return;  
//        case AnalysisG::enums::data::vv_l:     this -> _flush_buffer(&this -> vv_l   ); return;  
//        case AnalysisG::enums::data::vv_f:     this -> _flush_buffer(&this -> vv_f   ); return;  
//        case AnalysisG::enums::data::vv_i:     this -> _flush_buffer(&this -> vv_i   ); return;  
//        case AnalysisG::enums::data::vv_b:     this -> _flush_buffer(&this -> vv_b   ); return;  
//        case AnalysisG::enums::data::vv_c:     this -> _flush_buffer(&this -> vv_c   ); return;  
//        
//        case AnalysisG::enums::data::v_ull:    this -> _flush_buffer(&this -> v_ull  ); return;  
//        case AnalysisG::enums::data::v_ui:     this -> _flush_buffer(&this -> v_ui   ); return;  
//        case AnalysisG::enums::data::v_d:      this -> _flush_buffer(&this -> v_d    ); return;  
//        case AnalysisG::enums::data::v_l:      this -> _flush_buffer(&this -> v_l    ); return;  
//        case AnalysisG::enums::data::v_f:      this -> _flush_buffer(&this -> v_f    ); return;  
//        case AnalysisG::enums::data::v_i:      this -> _flush_buffer(&this -> v_i    ); return;  
//        case AnalysisG::enums::data::v_b:      this -> _flush_buffer(&this -> v_b    ); return;  
//        case AnalysisG::enums::data::v_c:      this -> _flush_buffer(&this -> v_c    ); return;  
//
//        case AnalysisG::enums::data::ull:      this -> _flush_buffer(&this -> ull    ); return; 
//        case AnalysisG::enums::data::ui:       this -> _flush_buffer(&this -> ui     ); return; 
//        case AnalysisG::enums::data::d:        this -> _flush_buffer(&this -> d      ); return; 
//        case AnalysisG::enums::data::l:        this -> _flush_buffer(&this -> l      ); return; 
//        case AnalysisG::enums::data::f:        this -> _flush_buffer(&this -> f      ); return; 
//        case AnalysisG::enums::data::i:        this -> _flush_buffer(&this -> i      ); return; 
//        case AnalysisG::enums::data::b:        this -> _flush_buffer(&this -> b      ); return; 
//        case AnalysisG::enums::data::c:        this -> _flush_buffer(&this -> c      ); return; 
//
//        default: break; 
//    }

//    if      (this -> vvv_ull){this -> type = AnalysisG::enums::data::vvv_ull;}
//    else if (this -> vvv_ui ){this -> type = AnalysisG::enums::data::vvv_ui; }
//    else if (this -> vvv_d  ){this -> type = AnalysisG::enums::data::vvv_d;  }
//    else if (this -> vvv_l  ){this -> type = AnalysisG::enums::data::vvv_l;  }
//    else if (this -> vvv_f  ){this -> type = AnalysisG::enums::data::vvv_f;  }
//    else if (this -> vvv_i  ){this -> type = AnalysisG::enums::data::vvv_i;  }
//    else if (this -> vvv_b  ){this -> type = AnalysisG::enums::data::vvv_b;  }
//    else if (this -> vvv_c  ){this -> type = AnalysisG::enums::data::vvv_c;  }
//
//    else if (this -> vv_ull ){this -> type = AnalysisG::enums::data::vv_ull; }
//    else if (this -> vv_ui  ){this -> type = AnalysisG::enums::data::vv_ui;  }
//    else if (this -> vv_d   ){this -> type = AnalysisG::enums::data::vv_d;   }
//    else if (this -> vv_l   ){this -> type = AnalysisG::enums::data::vv_l;   }
//    else if (this -> vv_f   ){this -> type = AnalysisG::enums::data::vv_f;   }
//    else if (this -> vv_i   ){this -> type = AnalysisG::enums::data::vv_i;   }
//    else if (this -> vv_b   ){this -> type = AnalysisG::enums::data::vv_b;   }
//    else if (this -> vv_c   ){this -> type = AnalysisG::enums::data::vv_c;   }
//
//    else if (this -> v_ull  ){this -> type = AnalysisG::enums::data::v_ull;  }
//    else if (this -> v_ui   ){this -> type = AnalysisG::enums::data::v_ui;   }
//    else if (this -> v_d    ){this -> type = AnalysisG::enums::data::v_d;    }
//    else if (this -> v_l    ){this -> type = AnalysisG::enums::data::v_l;    }
//    else if (this -> v_f    ){this -> type = AnalysisG::enums::data::v_f;    }
//    else if (this -> v_i    ){this -> type = AnalysisG::enums::data::v_i;    }
//    else if (this -> v_b    ){this -> type = AnalysisG::enums::data::v_b;    }
//    else if (this -> v_c    ){this -> type = AnalysisG::enums::data::v_c;    }
//
//    else if (this -> ull    ){this -> type = AnalysisG::enums::data::ull;    }
//    else if (this -> ui     ){this -> type = AnalysisG::enums::data::ui;     }
//    else if (this -> d      ){this -> type = AnalysisG::enums::data::d;      }
//    else if (this -> l      ){this -> type = AnalysisG::enums::data::l;      }
//    else if (this -> f      ){this -> type = AnalysisG::enums::data::f;      }
//    else if (this -> i      ){this -> type = AnalysisG::enums::data::i;      }
//    else if (this -> b      ){this -> type = AnalysisG::enums::data::b;      }
//    else if (this -> c      ){this -> type = AnalysisG::enums::data::c;      }
//
//    else if (this -> type == AnalysisG::enums::data::unset){return;}
//    else    {this -> type = AnalysisG::enums::data::undefined;}
//    // =================================================================== //
//
//    if (this -> type != AnalysisG::enums::data::undefined && this -> type != AnalysisG::enums::data::unset){return;}
    std::cout << "UNDEFINED DATA TYPE! SEE modules/structs/cxx/base.cxx" << std::endl;
    abort();
}

