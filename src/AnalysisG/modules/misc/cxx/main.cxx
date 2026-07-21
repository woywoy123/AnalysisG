#include <TInterpreter.h>
#include <misc/pcm.h>
#include <iostream>

void AnalysisG::misc::buildDict(std::string _name, std::string _shrt){
    std::string name = std::string(_name);
    gInterpreter -> GenerateDictionary(name.c_str(), _shrt.c_str()); 
}

void AnalysisG::misc::registerInclude(std::string pth, bool is_abs){
    std::string shrt = "#include ";
    if (is_abs){shrt += " \"" + pth + "\"";}
    else {shrt += "<" + pth + ">";}
    gInterpreter -> ProcessLine(shrt.c_str());
}

void AnalysisG::misc::buildPCM(AnalysisG::enums::data tr, std::string incl, bool exl){
    if (exl){return;}
    buildDict(AnalysisG::enums::as_string(tr), incl); 
}

// ============================= Add your type (2) =================================== //
void AnalysisG::misc::buildAll(){
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_ull, "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_ui , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_d  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_l  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_f  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_i  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_b  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vvv_c  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_ull , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_ui  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_d   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_l   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_f   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_i   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_b   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::vv_c   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_ull  , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_ui   , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_d    , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_l    , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_f    , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_i    , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_b    , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::v_c    , "vector", false); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::ull    , ""      , true ); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::ui     , ""      , true ); 
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::d      , ""      , true );      
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::l      , ""      , true );        
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::f      , ""      , true );       
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::i      , ""      , true );        
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::b      , ""      , true );        
    AnalysisG::misc::buildPCM(AnalysisG::enums::data::c      , ""      , true );        
} 
