#include "../selection/selection.h"

namespace CyTemplate
{
    CySelectionTemplate::CySelectionTemplate()
    {
        this -> sel = &(this -> selection);  
    }

    CySelectionTemplate::~CySelectionTemplate(){}

    bool CySelectionTemplate::operator == (CySelectionTemplate& selection)
    {
        selection_t* gr2 = &(selection.selection); 
        return this -> is_same(this -> sel, gr2);
    }

    void CySelectionTemplate::Import(selection_t sel)
    {
        this -> selection = sel; 
        this -> selection.selection = true; 
        this -> is_selection = true; 
    }; 

    void CySelectionTemplate::RegisterEvent(const event_t* evnt)
    {
        selection_t* sel = &(this -> selection); 
        this -> set_event_hash(sel, evnt); 
        this -> set_event_tag(sel, evnt); 
        this -> set_event_tree(sel, evnt); 
        this -> set_event_root(sel, evnt);
        this -> set_event_index(sel, evnt);
        this -> set_event_weight(sel, evnt); 
        this -> current_weight = evnt -> weight; 
        sel -> all_weights.push_back(evnt -> weight); 
    }
    
    bool CySelectionTemplate::CheckSelection(bool passed)
    {
        if (passed){ this -> sel -> cutflow["Selection::Passed"] += 1; }
        else { this -> sel -> cutflow["Selection::Rejected"] += 1; }
        return passed; 
    }

    bool CySelectionTemplate::CheckSelection(std::string passed)
    {
        int pass; 
        passed = "Selection::" + passed; 
        pass = Tools::count(passed, "::Passed"); 
        if (pass){ this -> sel -> cutflow[passed] += 1; return true;}
        
        pass = Tools::count(passed, "::Rejected"); 
        if (pass){ this -> sel -> cutflow[passed] += 1; return false;}
        
        pass = Tools::count(passed, "::Error"); 
        if (pass){ this -> sel -> cutflow[passed] += 1; return false;}

        this -> sel -> cutflow[passed + "::Ambiguous"] +=1; 
        return true;
    }

    bool CySelectionTemplate::CheckStrategy(bool passed)
    {
        if (passed){ this -> sel -> cutflow["Strategy::Passed"] += 1; }
        else { this -> sel -> cutflow["Strategy::Rejected"] += 1; }
        return passed; 
    }

    bool CySelectionTemplate::CheckStrategy(std::string passed)
    {
        int pass; 
        passed = "Strategy::" + passed; 
        pass = Tools::count(passed, "::Passed"); 
        if (pass){ 
            this -> sel -> cutflow[passed] += 1; 
            this -> sel -> selection_weights.push_back(this -> current_weight);
            return true;
        }
        
        pass = Tools::count(passed, "::Rejected"); 
        if (pass){ this -> sel -> cutflow[passed] += 1; return false;}
        
        pass = Tools::count(passed, "::Error"); 
        if (pass){ this -> sel -> cutflow[passed] += 1; return false;}

        this -> sel -> cutflow[passed + "::Ambiguous"] +=1; 
        return true;
    }

    void CySelectionTemplate::StartTime()
    {
        this -> ts = std::chrono::high_resolution_clock::now(); 
    }

    void CySelectionTemplate::EndTime()
    {        
        this -> te = std::chrono::high_resolution_clock::now(); 
        double tf = std::chrono::duration_cast<std::chrono::nanoseconds>(this -> te - this -> ts).count(); 
        this -> sel -> timestats.push_back(tf); 
    }

    double CySelectionTemplate::Mean()
    {
        double out = 0; 
        for (double i : this -> sel -> timestats){out += i;}
        return out; 
    }    

    double CySelectionTemplate::StandardDeviation()
    {
        double mean = this -> Mean(); 
        double out = 0; 
        for (double i : this -> sel -> timestats){ out += std::pow(i - mean, 2); }
        mean = this -> sel -> timestats.size(); 
        if (!mean){return -1;}
        return std::pow(out / mean, 0.5); 
    }

    double CySelectionTemplate::Luminosity()
    {
        double sel_sum = 0; 
        double all_sum = 0; 
        for (double i : this -> sel -> selection_weights){sel_sum += i;}
        for (double i : this -> sel -> all_weights){all_sum += i;}
        if (!all_sum){return -1;}
        return sel_sum/all_sum; 
    }


}
