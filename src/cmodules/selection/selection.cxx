#include "../selection/selection.h"

namespace CyTemplate
{
    CySelectionTemplate::CySelectionTemplate(){}
    CySelectionTemplate::~CySelectionTemplate(){}
    void CySelectionTemplate::Import(selection_t sel)
    {
        this -> selection = sel; 
        this -> selection.selection = true; 
    }; 
}
