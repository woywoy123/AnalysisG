#include "../Headers/Event.h"

CyTemplate::CyEvent::CyEvent(){}
CyTemplate::CyEvent::~CyEvent(){}

std::string CyTemplate::CyEvent::Hash(){ return this -> _hash; }
void CyTemplate::CyEvent::Hash(std::string inpt)
{
    if (this -> _hash != ""){return;}
    if (inpt.size() == 18){ this -> _hash = inpt; return; }
    inpt = inpt + "/" + Tools::ToString(this -> index) + "/" + (this -> tree); 
    this -> _hash = Tools::Hashing(inpt); 
}
