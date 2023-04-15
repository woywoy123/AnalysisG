#include "../Headers/Event.h"

CyTemplate::CyEvent::CyEvent(){}
CyTemplate::CyEvent::~CyEvent(){}

std::string CyTemplate::CyEvent::Hash(){ return this -> _hash; }
void CyTemplate::CyEvent::Hash(std::string inpt)
{
    inpt = inpt + "/" + Tools::ToString(this -> index) + "/" + (this -> tree); 
    this -> _hash = Tools::Hashing(inpt); 
}
