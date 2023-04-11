#include "../Headers/Event.h"
#include "../../Tools/Headers/Tools.h"

Sample::Event::Event(){} 
Sample::Event::~Event(){}
void Sample::Event::MakeHash()
{
    std::string name = this -> Hash; 
    name += "/"; 
    name += std::to_string(this -> EventIndex); 
    this -> Hash = Tools::Hashing(name); 
}
