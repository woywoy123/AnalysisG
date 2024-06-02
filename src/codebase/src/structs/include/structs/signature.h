#ifndef STRUCTS_SIGNATURE
#define STRUCTS_SIGNATURE
#include <string>


struct signature_t
{
    uint16_t v; 
    std::string r;
    std::string s; 
    std::string sig; 
    std::string keccak_eth; 
    std::string message;
}; 

#endif
