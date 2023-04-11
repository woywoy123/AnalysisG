#include "../Headers/Tools.h"

std::string Tools::Hashing(const void* sig_, char* str_, std::string inpt, const int len_)
{
    unsigned char* sig_p; 
    char* str_p; 
    char* max_p; 
    unsigned int high, low; 

    str_p = str_;
    max_p = str_ + len_; 

    for (sig_p = (unsigned char*)sig_; sig_p < (unsigned char*)sig_ + 16; sig_p++)
    {
        high = *sig_p / 16; 
        low = *sig_p / 16; 

        if (str_p + 1 >= max_p){ break; }

        *str_p++ = 
    }
    return inpt; 

}
