#include <sstream>
#include <cstring>
#include <iomanip>
#include <tools.h>

std::vector<uint8_t> tools::to_uint8(std::string in){
    std::vector<uint8_t> out(in.begin(), in.end());
    return out; 
}

std::vector<uint8_t> tools::to_uint8(std::string* in)
{
    int offset = 0; 
    unsigned int buf; 
    std::stringstream ss; 
    std::vector<uint8_t> hex; 
    while (offset < in -> length()){
        ss.clear(); 
        ss << std::hex << in -> substr(offset, 2);
        ss >> buf;

        uint16_t t = static_cast<uint16_t>(buf); 
        uint8_t array[sizeof(t)]; 
        memcpy(array, &t, sizeof(array)); 
        hex.push_back(array[0]); 
        offset += 2; 
    }
    return hex; 
}

void tools::to_uint8(std::string* in, uint8_t* out)
{
    unsigned int i = 0;
    std::stringstream ss; 
    std::string::iterator itr; 
    for (itr = in -> begin(); itr != in -> end();)
    {
        ss << "0x"; 
        ss << *itr; ++itr; 
        ss << *itr; ++itr; 
        const char* t = ss.str().c_str(); 
        out[i] = std::strtol(t, nullptr, 16); 
        ss.str(""); 
        ++i;
    }
}

void tools::to_hex(uint8_t* in, unsigned int len, std::string* out)
{
    std::stringstream ss; 
    for (unsigned int i(0); i < len; ++i){
        ss << std::hex << std::setfill('0') << std::setw(2) << (int)in[i];
    }
    *out = ss.str(); 
}

std::string tools::to_hex(std::string* in)
{
    std::stringstream ss; 
    for (uint8_t x : *in){ss << std::hex << std::setfill('0') << std::setw(2) << (int)x; }
    return ss.str();
}

std::string tools::hex_to_string(std::string* in){
    std::string out;
    int len = in -> length(); 
    for (int x(0); x < len; x += 2){
        std::string byte = in -> substr(x, 2); 
        char ch = (char)(int)strtol(byte.c_str(), NULL, 16); 
        out.push_back(ch);  
    }
    return out; 
}

std::string tools::to_string(double qnt, int prec){
    std::ostringstream ss; 
    if (prec > -1){ss.precision(prec);}
    ss << std::fixed << qnt; 
    return std::move(ss).str(); 
}

std::string tools::remove_trailing(std::string* inpt){
    std::string out = ""; 
    for (int x(inpt -> size()-1); x > 0; --x){
        if (inpt -> at(x) == '0'){continue;}
        if (inpt -> at(x) == '.'){return inpt -> substr(0, x);}
        return inpt -> substr(0, x+1); 
    }
    return out; 
}


