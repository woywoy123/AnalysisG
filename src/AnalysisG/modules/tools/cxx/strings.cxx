#include <tools/tools.h>
#include <sstream>
#include <string>

void tools::replace(std::string* in, std::string to_repl, std::string repl_w) {
    std::size_t pos = 0; 
    std::size_t ppos; 
    std::string buf; 

    while (true){
        ppos = pos;
        pos = in -> find(to_repl, pos); 
        if (pos == std::string::npos){break;}
        buf.append(*in, ppos, pos - ppos); 
        buf += repl_w; 
        pos += to_repl.size(); 
    }
    
    buf.append(*in, ppos, in -> size() - ppos); 
    in -> swap(buf); 
}

std::vector<std::string> tools::split(std::string inpt, std::string search) {
    size_t pos = 0;
    size_t s_dim = search.length();
    size_t index = 0;
    std::string token;
    std::vector<std::string> out = {};
    while ((pos = inpt.find(search)) != std::string::npos){
        out.push_back(inpt.substr(0, pos));
        inpt.erase(0, pos + s_dim);
        ++index;
    }
    out.push_back(inpt);
    return out;
}

std::vector<std::string> tools::split(std::string in, int n){
    int x = 0; 
    std::vector<std::string> out = {""}; 
    for (int i(0); i < in.size(); ++i){
        if (out[x].size() < n){}
        else {
            out.push_back(""); 
            ++x; 
        }
        out[x] += in[i]; 
    }
    return out; 
}

std::string tools::hash(std::string input, int len) {
    std::hash<std::string> hasher; 
    std::stringstream ss; 
    ss << "0x" << std::hex << hasher(input); 
    std::string out = ss.str(); 
    int diff = out.size() - len; 
    if (!diff) { return out; }
    out += std::string(std::abs(diff), '0'); 
    return out; 
}

std::string tools::to_string(double val){
    std::stringstream ss; 
    ss << val; 
    return ss.str(); 
}

bool tools::has_string(std::string* inpt, std::string trg){
    std::size_t f = inpt -> find(trg); 
    if (f != std::string::npos){return true;}
    return false; 
}

bool tools::ends_with(std::string* inpt, std::string val){
    if (inpt -> size() < val.size()){return false;}
    std::string l = inpt -> substr(inpt -> size() - val.size(), inpt -> size()-1); 
    return val == l; 
}

bool tools::has_value(std::vector<std::string>* data, std::string trg){
    for (int x(0); x < data -> size(); ++x){
        if (trg != data -> at(x)){continue;}
        return true; 
    }
    return false; 
}

std::string tools::lower(std::string* in){
    std::string out = *in;
    for (int t(0); t < in -> size(); ++t){out[t] = std::tolower(out[t]);}
    return out;
}

std::string tools::to_string(double val, int prec){
    std::ostringstream ss; 
    if (prec > -1){ss.precision(prec);}
    ss << std::fixed << val; 
    return std::move(ss).str(); 
}



static const std::string base64_chars = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

bool is_base64(unsigned char c){return (isalnum(c) || (c == '+') || (c == '/'));}

std::string tools::encode64(unsigned char const* bytes_to_encode, unsigned int in_len)
{
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    
    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3){
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i <4) ; i++){ret += base64_chars[char_array_4[i]];}
            i = 0;
        }
    }

    if (i){
        for(j = i; j < 3; j++){char_array_3[j] = '\0';}
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; (j < i + 1); j++){ret += base64_chars[char_array_4[j]];}
        while((i++ < 3)){ret += '=';}
    }
    return ret;
}

 

std::string tools::decode64(std::string* inpt){
    return tools::decode64(*inpt); 
}


std::string tools::decode64(std::string const& encoded_string){
    size_t in_len = encoded_string.size();
    size_t i = 0;
    size_t j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++){char_array_4[i] = static_cast<unsigned char>(base64_chars.find(char_array_4[i]));}
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++){ret += char_array_3[i];}
            i = 0;
        }
    }
    
    if (i) {
        for (j = i; j <4; j++){char_array_4[j] = 0;}
        for (j = 0; j <4; j++){char_array_4[j] = static_cast<unsigned char>(base64_chars.find(char_array_4[j]));}
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }
    return ret;
}

