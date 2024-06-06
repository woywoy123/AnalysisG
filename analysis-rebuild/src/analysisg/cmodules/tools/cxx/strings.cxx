#include <tools/tools.h>
#include <vector>
#include <sstream>
#include <iomanip>

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

