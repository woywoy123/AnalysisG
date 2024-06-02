#include <tools.h>
#include <vector>
#include <sstream>
#include <iomanip>

void tools::replace(std::string* in, std::string to_repl, std::string repl_w)
{
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

std::vector<std::string> tools::split(std::string inpt, std::string search)
{
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

std::string tools::hash(std::string input, int len)
{
    std::hash<std::string> hasher; 
    std::stringstream ss; 
    ss << "0x" << std::hex << hasher(input); 
    std::string out = ss.str(); 
    int diff = out.size() - len; 
    if (!diff) { return out; }
    out += std::string(std::abs(diff), '0'); 
    return out; 
}

std::string tools::upper(std::string* in)
{
    std::string out = *in;
    for (int t(0); t < in -> size(); ++t){out[t] = std::toupper(out[t]);}
    return out;
}

std::string tools::lower(std::string* in)
{
    std::string out = *in;
    for (int t(0); t < in -> size(); ++t){out[t] = std::tolower(out[t]);}
    return out;
}

std::string tools::capitalize(std::string* in)
{
    std::string out = *in;
    out[0] = std::toupper(out[0]);
    for (std::size_t t = 1; t < in->size(); ++t) {out[t] = std::tolower(out[t]);}
    return out;
}

std::string tools::urlencode(const std::string& value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }
        escaped << std::uppercase;
        escaped << '%' << std::setw(2) << int((unsigned char)c);
        escaped << std::nouppercase;
    }

    return escaped.str();
}

bool tools::has_string(std::string* inpt, std::string trg){
    std::size_t f = inpt -> find(trg); 
    if (f != std::string::npos){return true;}
    return false; 
}

std::string tools::remove_leading(const std::string& input, const std::string& leader){
        size_t startPos = input.find_first_not_of(leader);
        if (startPos == std::string::npos){return "";}
        return input.substr(startPos);
}
