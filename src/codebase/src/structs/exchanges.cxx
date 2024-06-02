#include <trading.h>
#include <market.h>
#include <iomanip>
#include <iostream>
#include <sstream>

std::string params(std::string key){return "\"" + key + "\"";}
std::string params(std::string key, std::string value, std::string delim){
    return params(key) + ":" + params(value) + delim; 
}

std::string params(std::string key, std::string value, bool is_quote){
    std::string out = params(key) + ":"; 
    if (is_quote){out += params(value);}
    else {out += value;}
    return out; 
}


std::vector<uint8_t> params(std::vector<uint8_t>* v1, std::vector<uint8_t>* v2){
    std::vector<uint8_t> vec; 
    vec.insert(vec.end(), v1 -> begin(), v1 -> end());
    vec.insert(vec.end(), v2 -> begin(), v2 -> end());
    return vec; 
}

void post_t::add_params(std::string key, std::string val, bool is_quote){
    field_keys.push_back(key); 
    field_values.push_back(val); 
    field_quotes.push_back(is_quote); 
}

void post_t::add_params_front(std::string key, std::string val, bool is_quote){
    field_keys.insert(field_keys.begin(), key); 
    field_values.insert(field_values.begin(), val); 
    field_quotes.insert(field_quotes.begin(), is_quote); 
}


std::string post_t::build_url(){
    std::string out; 
    if (!field_keys.size()){return "";}
    size_t t = field_keys.size()-1; 
    for (size_t x(0); x < t; ++x){
        out += field_keys[x] + "=";
        out += field_values[x] + "&"; 
    } 
    out += field_keys[t] + "=";
    out += field_values[t]; 
    return out; 
}

void post_t::build_eth(){
    eth_signature = ""; 
    int s = data.size()-1; 
    if (s < 0){return;}
    s = field_keys.size()-1; 
    for (int x(0); x < s; ++x){
        eth_signature += params(field_keys[x], field_values[x], ",");
    }
    eth_signature += params(field_keys[s], field_values[s], ""); 

    to_sign = {}; 
    for (int x(0); x < data.size(); ++x){to_sign = params(&to_sign, &data[x]);}
} 

std::string post_t::build_data(){
    if (!field_keys.size()){return "";}
    data_to_send = "{";
    int s = field_keys.size()-1; 
    for (int x(0); x < s; ++x){
        data_to_send += params(field_keys[x], field_values[x], field_quotes[x]) + ",";
    }
    data_to_send += params(field_keys[s], field_values[s], field_quotes[s]);
    data_to_send += "}"; 
    return data_to_send; 
}

void post_t::build_header(){
    header_send = {}; 
    for (int x(0); x < header_keys.size(); ++x){
        header_send.push_back(header_keys[x] + ":" + header_values[x]);  
    }
}
