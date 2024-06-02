#ifndef STRUCTS_TRADING_H
#define STRUCTS_TRADING_H
#include <iostream>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

struct post_t {
    bool use_rest = false;
    std::string nonce  = ""; 
    std::string domain = ""; 
    std::string path = "";
    
    std::string eth_signature = "";     
    std::string api_signature = ""; 
    
    std::vector<std::vector<uint8_t>> data; 
    
    std::vector<std::string> field_keys;
    std::vector<std::string> field_values;
    std::vector<bool>        field_quotes;  
    
    std::vector<std::string> header_keys; 
    std::vector<std::string> header_values; 
    
    std::string data_to_sign;  
    std::string data_to_send;  
    std::vector<uint8_t> to_sign; 
    std::vector<std::string> header_send; 
   
    void build_eth(); 
    std::string build_data();
    void build_header();
    std::string build_url(); 

    void add_params(std::string key, std::string value, bool is_quote = true); 
    void add_params_front(std::string key, std::string value, bool is_quote = true);
}; 

struct delta_t {
    std::string algo_name = "";
    std::string sell_path = ""; 
    std::string buy_path  = "";
    std::string exchange_1 = "";
    std::string exchange_2 = "";  

    bool make_trade = false; 
    double profit_percentage = -1; 
    double cprofit = -1;
    double volume  =  0;  
    long epoch_time = 0; 

    std::vector<double> sell_volume = {}; 
    std::vector<double> buy_volume  = {};
}; 


#endif
