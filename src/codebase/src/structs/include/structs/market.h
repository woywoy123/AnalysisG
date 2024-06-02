#ifndef STRUCTS_MARKET_H
#define STRUCTS_MARKET_H
#include <iostream>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include "coins.h"

std::string params(std::string key); 
std::string params(std::string key, std::string value, bool is_quote); 
std::string params(std::string key, std::string value, std::string delim = ","); 
std::vector<uint8_t> params(std::vector<uint8_t>* v1, std::vector<uint8_t>* v2); 

struct book_t {
    // z - (0 - bid, 1 - ask) | 
    // y - depth | 
    // x(0) - price (USDC/X), 
    // x(1) - vol (X), 
    // x(2) - vol*price [(USDC/X)*X -> USDC]
    // x(3) - vol/price [(X)/(X/USDC) -> USDC]
    // x(4) - sum(vol)
    // x(5) - sum(price*vol)
    // x(6) - sum(vol/price)
    //int sequence[2][50] = {0};
    double book[2][1024][7] = {0};
    int max_len = 1024;
    
    long sequence_norm = -1; 
    
    int bid_len = 0;
    int ask_len = 0; 
    
    char base_asset[64] = {0x0}; 
    char quote_asset[64] = {0x0}; 
    char market[128] = {0x0};
    
    double ask_price(int idx){return (idx < ask_len) ? book[1][idx][0] : -1;}
    double ask_vol(int idx){return (idx < ask_len) ? book[1][idx][1] : -1;}
    double ask_vol_price(int idx){return (idx < ask_len) ? book[1][idx][2] : -1;}
    double ask_vol_div_price(int idx){return (idx < ask_len) ? book[1][idx][3] : -1;}
    double ask_cum_vol(int idx){return (idx < ask_len) ? book[1][idx][4] : -1;}
    double ask_cum_price_vol(int idx){return (idx < ask_len) ? book[1][idx][5] : -1;}
    double ask_cum_vol_div_price(int idx){return (idx < ask_len) ? book[1][idx][6] : -1;}

    double bid_price(int idx){return (idx < ask_len) ? book[0][idx][0] : -1;}
    double bid_vol(int idx){return (idx < ask_len) ? book[0][idx][1] : -1;}
    double bid_vol_price(int idx){return (idx < ask_len) ? book[0][idx][2] : -1;}
    double bid_vol_div_price(int idx){return (idx < ask_len) ? book[0][idx][3] : -1;}
    double bid_cum_vol(int idx){return (idx < ask_len) ? book[0][idx][4] : -1;}
    double bid_cum_price_vol(int idx){return (idx < ask_len) ? book[0][idx][5] : -1;}
    double bid_cum_vol_div_price(int idx){return (idx < ask_len) ? book[1][idx][6] : -1;}

};

struct market_t {
    // format: (base-quote) - all lower case and with "-"
    char market[64] = {0x0}; 
    
    // format: lower case
    char base_asset[64] = {0x0}; 
    
    // format: lower case
    char quote_asset[64] = {0x0}; 
   
    int base_precision = 0;
    int quote_precision = 0;

    int pair_precision = 0;
    int qty_precision = 0; 
    
    double maker_fee = 0; 
    double taker_fee = 0; 
    double taker_lp_fee = 0; 
   
    double base_minimum = 0; 
    double quote_minimum = 0; 
    bool is_trading = true; 
 
    // hdf5 reconstruction
    std::string key(){return std::string(market);} 
}; 
#endif
