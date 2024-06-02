#ifndef COIN_H
#define COIN_H

struct coin_t {
    // meta data
    char symbol[128]      = {0x0};  //<lower case (symbol)>-<lower case (network)>
    char name[128]        = {0x0}; 
    char base_symbol[128] = {0x0}; 
    
    char network_symbol[128] = {0x0};
    char network_name[128]   = {0x0};
    bool is_busy             = false; 
    
    char token_contract_address[128] = {0x0};
    
    // exchange withdraw details
    double min_withdraw    =  0;
    double max_withdraw    = -1;  
    double fee_withdraw    =  0;
    bool withdraw_enabled  = true;
    
    // exchange deposit details
    int min_confirmations = -1; 
    double min_deposit    =  0; 
    bool deposit_enabled  = true; 
    
    // coin precision
    int asset_decimals    = 0; 
    int exchange_decimals = 0; 
    
    double balance = 0;
    char deposit_address[128] = {0x0};
    
    // hdf5 reconstruction
    std::string key(){return std::string(symbol);} 
}; 

#endif
