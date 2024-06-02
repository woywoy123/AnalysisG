#include <iostream> 
#include <string>

#ifndef STRUCTS_AUTHENTICATION_H
#define STRUCTS_AUTHENTICATION_H

struct authentication_t
{
    std::string header = ""; 

    std::string username = ""; 
    std::string password = ""; 
    int index = 0; // number of hashing iterations

    std::string public_key = ""; 
    std::string private_key = ""; 
    std::string password_key = ""; 

    // ---- relating to websocket ---- //
    std::string ws_auth_token = "";
    std::string ws_list_token = ""; 

    long ws_auth_since = -1; 
    
    // ---- ignore this part ---- //
    uint8_t cprivate_key[32]; 
    std::string hash = ""; 

    void flush(){
        username = ""; 
        password = ""; 
        public_key = ""; 
        private_key = ""; 
        password_key = ""; 
    }

    bool has_creds(){
        if (public_key.size()){return true;}
        if (private_key.size()){return true;}
        if (password_key.size()){return true;}
        return false;
    }
}; 

struct account_t {

    // internal authentication properties
    int api_public_key   = -1; 
    int api_private_key  = -1; 
    int api_password_key = -1; 
    int private_key      = -1; 

    // exchange data to be populated
    bool verified         = false; 
    bool deposit_enabled  = false; 
    bool order_enabled    = false;  
    bool cancel_enabled   = false; 
    bool withdraw_enabled = false; 
}; 

#endif
