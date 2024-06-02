#include <iostream>
#include <string>

#ifndef STRUCTS_USER_H
#define STRUCTS_USER_H

struct config_t
{
    std::string root_path = "."; 
    std::string bot_name     = "CrossBot";
    std::string config_path  = "settings"; 
    std::string exchange_path = "exchanges";

    // files
    std::string auth_file    = "secrets"; 

};

#endif
