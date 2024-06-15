#ifndef EVENT_STRUCTS_H
#define EVENT_STRUCTS_H

#include <iostream>
#include <string>

struct event_t {
    std::string name = "";

    // state variables
    double weight = 1;
    long   index = -1;

    std::string hash = "";
    std::string tree = "";
};

#endif
