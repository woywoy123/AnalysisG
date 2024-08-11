#ifndef PARTICLES_STRUCTS_H
#define PARTICLES_STRUCTS_H

#include <string>
#include <vector>
#include <map>

struct particle_t {
    double e = -0.000000000000001; 
    double mass = -1;  

    double px = 0; 
    double py = 0; 
    double pz = 0; 

    double pt = 0; 
    double eta = 0; 
    double phi = 0; 

    bool cartesian = false; 
    bool polar = false; 

    double charge = 0; 
    int pdgid = 0; 
    int index = -1; 

    std::string type = ""; 
    std::string hash = "";
    std::string symbol = "";  

    std::vector<int> lepdef = {11, 13, 15};
    std::vector<int> nudef  = {12, 14, 16};         
    std::map<std::string, bool> children = {};
    std::map<std::string, bool> parents = {};

}; 

#endif
