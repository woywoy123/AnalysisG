#ifndef PARTICLES_H
#define PARTICLES_H
#include <structs/particles.h>
#include <tools/tools.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>

class particles
{
    public:
        particles();
        ~particles(); 

        particles(particle_t* p);
        particles(double px, double py, double pz, double e); 
        particles(double px, double py, double pz);

        // interface.cxx
        void e(double val); 
 
        // physics.cxx
        double e(); 
        double DeltaR(particles* p);
 
        void mass(double val); 
        double mass(); 

        void pdgid(int val); 
        int pdgid(); 

        void symbol(std::string val); 
        std::string symbol(); 

        void charge(double val); 
        double charge(); 

        bool is_lep(); 
        bool is_nu(); 
        bool is_b(); 
        bool is_add(); 
        bool lep_decay(std::vector<particle_t>*);

        // ===== Cartesian ===== //
        // getter
        double px(); 
        double py(); 
        double pz(); 

        // setter
        void px(double val); 
        void py(double val);
        void pz(double val); 

        // ===== Polar ==== // 
        // getter
        double pt(); 
        double eta(); 
        double phi(); 

        // setter
        void pt(double val); 
        void eta(double val);
        void phi(double val); 

        void to_cartesian(); 
        void to_polar(); 

        std::string hash(); 
        void add_leaf(std::string key, std::string leaf); 
        bool is(std::vector<int> p); 

        bool operator == (particles& p); 
        particles* operator+(particles* p);
        void operator += (particles* p); 
        void iadd(particles* p); 
      
        bool register_parent(particles* p);
        bool register_child(particles* p);

        std::map<std::string, std::string> leaves = {}; 
        std::map<std::string, particles*> parents = {}; 
        std::map<std::string, particles*> children = {}; 
        particle_t data;  
}; 
#endif

