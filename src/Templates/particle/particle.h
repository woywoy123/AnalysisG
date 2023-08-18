#include <iostream>
#include <vector>
#include <map>

#ifndef PARTICLE_H
#define PARTICLE_H

struct ExportParticleTemplate
{
    double e; 
    
    double px; 
    double py; 
    double pz; 

    double pt; 
    double eta; 
    double phi; 

    double mass; 
    double charge; 

    int pdgid; 
    int index; 

    std::string hash; 
    std::string symbol;
    std::vector<int> lepdef; 
    std::vector<int> nudef;  
}; 


namespace CyTemplate
{
    class CyParticleTemplate
    {
        public:
            CyParticleTemplate(); 
            CyParticleTemplate(double px, double py, double pz, double e); 
            CyParticleTemplate(double px, double py, double pz); 
            ExportParticleTemplate MakeMapping(); 

            ~CyParticleTemplate(); 

            double e(); 
            void e(double val); 
            
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
            bool lep_decay();

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

            void ToCartesian(); 
            void ToPolar(); 

            std::string hash(); 
            void addleaf(std::string key, std::string leaf); 

            bool operator==(CyParticleTemplate* p); 
            CyParticleTemplate* operator+(CyParticleTemplate* p);
            void operator += (CyParticleTemplate* p); 
            void iadd(CyParticleTemplate* p); 
            
std::string type = ""; 
            int index = -1; 
            std::map<std::string, std::string> leaves = {}; 
            std::map<std::string, CyParticleTemplate*> children = {}; 
            std::map<std::string, CyParticleTemplate*> parent = {}; 

            std::vector<int> lepdef = {11, 13, 15};
            std::vector<int> nudef  = {12, 14, 16};         
        
        private:
            double _e = -0.000000000000001; 
            
            double _px = 0; 
            double _py = 0; 
            double _pz = 0; 
            
            double _pt = 0; 
            double _eta = 0; 
            double _phi = 0; 
            
            double _mass = -1;  

            bool _cartesian = false; 
            bool _polar = false; 

            int _pdgid = 0; 
            double _charge = 0; 
            std::string _hash = "";
            std::string _symbol = ""; 
            
            bool is(std::vector<int> p); 
    }; 
}
#endif
