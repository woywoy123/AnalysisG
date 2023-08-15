#include <string>
#include <map>
#include <cmath>
#include <vector>
#include <iostream>
#include "../Headers/Tools.h"

#ifndef __CYTEMPLATE__H
#define __CYTEMPLATE__H

namespace CyTemplate
{
    class CyEventTemplate
    {
        public:
            CyEventTemplate(); 
            ~CyEventTemplate(); 
            
            signed int index = -1;
            double weight = 1; 
            bool deprecated = false; 
            std::string tree = ""; 
            std::string commit_hash = ""; 
           
            std::string Hash();  
            void Hash(std::string val); 
        
        private:
            std::string _hash = ""; 

    }; 
    
    class CyParticleTemplate
    {
        public:
            // Constructors 
            CyParticleTemplate(); 
            CyParticleTemplate(double px, double py, double pz, double e); 

            // Destructor 
            ~CyParticleTemplate(); 

            // Operators
            CyParticleTemplate operator+(const CyParticleTemplate& p)
            {
                CyParticleTemplate p2; 
                p2._px = this -> _px + p._px; 
                p2._py = this -> _py + p._py; 
                p2._pz = this -> _pz + p._pz; 
                p2._e  = this ->  _e + p._e;
                p2._initC = true;
                return p2; 
            }

            void operator+=(const CyParticleTemplate& p)
            {
                this -> _px += p._px; 
                this -> _py += p._py; 
                this -> _pz += p._pz; 
                this ->  _e += p._e;
                this -> _initC = true; 
            }

            bool operator==(const CyParticleTemplate& p2)
            {
                return this -> Hash() == p2._hash; 
            }
            
            // Book keeping variables
            signed int index = -1;
 
            // State indicator
            bool _initC = false; 
            bool _initP = false; 
            
            // ============== Transformation ================ //
            // Getter Functions
            double px(), py(), pz();  
            double pt(), eta(), phi(); 
            double e(); 

            // Setter Functions
            void px(double val), py(double val), pz(double val);  
            void pt(double val), eta(double val), phi(double val); 
            void e(double val); 

            // ============== End Transformation ================ //
            
            // ============== Physics ================ //
            // Getter Functions
            double Mass(), DeltaR(const CyParticleTemplate& p);  

            // Setter Functions
            void Mass(double val);  
            // ============== End Physics ================ //

            // ============== Particle ================ //
            // Getter Functions
            std::string Type = ""; 
            signed int pdgid(); 
            double charge(); 
            std::string symbol();
            bool is_lep(), is_nu(), is_b(); 

            // Setter Functions
            void pdgid(signed int id), charge(double val); 
            void symbol(std::string val); 
            std::vector<signed int> _lepdef = {11, 13, 15};
            std::vector<signed int> _nudef = {12, 14, 16}; 
            // ============== End Physics ================ //

            std::string Hash();

        private:
            // Internal Kinematic Variables
            double _phi = 0; 
            double _eta = 0; 
            double _pt = 0; 
            double _px = 0; 
            double _py = 0; 
            double _pz = 0; 
            double _e = 0; 
            double _mass = 0; 
            double _charge = 0;

            // PDGID Definitions 
            signed int _pdgid = 0;
            std::string _symbol = ""; 
            
            bool is(std::vector<signed int> p); 
            
            // Book keeping 
            std::string _hash = ""; 
    };
}

#endif 

