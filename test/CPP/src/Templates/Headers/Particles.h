#include <iostream>
#include <cmath>
#include <vector>
#include <map>

#ifndef CYTEMPLATE_H
#define CYTEMPLATE_H

namespace CyTemplate
{
    class CyParticle
    {
        public:
            // Constructors 
            CyParticle(); 
            CyParticle(double px, double py, double pz, double e); 

            // Destructor 
            ~CyParticle(); 

            // Operators
            CyParticle operator+(const CyParticle& p)
            {
                CyParticle p2; 
                p2._px = this -> _px + p._px; 
                p2._py = this -> _py + p._py; 
                p2._pz = this -> _pz + p._pz; 
                p2._e  = this ->  _e + p._e; 
                return p2; 
            }

            void operator+=(const CyParticle& p)
            {
                this -> _px += p._px; 
                this -> _py += p._py; 
                this -> _pz += p._pz; 
                this ->  _e += p._e; 
            }

            bool operator==(const CyParticle& p2)
            {
                return (this -> _hash) == (p2._hash); 
            }
            
            // Book keeping variables
            signed int index = -1;
 
            // State indicator
            bool _edited = true; 
            
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
            double Mass(), DeltaR(const CyParticle& p);  

            // Setter Functions
            void Mass(double val);  
            // ============== End Physics ================ //

            // ============== Particle ================ //
            // Getter Functions
            signed int pdgid(); 
            double charge(); 
            std::string symbol();
            bool is_lep(), is_nu(), is_b(); 

            // Setter Functions
            void pdgid(signed int id), charge(double val); 
            std::vector<signed int> _lepdef = {11, 13, 15};
            std::vector<signed int> _nudef = {12, 14, 16}; 
            // ============== End Physics ================ //

            std::string Hash(); 
            void _UpdateState();

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
            std::string _hash; 
 
            
    };
}
#endif 
