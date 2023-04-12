#include <iostream>
#include <cmath>

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
            signed int pdgid = 0; 

            std::string _hash; 
 

    };
}
#endif 
