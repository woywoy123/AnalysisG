#include <iostream>

#ifndef CYTEMPLATE_H
#define CYTEMPLATE_H

namespace CyTemplate
{
    class CyParticle
    {
        public:
            CyParticle(); 
            ~CyParticle(); 
            
            // Book keeping variables
            signed int index = -1;
            std::string hash = ""; 
 
            // State indicator
            bool _edited = true; 

            double Px(); 
            double Py(); 
            double Pz(); 
            double E(); 
        
        private:
            
            // Internal Kinematic Variables
            double _phi, _eta, _pt, _px, _py, _pz, _e; 
            
            // Internal Functions 
            void _Hash();



    };
}
#endif 
