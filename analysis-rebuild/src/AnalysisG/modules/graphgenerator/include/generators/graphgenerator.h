#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include <generators/sampletracer.h>

class graphgenerator: public sampletracer 
{
    public:
        graphgenerator();
        ~graphgenerator();

        void add_graph_template(std::map<std::string, graph_template*>* inpt); 
}; 

#endif
