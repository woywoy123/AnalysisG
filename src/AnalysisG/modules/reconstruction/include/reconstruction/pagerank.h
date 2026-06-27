#ifndef RECONSTRUCTION_PAGERANK_H
#define RECONSTRUCTION_PAGERANK_H

#include <templates/particle_template.h>
#include <notification/notification.h>
#include <tools/tools.h>

class pagerank: 
    public tools, 
    public notification
{
    public:
        pagerank(); 
        ~pagerank(); 

        void particles(std::vector<particle_template*> prt); 
        void edges_index(std::vector<long>* src, std::vector<long>* dst); 
        void edge_scores(std::vector<std::vector<double>>* scores, long cls); 
        void predict(); 
        
        long   max_iteration = 1e6;  
        long   min_partitions = 2; 

        double normalized    = 1e-6; 
        double alpha         = 0.85;
        double min_score     = 0.5; 


        bool converged = false; 
        long num_iter  = 0; 


    private:
        void weight_matrix(); 
        long get_index(std::vector<double>* val); 

        std::map<long, std::map<long, double>> wMij;
        std::map<long, std::map<long, double>> binary_map;
        std::map<long, double> reco_map; 



        // user input 
        std::vector<particle_template*> m_particles = {}; 
        std::vector<std::vector<double>> m_sc = {}; 
        std::vector<long> m_src = {}; 
        std::vector<long> m_dst = {}; 
        int m_cls = 0; 



    
}; 

#endif 
