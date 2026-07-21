#include <reconstruction/pagerank.h>
#include <cmath>

pagerank::pagerank(){}
pagerank::~pagerank(){}

void pagerank::particles(std::vector<particle_template*> prt){this -> m_particles = prt;}
void pagerank::edges_index(std::vector<long>* src, std::vector<long>* dst){
    this -> m_src = *src; this -> m_dst = *dst; 
}

void pagerank::edge_scores(std::vector<std::vector<double>>* scores, long cls){
    this -> m_sc = *scores; this -> m_cls = cls; 
}

long pagerank::get_index(std::vector<double>* val){
    long idx = 0; double idv = val -> at(0); 
    for (long x(1); x < val -> size(); ++x){
        double idv_ = val -> at(x); 
        if (idv_ <= idv){continue;}
        idv = idv_; idx = x; 
    }
    return idx; 
}

void pagerank::weight_matrix(){
    for (size_t x(0); x < this -> m_src.size(); ++x){
        long src = this -> m_src[x]; 
        long dst = this -> m_dst[x]; 
        long idy = this -> get_index(&m_sc[x]); 
        bool trig = (idy == this -> m_cls); 
        this -> reco_map[src] = 0;
        this -> reco_map[dst] = 0;
        this -> wMij[src][dst] = this -> m_sc[x][this -> m_cls]; 
    }
}

void pagerank::predict(){
    double n_nodes = double(this -> binary_map.size()); 
    long   l_nodes = long(this -> binary_map.size()); 

    // create a weighted adjacency matrix of the GNN output //
    std::map<long, std::map<long, double>> _Mij; 
    std::map<long, std::map<long, double>>::iterator itr;
    for (itr = this -> binary_map.begin(); itr != this ->  binary_map.end(); ++itr){
        long src = itr -> first; 
        for (long y(0); y < l_nodes; ++y){_Mij[src][y] = (src != y)* this -> wMij[src][y];}
    }

    // Normalize the adjacency matrix Mij such that the sum along the column 
    // gives 1 and can therefore be interpreted as a transition probability.
    // This transition probability can be interpreted as the probability
    // of moving from node i -> j.
    std::map<long, double> pr_;
    for (long y(0); y < l_nodes; ++y){
        double sm = 0; 
        for (long x(0); x < l_nodes; ++x){sm += _Mij[x][y];} 

        // account for null transition probability i.e. i -> j should not occur!
        sm = ((sm) ? 1.0/sm : 0); 

        // Magical alpha parameter assures that there is always some chance of the transition between i -> j. 
        // An analogy is that eventhough a website is not directly reachable via following links, something like
        // Google allows the user to still access it with some uniform probability i.e. (1/nodes)*alpha.
        for (long x(0); x < l_nodes; ++x){_Mij[x][y] = ((sm) ? _Mij[x][y]*sm : 1.0/n_nodes)*this -> alpha;}

        // makes sure the current node's probability is normalized by the number of nodes in the network
        pr_[y] = this -> wMij[y][y]/n_nodes;   
    }

    // PageRank equation: PR(p_i) = (1-alpha)/nodes + alpha*Sum(PR(p_j)/L(p_j))
    // p_i is the probability of reaching the current node under consideration, given a number of incident links
    // p_j are the linked probabilities that allow the transition between node j -> i.
    // PR(p_j) the current pagerank score of incident node j 
    // L(p_j) the number of outbound links from the perspective of node j.
    // The algorithm basically states that if there is a link between some pair of nodes, i and j, 
    // then what is the probability of moving from j to i, given that there are multiple nodes linking to j (L).

    // Make sure to account for non-convergence 
    unsigned long timeout = 0; 

    // copy the current pagerank state
    std::map<long, double> PR = pr_; 
    while (timeout < this -> max_iteration){
        // reset all pagerank values.
        pr_.clear(); 

        // compute the alpha * Sum(PR(p_j)/L(p_j)) term.
        double sx = 0; 
        for (long src(0); src < l_nodes; ++src){
            for (size_t x(0); x < l_nodes; ++x){pr_[src] += (_Mij[src][x]*PR[x]);}
            pr_[src] += (1- this -> alpha)/n_nodes; // (1-alpha)/nodes the probability of randomly landing on the current node
            sx += pr_[src]; // sx is for normalization purposes. This can occur when the GNN score is not between 0 -> 1.
        }

        // check whether the normalization has changed.
        // i.e. norm != 0
        double norm = 0;
        for (itr = this -> binary_map.begin(); itr != this -> binary_map.end(); ++itr){
            long n = itr -> first; 
            pr_[n] = pr_[n] / sx;
            norm += std::abs(pr_[n] - PR[n]); 
            PR[n] = pr_[n]; 
        }

        // continue if the iteration norm is not really small.
        ++timeout; 
        if (norm > this -> normalized){continue;}

        // Do a final PageRank computation
        norm = 0; 
        for (long x(0); x < l_nodes; ++x){
            double sc = 0; 
            for (long y(0); y < l_nodes; ++y){sc += (x != y) * _Mij[x][y] * (pr_[y]);}
            PR[x] = sc; norm += sc;
        }

        // check whether the normalization is null. Meaning that these are simply self connected nodes.
        // i.e. i -> i -> i -> i -> i -> i. This would cause self loops to be scored higher than 
        // and effectively be meaningless.
        if (!norm){break;}

        // final normalization of the node's pagerank
        for (long x(0); x < l_nodes; ++x){PR[x] = PR[x] / norm;}
        break; 
    }

    this -> num_iter = timeout; 
    this -> converged = this -> max_iteration > timeout;

    // clean up the node score
    std::map<long, double> tmp; 
    std::map<long, double>::iterator itp; 
    for (itr = this -> binary_map.begin(); itr != this -> binary_map.end(); ++itr){
        long src = itr -> first; 

        // if the current node has a null value for the pagerank, we dont care for it.
        if (!PR[src]){continue;}

        // more interesting if there is some score attached to it 
        for (itp = itr -> second.begin(); itp != itr -> second.end(); ++itp){
            double ptr = itp -> second; 
            long    dst = itp -> first; 
            // Make sure the GNN output score is greater than or equal to 0.5. 
            // This means we want to boost these connected nodes as import topological edges
            if (this -> binary_map[src][dst] < this -> min_score){continue;}
            tmp[dst] = ptr;
        }

        // If there is a low number of pageranked nodes, we dont particularly care for them.
        // Example: t -> W b -> q qbar b -> if b is lost, tmp <= 2 (only q qbar jet contributes to the partition). 
        // Do we care for them? Probably not.
        if (tmp.size() <= this -> min_partitions){continue;}

        // finalize the output score.
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){this -> reco_map[src] += PR[itp -> first];}
    }
} 



