from common import *


def attestation(t, m):
    for i in range(len(t)):
        for j in range(len(t)):
            assert t[i][j] == m[i][j]

def PageRank(event):
    edge_index  = event.edge_index
    edge_scores = event.edge_scores
    bin_top     = event.bin_top
    matrix      = event.bin_top_matrix
    mij         = event.Mij
    prg_i       = event.PR

    mx_top = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(edge_index[0])):
        src, dst = edge_index[0][i], edge_index[1][i]
        binx = edge_scores[0][i] < edge_scores[1][i]
        assert binx == bin_top[i]
        mx_top[src][dst] = edge_scores[1][i]

    attestation(matrix, mx_top)
    mx_ij = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)): mx_ij[i][j] = (i != j)*mx_top[i][j]

    attestation(mx_ij, mij)
    print(mij)




#    auto cluster = [this](
#            std::map<int, std::map<std::string, particle_gnn*>>* clust, 
#            std::map<std::string, float>* bin_out,
#            std::map<int, std::map<int, float>>* bin_data
#    ){
#
#        float alpha = 0.85; 
#        float n_nodes = clust -> size(); 
#        std::map<int, std::map<int, float>> Mij; 
#        std::map<int, std::map<std::string, particle_gnn*>>::iterator itr;
#
#        for (itr = clust -> begin(); itr != clust -> end(); ++itr){
#            int src = itr -> first; 
#            for (size_t y(0); y < n_nodes; ++y){Mij[src][y] = (src != y)*(*bin_data)[src][y];}
#        }
#        //std::cout << " ---------- Mij ----------- "<< std::endl;
#        //this -> print(&Mij); 
#
#        std::map<int, float> pr_;
#        for (size_t y(0); y < n_nodes; ++y){
#            float sm = 0; 
#            for (size_t x(0); x < n_nodes; ++x){sm += Mij[x][y];} 
#            sm = ((sm) ? 1.0/sm : 0); 
#            for (size_t x(0); x < n_nodes; ++x){Mij[x][y] = ((sm) ? Mij[x][y]*sm : 1.0/n_nodes)*alpha;}
#            pr_[y] = (*bin_data)[y][y]/n_nodes;  
#        }
#        //std::cout << "---------- pr_0 --------- " << std::endl;
#        //this -> print(&pr_); 
#
#        int timeout = 0; 
#        std::map<int, float> PR = pr_; 
#        while (bin_data){
#            pr_.clear(); 
#            float sx = 0; 
#            for (size_t src(0); src < n_nodes; ++src){
#                for (size_t x(0); x < n_nodes; ++x){pr_[src] += (Mij[src][x]*PR[x]);}
#                pr_[src] += (1-alpha)/n_nodes; 
#                sx += pr_[src]; 
#            }
#            itr = clust -> begin(); 
#
#            float norm = 0; 
#            for (; itr != clust -> end(); ++itr){
#                pr_[itr -> first] = pr_[itr -> first] / sx;
#                norm += std::abs(pr_[itr -> first] - PR[itr -> first]); 
#                PR[itr -> first] = pr_[itr -> first]; 
#            }
#            timeout += 1; 
#
#            //std::cout << "------------ PR_" << timeout << "----------" << std::endl;
#            //this -> print(&PR);
#
#            if (norm > 1e-6 && timeout < 1e6){continue;}
#            norm = 0; 
#            for (size_t x(0); x < n_nodes; ++x){
#                float sc = 0; 
#                for (size_t y(0); y < n_nodes; ++y){
#                    //if ((*bin_data)[x][y] <= 0.5){continue;}
#                    sc += (x != y) * Mij[x][y] * (pr_[y]); 
#                }
#                PR[x] = sc;  
#                norm += sc;
#            }
#            if (!norm){break;}
#            for (size_t x(0); x < n_nodes; ++x){PR[x] = PR[x] / norm;}
#            //std::cout << "------------ PR_F-------------" << std::endl;
#            //this -> print(&PR); 
#            break; 
#        }









#interpret(10)
data = loadsPage()
for i in data:
    PageRank(data[i])
    exit()





