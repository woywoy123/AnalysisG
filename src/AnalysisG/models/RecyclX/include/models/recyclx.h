#ifndef GNN_RECYCLIX_H
#define GNN_RECYCLIX_H
#include <templates/model_template.h>

class recyclx: public model_template
{
    public:
        recyclx(); 
        ~recyclx();
        model_template* clone() override;
        void forward(graph_t*) override; 

<<<<<<< HEAD:src/AnalysisG/models/grift/include/models/grift.h
        torch::Tensor message(
            torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc,  
            torch::Tensor  hx_i, torch::Tensor hx_j
        ); 
=======
        torch::Tensor message(torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc); 

        bool break_loop(torch::Tensor inpt); 
>>>>>>> 9efb8567 (Recyclx):src/AnalysisG/models/RecyclX/include/models/recyclx.h

        // Neural Network Parameters
        int _hidden = 1024; 
        int _xrec = 128; 

        int _xin  = 6; 
        int _xout = 2; 
        int _xtop = 5; 
        double drop_out = 0.01; 

        // Misc
        bool is_mc    = true; 
        bool init     = false; 

        torch::nn::Sequential* rnn_x        = nullptr; 
        torch::nn::Sequential* rnn_dx       = nullptr; 
        torch::nn::Sequential* rnn_top_edge = nullptr;
        torch::nn::Sequential* rnn_res_edge = nullptr; 
        torch::nn::Sequential* rnn_hxx      = nullptr; 

        torch::nn::Sequential* autoenc = nullptr; 
        torch::nn::Sequential* autodec = nullptr; 

        torch::nn::Sequential* mlp_ntop = nullptr; 
        torch::nn::Sequential* mlp_sig  = nullptr; 

<<<<<<< HEAD:src/AnalysisG/models/grift/include/models/grift.h
        torch::Tensor  x_nulls; 
=======
>>>>>>> 9efb8567 (Recyclx):src/AnalysisG/models/RecyclX/include/models/recyclx.h
        torch::Tensor dx_nulls; 
        torch::Tensor te_nulls; 
}; 

#endif
