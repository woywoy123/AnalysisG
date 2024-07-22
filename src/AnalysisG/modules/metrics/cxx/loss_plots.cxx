#include <metrics/metrics.h>

void metrics::generic_painter(
        std::vector<TGraph*> k_graphs,
        std::string path, std::string title, 
        std::string xtitle, std::string ytitle
){
    TCanvas* can = new TCanvas();
    gStyle -> SetOptStat(0); 
    gStyle -> SetImageScaling(3); 
    can -> SetTopMargin(0.1);
    gStyle -> SetTitleOffset(1.25);
    gStyle -> SetTitleSize(0.025); 
    gStyle -> SetLabelSize(0.025, "XY"); 
    
    TMultiGraph* mtg = new TMultiGraph();
    mtg -> SetTitle(title.c_str()); 
    for (size_t x(0); x < k_graphs.size(); ++x){
        TGraph* gr = k_graphs[x]; 
        gr -> SetLineWidth(1); 
        gr -> SetLineColor(this -> colors_h[x]); 
        mtg -> Add(gr); 
    }
    mtg -> Draw("APL"); 
    mtg -> GetXaxis() -> SetTitle(xtitle.c_str()); 
    mtg -> GetXaxis() -> CenterTitle(xtitle.c_str()); 
    mtg -> GetYaxis() -> SetTitle(ytitle.c_str()); 
    mtg -> GetYaxis() -> CenterTitle(ytitle.c_str()); 
    mtg -> SetMinimum(0.); 
    gPad -> Modified();
    
    can -> Modified(); 
    //can -> BuildLegend(); 
    can -> Update(); 
    
    this -> create_path(path); 
    can -> SaveAs(path.c_str()); 
    can -> Close(); 
    delete can; 
    delete mtg; 
} 


std::map<std::string, std::vector<TGraph*>> metrics::build_graphs(
        std::map<std::string, TH1F*>* train, std::map<std::string, float>* rep_tr, 
        std::map<std::string, TH1F*>* valid, std::map<std::string, float>* rep_va, 
        std::map<std::string, TH1F*>* eval,  std::map<std::string, float>* rep_ev, 
        int epoch
){

    std::map<std::string, std::vector<TGraph*>> output = {};
    std::map<std::string, TH1F*>::iterator thi = train -> begin(); 
    for (; thi != train -> end(); ++thi){
        (*rep_tr)[thi ->first] = thi -> second -> GetBinContent(epoch+1); 
        output[thi -> first].push_back(new TGraph(thi -> second));
    }

    thi = valid -> begin(); 
    for (; thi != valid -> end(); ++thi){
        (*rep_va)[thi ->first] = thi -> second -> GetBinContent(epoch+1); 
        output[thi -> first].push_back(new TGraph(thi -> second));
    }

    thi = eval -> begin(); 
    for (; thi != eval -> end(); ++thi){
        (*rep_ev)[thi ->first] = thi -> second -> GetBinContent(epoch+1); 
        output[thi -> first].push_back(new TGraph(thi -> second));
    }
    return output; 
}


void metrics::dump_loss_plots(int k){
    std::map<std::string, std::vector<TGraph*>>::iterator gri; 
    std::map<int, analytics_t>::iterator itr = this -> registry.begin(); 
    for (; itr != this -> registry.end(); ++itr){
        if (itr -> first != k){continue;}

        int k_ = itr -> first+1; 
        int ep = itr -> second.this_epoch; 

        // graphs - features
        std::map<std::string, std::vector<TGraph*>> k_graph = this -> build_graphs(
            &itr -> second.loss_graph[mode_enum::training]  , &itr -> second.report -> loss_graph[mode_enum::training],
            &itr -> second.loss_graph[mode_enum::validation], &itr -> second.report -> loss_graph[mode_enum::validation],
            &itr -> second.loss_graph[mode_enum::evaluation], &itr -> second.report -> loss_graph[mode_enum::evaluation],
            ep
        ); 

        for (gri = k_graph.begin(); gri != k_graph.end(); ++gri){
            std::string var_name = gri -> first; 
            std::string title = "MVA Loss of: " + var_name + " for k-fold: " + std::to_string(k_); 
            std::string ptx = this -> m_settings.output_path + "loss-graph/";
            ptx += var_name + "-kfold_" + std::to_string(k_) + ".png"; 
            this -> generic_painter(gri -> second, ptx, title, "Epochs", "MVA Loss (Arb.)"); 
        }

        // nodes - features
        std::map<std::string, std::vector<TGraph*>> k_nodes = this -> build_graphs(
            &itr -> second.loss_node[mode_enum::training],   &itr -> second.report -> loss_node[mode_enum::training],
            &itr -> second.loss_node[mode_enum::validation], &itr -> second.report -> loss_node[mode_enum::validation],
            &itr -> second.loss_node[mode_enum::evaluation], &itr -> second.report -> loss_node[mode_enum::evaluation],
            ep
        ); 

        for (gri = k_nodes.begin(); gri != k_nodes.end(); ++gri){
            std::string var_name = gri -> first; 
            std::string title = "MVA Loss of: " + var_name + " for k-fold: " + std::to_string(k_); 
            std::string ptx = this -> m_settings.output_path + "loss-node/"; 
            ptx += var_name + "-kfold_" + std::to_string(k_) + ".png"; 
            this -> generic_painter(gri -> second, ptx, title, "Epochs", "MVA Loss (Arb.)"); 
        }

        // edges - features
        std::map<std::string, std::vector<TGraph*>> k_edge = this -> build_graphs(
            &itr -> second.loss_edge[mode_enum::training],   &itr -> second.report -> loss_edge[mode_enum::training],
            &itr -> second.loss_edge[mode_enum::validation], &itr -> second.report -> loss_edge[mode_enum::validation],  
            &itr -> second.loss_edge[mode_enum::evaluation], &itr -> second.report -> loss_edge[mode_enum::evaluation],
            ep
        ); 

        for (gri = k_edge.begin(); gri != k_edge.end(); ++gri){
            std::string var_name = gri -> first; 
            std::string title = "MVA Loss of: " + var_name + " for k-fold: " + std::to_string(k_); 
            std::string ptx = this -> m_settings.output_path + "loss-edge/";
            ptx += var_name + "-kfold_" + std::to_string(k_) + ".png"; 
            this -> generic_painter(gri -> second, ptx, title, "Epochs", "MVA Loss (Arb.)"); 
        }
    }
}

void metrics::add_th1f_loss(
        std::map<std::string, torch::Tensor>* type, 
        std::map<std::string, TH1F*>* lss_type, 
        int kfold, int len
){
    analytics_t* an = &this -> registry[kfold];  
    std::map<std::string, torch::Tensor>::iterator itx = type -> begin();
    for (; itx != type -> end(); ++itx){
        (*lss_type)[itx -> first] -> Fill(an -> this_epoch, itx -> second.item<float>()/float(len));
    }
}

void metrics::build_th1f_loss(
        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* type, 
        graph_enum g_num, int kfold
){
    analytics_t* an = &this -> registry[kfold]; 
    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 
    for (itx = type -> begin(); itx != type -> end(); ++itx){
        std::string var_name = itx -> first; 

        std::string title = "";
        std::map<mode_enum, std::map<std::string, TH1F*>>* type_ = nullptr; 
        switch (g_num){
            case graph_enum::truth_graph: title = "Graph"; type_ = &an -> loss_graph; break; 
            case graph_enum::truth_node:  title = "Node";  type_ = &an -> loss_node; break; 
            case graph_enum::truth_edge:  title = "Edge";  type_ = &an -> loss_edge; break; 
            default: return; 
        }


        int ep = this -> m_settings.epochs; 
        std::string run_name = this -> m_settings.run_name; 
        std::string base_name = run_name + "-" + var_name + "-" + title; 

        std::string tr = base_name + " Parameter (training) Loss " + std::to_string(kfold+1) + "-fold"; 
        (*type_)[mode_enum::training][var_name] = new TH1F(tr.c_str(), "training", ep, 0, ep); 

        std::string va = base_name + " Parameter (validation) Loss" + std::to_string(kfold+1) + "-fold"; 
        (*type_)[mode_enum::validation][var_name] = new TH1F(va.c_str(), "validation", ep, 0, ep); 

        std::string ev = base_name + " Parameter (evaluation) Loss" + std::to_string(kfold+1) + "-fold"; 
        (*type_)[mode_enum::evaluation][var_name] = new TH1F(ev.c_str(), "evaluation", ep, 0, ep); 
    }
}


void metrics::build_th1f_accuracy(
        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* type, 
        graph_enum g_num, int kfold
){
    analytics_t* an = &this -> registry[kfold]; 
    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 
    for (itx = type -> begin(); itx != type -> end(); ++itx){
        std::string var_name = itx -> first; 

        std::string title = "";
        std::map<mode_enum, std::map<std::string, TH1F*>>* type_ = nullptr; 
        switch (g_num){
            case graph_enum::truth_graph: title = "Graph"; type_ = &an -> accuracy_graph; break; 
            case graph_enum::truth_node:  title = "Node";  type_ = &an -> accuracy_node; break; 
            case graph_enum::truth_edge:  title = "Edge";  type_ = &an -> accuracy_edge; break; 
            default: return; 
        }

        int ep = this -> m_settings.epochs; 
        std::string run_name = this -> m_settings.run_name; 
        std::string base_name = run_name + "-" + var_name + "-" + title; 

        std::string tr = base_name + " Parameter (training) Accuracy " + std::to_string(kfold+1) + "-fold"; 
        TH1F* tr_ = new TH1F(tr.c_str(), "training", ep, 0, ep); 
        tr_ -> GetYaxis() -> SetRangeUser(0, 100); 
        (*type_)[mode_enum::training][var_name] = tr_; 

        std::string va = base_name + " Parameter (validation) Accuracy" + std::to_string(kfold+1) + "-fold"; 
        TH1F* va_ = new TH1F(va.c_str(), "validation", ep, 0, ep); 
        va_ -> GetYaxis() -> SetRangeUser(0, 100); 
        (*type_)[mode_enum::validation][var_name] = va_; 

        std::string ev = base_name + " Parameter (evaluation) Accuracy" + std::to_string(kfold+1) + "-fold"; 
        TH1F* ev_ = new TH1F(ev.c_str(), "evaluation", ep, 0, ep); 
        ev_ -> GetYaxis() -> SetRangeUser(0, 100); 
        (*type_)[mode_enum::evaluation][var_name] = ev_; 
    }
}

void metrics::add_th1f_accuracy(torch::Tensor* pred, torch::Tensor* truth, TH1F* hist, int kfold, int len){
    analytics_t* an = &this -> registry[kfold]; 
    torch::Tensor pred_  = std::get<1>(pred -> max({-1})).view({-1}); 
    torch::Tensor truth_ = truth -> view({-1});
    torch::Tensor acc = (truth_ == pred_).sum({-1}).to(torch::kFloat)/float(pred_.size({-1})); 
    hist -> Fill(an -> this_epoch, 100*acc.item<float>()/float(len));
}


void metrics::dump_accuracy_plots(int k){
    std::map<int, analytics_t>::iterator itr = this -> registry.begin(); 
    for (; itr != this -> registry.end(); ++itr){
        if (itr -> first != k){continue;}

        std::map<std::string, std::vector<TGraph*>>::iterator gri; 

        int k_ = itr -> first+1; 
        int ep = itr -> second.this_epoch; 

        // graphs - features
        std::map<std::string, std::vector<TGraph*>> k_graph = this -> build_graphs(
            &itr -> second.accuracy_graph[mode_enum::training],   &itr -> second.report -> accuracy_graph[mode_enum::training],
            &itr -> second.accuracy_graph[mode_enum::validation], &itr -> second.report -> accuracy_graph[mode_enum::validation],
            &itr -> second.accuracy_graph[mode_enum::evaluation], &itr -> second.report -> accuracy_graph[mode_enum::evaluation],
            ep
        ); 

        std::string base_name = this -> m_settings.output_path + "accuracy-graph/"; 
        for (gri = k_graph.begin(); gri != k_graph.end(); ++gri){
            std::string var_name = gri -> first; 
            std::string title = "MVA Accuracy of: " + var_name + " for k-fold: " + std::to_string(k_); 
            std::string ptx = base_name + var_name + "-kfold_" + std::to_string(k_) + ".png"; 
            this -> generic_painter(gri -> second, ptx, title, "Epochs", "MVA Accuracy (%)"); 
        }

        // nodes - features
        std::map<std::string, std::vector<TGraph*>> k_nodes = this -> build_graphs(
            &itr -> second.accuracy_node[mode_enum::training],   &itr -> second.report -> accuracy_node[mode_enum::training],
            &itr -> second.accuracy_node[mode_enum::validation], &itr -> second.report -> accuracy_node[mode_enum::validation],
            &itr -> second.accuracy_node[mode_enum::evaluation], &itr -> second.report -> accuracy_node[mode_enum::evaluation],
            ep
        ); 

        base_name = this -> m_settings.output_path + "accuracy-node/"; 
        for (gri = k_nodes.begin(); gri != k_nodes.end(); ++gri){
            std::string var_name = gri -> first; 
            std::string title = "MVA Accuracy of: " + var_name + " for k-fold: " + std::to_string(k_); 
            std::string ptx = base_name + var_name + "-kfold_" + std::to_string(k_) + ".png"; 
            this -> generic_painter(gri -> second, ptx, title, "Epochs", "MVA Accuracy (%)"); 
        }

        // edges - features
        std::map<std::string, std::vector<TGraph*>> k_edge = this -> build_graphs(
            &itr -> second.accuracy_edge[mode_enum::training],   &itr -> second.report -> accuracy_edge[mode_enum::training],
            &itr -> second.accuracy_edge[mode_enum::validation], &itr -> second.report -> accuracy_edge[mode_enum::validation],
            &itr -> second.accuracy_edge[mode_enum::evaluation], &itr -> second.report -> accuracy_edge[mode_enum::evaluation],
            ep
        ); 

        base_name = this -> m_settings.output_path + "accuracy-edge/"; 
        for (gri = k_edge.begin(); gri != k_edge.end(); ++gri){
            std::string var_name = gri -> first; 
            std::string title = "MVA Accuracy of: " + var_name + " for k-fold: " + std::to_string(k_); 
            std::string ptx = base_name + var_name + "-kfold_" + std::to_string(k_) + ".png"; 
            this -> generic_painter(gri -> second, ptx, title, "Epochs", "MVA Accuracy (%)"); 
        }
    }
}

