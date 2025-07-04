#include <metrics/metrics.h>
#include <TRatioPlot.h>
#include <THStack.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

void metrics::dump_mass_plots(int k){
    std::string out_pth = this -> m_settings.output_path + "masses/"; 
    analytics_t* an = &this -> registry[k]; 

    std::map<std::string, std::vector<TH1F*>> hists_pred = {}; 
    std::map<std::string, TH1F*>* pred_tr = &an -> pred_mass_edge[mode_enum::training]; 
    std::map<std::string, TH1F*>* pred_va = &an -> pred_mass_edge[mode_enum::validation]; 
    std::map<std::string, TH1F*>* pred_ev = &an -> pred_mass_edge[mode_enum::evaluation]; 

    std::map<std::string, TH1F*>::iterator ith; 
    for (ith = pred_tr -> begin(); ith != pred_tr -> end(); ++ith){
        ith -> second -> SetTitle("training"); 
        hists_pred[ith -> first].push_back(ith -> second); 
    } 

    for (ith = pred_va -> begin(); ith != pred_va -> end(); ++ith){
        ith -> second -> SetTitle("validation"); 
        hists_pred[ith -> first].push_back(ith -> second); 
    } 

    for (ith = pred_ev -> begin(); ith != pred_ev -> end(); ++ith){
        ith -> second -> SetTitle("evaluation"); 
        hists_pred[ith -> first].push_back(ith -> second); 
    } 

    std::map<std::string, TH1F*>* tru_tr = &an -> truth_mass_edge[mode_enum::training]; 
    std::map<std::string, TH1F*>* tru_va = &an -> truth_mass_edge[mode_enum::validation]; 
    std::map<std::string, TH1F*>* tru_ev = &an -> truth_mass_edge[mode_enum::evaluation]; 

    int nb = this -> m_settings.nbins;
    int mxr = this -> m_settings.max_range; 

    std::map<std::string, TH1F*> merged = {}; 
    for (ith = tru_tr -> begin(); ith != tru_tr -> end(); ++ith){
        merged[ith -> first] = new TH1F(("truth-" + ith -> first).c_str(), "truth", nb, 0, mxr); 
        merged[ith -> first] -> Add(ith -> second);
        ith -> second -> Reset();  
    } 
    for (ith = tru_va -> begin(); ith != tru_va -> end(); ++ith){
        merged[ith -> first] -> Add(ith -> second);
        ith -> second -> Reset(); 
    } 
    for (ith = tru_ev -> begin(); ith != tru_ev -> end(); ++ith){
        merged[ith -> first] -> Add(ith -> second);
        ith -> second -> Reset(); 
    } 


    for (ith = merged.begin(); ith != merged.end(); ++ith){
        THStack* h_sum = new THStack(("THStack" +  ith -> first).c_str(), ("Mass Reconstruction - " + ith -> first).c_str()); 
        TLegend* legend = new TLegend(0.6,0.6,0.9,0.9);
        legend -> SetHeader("MVA Mass Reconstruction", "C"); 
        for (size_t x(0); x < hists_pred[ith -> first].size(); ++x){
            hists_pred[ith -> first][x] -> SetLineColor(this -> colors_h[x]); 
            hists_pred[ith -> first][x] -> SetFillColor(this -> colors_h[x]); 
            h_sum -> Add(hists_pred[ith -> first][x]);
            legend -> AddEntry(hists_pred[ith -> first][x]);
        }

        // truth histogram 
        TH1F* truth_ = ith -> second; 
        truth_ -> SetLineColor(kBlack); 
        truth_ -> SetFillColor(kGray); 
        truth_ -> SetFillColorAlpha(kGray, 0.8); 
        legend -> AddEntry(truth_); 

        TCanvas* can = new TCanvas();
        gStyle -> SetOptStat(0); 
        gStyle -> SetImageScaling(3); 
        can -> SetTopMargin(0.1);
        gStyle -> SetTitleOffset(1.25);
        gStyle -> SetTitleSize(0.025); 
        gStyle -> SetLabelSize(0.025, "XY"); 

        double dx = truth_ -> GetMaximum();
        double di = h_sum -> GetMaximum(); 
        h_sum -> SetMaximum(((dx > di) ? dx : di)*1.1); 
        TRatioPlot* rp = new TRatioPlot(h_sum, truth_, "diffsig");   

        rp -> Draw();
        h_sum -> GetXaxis() -> SetTitle("Invariant Mass (GeV)"); 
        h_sum -> GetXaxis() -> CenterTitle("Invariant Mass (GeV)"); 

        rp -> GetLowerRefYaxis() -> SetTitle("Ratio"); 
        rp -> GetUpperRefYaxis() -> SetTitle("Entries (Arb.)"); 
        rp -> GetUpperPad() -> cd(); 
        legend -> Draw();

        if (this -> m_settings.logy){gPad -> SetLogy();}
        gPad -> Modified(); 
        can -> Modified();
        can -> Update();
            
        std::string path_ = out_pth + ith -> first;
        path_ += "/fold_" + std::to_string(k+1) + "/";
        path_ += "epoch_" + std::to_string(an -> this_epoch+1) + ".png"; 
        this -> create_path(path_); 
        can -> SaveAs(path_.c_str()); 
        can -> Close();

        delete rp; 
        delete h_sum; 
        delete truth_; 
        delete legend; 
        delete can; 

        for (size_t x(0); x < hists_pred[ith -> first].size(); ++x){
            hists_pred[ith -> first][x] -> Reset();
        }
    }
}

void metrics::build_th1f_mass(std::string var_name, graph_enum typ, int kfold){
   
    analytics_t* an = &this -> registry[kfold]; 

    std::string title = ""; 
    std::map<mode_enum, std::map<std::string, TH1F*>>* type_ = nullptr; 
    switch(typ){
        case graph_enum::truth_edge: type_ = &an -> truth_mass_edge; title = "truth"; break; 
        case graph_enum::data_edge:  type_ = &an -> pred_mass_edge;  title = "predicted"; break; 
        default: return; 
    }

    int nb = this -> m_settings.nbins; 
    int mxr = this -> m_settings.max_range;
    std::string base_name = this -> m_settings.run_name + var_name + " - " + title; 

    std::string tr = base_name + " (training) Mass " + std::to_string(kfold+1) + "-fold"; 
    (*type_)[mode_enum::training][var_name] = new TH1F(tr.c_str(), "training", nb, 0, mxr); 

    std::string va = base_name + " (validation) Mass" + std::to_string(kfold+1) + "-fold"; 
    (*type_)[mode_enum::validation][var_name] = new TH1F(va.c_str(), "validation", nb, 0, mxr); 

    std::string ev = base_name + " (evaluation) Mass" + std::to_string(kfold+1) + "-fold"; 
    (*type_)[mode_enum::evaluation][var_name] = new TH1F(ev.c_str(), "evaluation", nb, 0, mxr); 
}

void metrics::add_th1f_mass(
        torch::Tensor* pmc, torch::Tensor* edge_index, 
        torch::Tensor* truth, torch::Tensor* pred, 
        int kfold, mode_enum mode, std::string var_name
){

    analytics_t* an = &this -> registry[kfold]; 
    std::map<mode_enum, std::map<std::string, TH1F*>>* type_ = nullptr;
    torch::Tensor edge_index_ = edge_index -> to(torch::kLong); 
    torch::Tensor pred_ = pred -> clone();
    torch::Tensor pmc_  = pmc -> clone();  

    std::vector<double> v; 
    torch::Dict<std::string, torch::Tensor> pred_mass = pyc::graph::edge_aggregation(edge_index_, pred_, pmc_); 
    pred_mass = pyc::graph::unique_aggregation(pred_mass.at("cls::1::node-indices"), pmc_); 
    torch::Tensor pred_mass_cu = pred_mass.at("node-sum").index({(pred_mass.at("unique") > -1).sum({-1}) > 1}); 
    if (pred_mass_cu.size({0})){
        pred_mass_cu = pyc::physics::cartesian::combined::M(pred_mass_cu); 
        tensor_to_vector(&pred_mass_cu, &v); 
    }
    type_ = &an -> pred_mass_edge; 
    for (size_t x(0); x < v.size(); ++x){(*type_)[mode][var_name] -> Fill(v[x]);} 


    torch::Tensor truth_t = truth -> view({-1}); 
    torch::Tensor truth_ = torch::zeros_like(pred_); 
    for (signed int x(0); x < pred_.size({-1}); ++x){truth_.index_put_({truth_t == x, x}, 1);}

    v.clear(); 
    torch::Dict<std::string, torch::Tensor> truth_mass = pyc::graph::edge_aggregation(edge_index_, truth_, pmc_); 
    truth_mass = pyc::graph::unique_aggregation(truth_mass.at("cls::1::node-indices"), pmc_); 
    torch::Tensor truth_mass_cu = truth_mass.at("node-sum").index({(truth_mass.at("unique") > -1).sum({-1}) > 1}); 
    if (truth_mass_cu.size({0})){
        truth_mass_cu = pyc::physics::cartesian::combined::M(truth_mass_cu);
        tensor_to_vector(&truth_mass_cu, &v); 
    }
    type_ = &an -> truth_mass_edge; 
    for (size_t x(0); x < v.size(); ++x){(*type_)[mode][var_name] -> Fill(v[x]);} 
} 


