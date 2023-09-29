#include "../plotting/plotting.h"

CyPlotting::CyPlotting(){}
CyPlotting::~CyPlotting(){}

CyMetric::CyMetric(){}
CyMetric::~CyMetric(){}

void CyMetric::AddMetric(std::map<std::string, metric_t>* values, std::string mode)
{
    int c_epoch = this -> current_epoch; 
    if (this -> epoch_data.count(c_epoch)){}
    else { this -> epoch_data[c_epoch].initialize(c_epoch); }

    epoch_t* this_epoch = &(this -> epoch_data[c_epoch]);
    std::map<std::string, metric_t>::iterator k_itr = values -> begin(); 
    for (; k_itr != values -> end(); ++k_itr){
        this_epoch -> add(mode, k_itr -> first, &k_itr -> second); 
    }
}

std::vector<roc_t*> CyMetric::FetchROC()
{
    epoch_t* epoch = &(this -> epoch_data[this -> current_epoch]); 
    return epoch -> release_roc();
}

void CyMetric::dress_abstraction(abstract_plot* inpt,
                std::string title, std::string xtitle, std::string ytitle, 
                std::string fname, bool hist, bool line, int n_events)
{
    inpt -> cosmetic.n_events = n_events; 
    inpt -> cosmetic.atlas_style = true; 
    inpt -> cosmetic.atlas_year = 2016;
    inpt -> cosmetic.atlas_com = 13; 
    inpt -> figure.title = title; 
    inpt -> figure.histogram = hist; 
    inpt -> figure.line = line; 
    inpt -> file.filename = fname;
    inpt -> file.outputdir = this -> outpath; 
    inpt -> x.title = xtitle; 
    inpt -> y.title = ytitle; 
}


void CyMetric::BuildNodes(abstract_plot* plt, epoch_t* inpt)
{
    std::tuple<bool, node_t> tr = inpt -> get_nodes("training", "summary"); 
    if (!std::get<0>(tr)){}
    else {std::get<1>(tr).collect(&plt -> stacked["training"]);}

    std::tuple<bool, node_t> va = inpt -> get_nodes("validation", "summary"); 
    if (!std::get<0>(va)){}
    else {std::get<1>(va).collect(&plt -> stacked["validation"]);}

    std::tuple<bool, node_t> te = inpt -> get_nodes("evaluation", "summary"); 
    if (!std::get<0>(te)){}
    else {std::get<1>(te).collect(&plt -> stacked["evaluation"]);}
}

void CyMetric::BuildLoss(std::vector<abstract_plot>* out, std::vector<int> these)
{
    atomic_t atm;
    std::map<std::string, stats_t>::iterator itr; 
    std::map<std::string, std::vector<stats_t>> loss_tr;
    std::map<std::string, std::vector<stats_t>> loss_va;
    std::map<std::string, std::vector<stats_t>> loss_te;
    for (int ep : these){
        epoch_t* ep_ = &(this -> epoch_data[ep]); 
        atm = std::get<1>(ep_ -> get_atomic("training", "summary"));
        itr = atm.loss.begin();
        for (; itr != atm.loss.end(); itr++){
            loss_tr[itr -> first].push_back(itr -> second);
            this -> report.loss_train[itr -> first] = itr -> second.average; 
            this -> report.loss_train_up[itr -> first] = itr -> second.up; 
            this -> report.loss_train_down[itr -> first] = itr -> second.down;  
        }

        atm = std::get<1>(ep_ -> get_atomic("validation", "summary"));
        itr = atm.loss.begin();
        for (; itr != atm.loss.end(); itr++){
            loss_va[itr -> first].push_back(itr -> second); 
            this -> report.loss_valid[itr -> first] = itr -> second.average; 
            this -> report.loss_valid_up[itr -> first] = itr -> second.up; 
            this -> report.loss_valid_down[itr -> first] = itr -> second.down; 
        }

        atm = std::get<1>(ep_ -> get_atomic("evaluation", "summary"));
        itr = atm.loss.begin();
        for (; itr != atm.loss.end(); itr++){
            loss_te[itr -> first].push_back(itr -> second); 
            this -> report.loss_eval[itr -> first] = itr -> second.average; 
            this -> report.loss_eval_up[itr -> first] = itr -> second.up; 
            this -> report.loss_eval_down[itr -> first] = itr -> second.down; 
        }
    };
    
    std::map<std::string, std::vector<stats_t>>::iterator its = loss_tr.begin();
    for (; its != loss_tr.end(); its++){
        std::string var_name = its -> first; 
        std::string xTitle = "Epoch"; 
        std::string yTitle  = "Loss (Arb.)";
        abstract_plot plt; 
        this -> dress_abstraction(&plt, var_name, xTitle, yTitle, "/loss/" + var_name, false, true, -1);
        
        std::vector<stats_t> train = its -> second; 
        for (unsigned int x = 0; x < train.size(); ++x){
            stats_t point = train[x]; 
            std::string epoch = Tools::ToString(these[x]); 
            plt.stacked["training"].random_data.push_back(point.average); 
            plt.stacked["training"].random_data_up.push_back(point.up-point.average); 
            plt.stacked["training"].random_data_down.push_back(point.average - point.down); 
            plt.x.label_data[epoch] = these[x]; 
        }

        std::vector<stats_t> valid = loss_va[var_name]; 
        for (unsigned int x = 0; x < valid.size(); ++x){
            stats_t point = valid[x]; 
            plt.stacked["validation"].random_data.push_back(point.average); 
            plt.stacked["validation"].random_data_up.push_back(point.up - point.average); 
            plt.stacked["validation"].random_data_down.push_back(point.average - point.down); 
        }

        std::vector<stats_t> eval = loss_te[var_name]; 
        for (unsigned int x = 0; x < eval.size(); ++x){
            stats_t point = eval[x]; 
            plt.stacked["evaluation"].random_data.push_back(point.average); 
            plt.stacked["evaluation"].random_data_up.push_back(point.up-point.average); 
            plt.stacked["evaluation"].random_data_down.push_back(point.average - point.down); 
        }

        out -> push_back(plt); 
    }
}

void CyMetric::BuildAccuracy(std::vector<abstract_plot>* out, std::vector<int> these)
{
    atomic_t atm;
    std::map<std::string, stats_t>::iterator itr; 
    std::map<std::string, std::vector<stats_t>> acc_tr;
    std::map<std::string, std::vector<stats_t>> acc_va;
    std::map<std::string, std::vector<stats_t>> acc_te;
    for (int ep : these){
        epoch_t* ep_ = &(this -> epoch_data[ep]); 
        atm = std::get<1>(ep_ -> get_atomic("training", "summary"));
        itr = atm.accuracy.begin();
        for (; itr != atm.accuracy.end(); itr++){
            acc_tr[itr -> first].push_back(itr -> second); 
            this -> report.acc_train[itr -> first] = itr -> second.average; 
            this -> report.acc_train_up[itr -> first] = itr -> second.up; 
            this -> report.acc_train_down[itr -> first] = itr -> second.down;  
        }

        atm = std::get<1>(ep_ -> get_atomic("validation", "summary"));
        itr = atm.accuracy.begin();
        for (; itr != atm.accuracy.end(); itr++){
            acc_va[itr -> first].push_back(itr -> second); 
            this -> report.acc_valid[itr -> first] = itr -> second.average; 
            this -> report.acc_valid_up[itr -> first] = itr -> second.up; 
            this -> report.acc_valid_down[itr -> first] = itr -> second.down;  
        }

        atm = std::get<1>(ep_ -> get_atomic("evaluation", "summary"));
        itr = atm.accuracy.begin();
        for (; itr != atm.accuracy.end(); itr++){
            acc_te[itr -> first].push_back(itr -> second); 
            this -> report.acc_eval[itr -> first] = itr -> second.average; 
            this -> report.acc_eval_up[itr -> first] = itr -> second.up; 
            this -> report.acc_eval_down[itr -> first] = itr -> second.down;
        }
        this -> report.current_epoch = ep; 
    };

    std::map<std::string, std::vector<stats_t>>::iterator its = acc_tr.begin();
    for (; its != acc_tr.end(); its++){
        std::string var_name = its -> first; 
        std::string xTitle = "Epoch"; 
        std::string yTitle  = "Accuracy (percentage)";

        abstract_plot plt; 
        this -> dress_abstraction(&plt, var_name, xTitle, yTitle, "/accuracy/" + var_name, false, true, -1);
        
        std::vector<stats_t> train = its -> second; 
        for (unsigned int x = 0; x < train.size(); ++x){
            stats_t point = train[x]; 
            std::string epoch = Tools::ToString(these[x]); 
            plt.stacked["training"].random_data.push_back(point.average); 
            plt.stacked["training"].random_data_up.push_back(point.up - point.average); 
            plt.stacked["training"].random_data_down.push_back(point.average - point.down); 
            plt.x.label_data[epoch] = these[x]; 
        }

        std::vector<stats_t> valid = acc_va[var_name]; 
        for (unsigned int x = 0; x < valid.size(); ++x){
            stats_t point = valid[x]; 
            plt.stacked["validation"].random_data.push_back(point.average); 
            plt.stacked["validation"].random_data_up.push_back(point.up - point.average); 
            plt.stacked["validation"].random_data_down.push_back(point.average - point.down); 
        }

        std::vector<stats_t> eval = acc_te[var_name]; 
        for (unsigned int x = 0; x < eval.size(); ++x){
            stats_t point = eval[x]; 
            plt.stacked["evaluation"].random_data.push_back(point.average); 
            plt.stacked["evaluation"].random_data_up.push_back(point.up - point.average); 
            plt.stacked["evaluation"].random_data_down.push_back(point.average - point.down); 
        }

        out -> push_back(plt); 
    }
}

void CyMetric::BuildROC(std::vector<abstract_plot>* out, std::vector<int> these)
{
    atomic_t atm;
    std::map<std::string, roc_t>::iterator itr; 
    std::map<std::string, std::vector<roc_t>>::iterator it_roc; 
    std::map<std::string, std::vector<roc_t>> rc_tr;
    std::map<std::string, std::vector<roc_t>> rc_va;
    std::map<std::string, std::vector<roc_t>> rc_te;
    for (int ep : these){
        epoch_t* ep_ = &(this -> epoch_data[ep]); 
        atm = std::get<1>(ep_ -> get_atomic("training", "summary"));
        itr = atm.roc_curve.begin();
        for (; itr != atm.roc_curve.end(); itr++){
            rc_tr[itr -> first].push_back(itr -> second); 
        }

        atm = std::get<1>(ep_ -> get_atomic("validation", "summary"));
        itr = atm.roc_curve.begin();
        for (; itr != atm.roc_curve.end(); itr++){
            rc_va[itr -> first].push_back(itr -> second); 
        }

        atm = std::get<1>(ep_ -> get_atomic("evaluation", "summary"));
        itr = atm.roc_curve.begin();
        for (; itr != atm.roc_curve.end(); itr++){
            rc_te[itr -> first].push_back(itr -> second); 
        }
    };
  
    it_roc = rc_tr.begin(); 
    for (; it_roc != rc_tr.end(); ++it_roc){
        std::string var_name = it_roc -> first; 
        std::map<std::string, abstract_plot> auc_d;  
        for (unsigned int x=0; x < these.size(); ++x){
            roc_t train = rc_tr[var_name][x]; 
            if (!rc_va.count(var_name)){continue;}
            roc_t valid = rc_va[var_name][x]; 
            if (!rc_te.count(var_name)){continue;}
            roc_t eval  = rc_te[var_name][x]; 
            std::map<int, std::vector<float>>::iterator cls;
            for (cls = train.tpr.begin(); cls != train.tpr.end(); ++cls){
                std::string title = "Receiver-Operator-Curve ("; 
                title += var_name + ") @ Epoch: ";
                title += Tools::ToString(these[x]);
                title += " for classification: " + Tools::ToString(cls -> first);  

                std::string fname = var_name;
                fname += "/roc/class-" + Tools::ToString(cls -> first);
                fname += "-epoch-" + Tools::ToString(these[x]); 

                std::string xtitle = "False Positive Rate";
                std::string ytitle = "True Positive Rate"; 

                abstract_plot plt;
                this -> dress_abstraction(&plt, title, xtitle, ytitle, fname, false, true, -1);
                plt.x.sorted_data["training"] = train.tpr[cls -> first]; 
                plt.y.sorted_data["training"] = train.fpr[cls -> first];  

                plt.x.sorted_data["validation"] = valid.tpr[cls -> first]; 
                plt.y.sorted_data["validation"] = valid.fpr[cls -> first];  

                plt.x.sorted_data["evaluation"] = eval.tpr[cls -> first]; 
                plt.y.sorted_data["evaluation"] = eval.fpr[cls -> first];  
                out -> push_back(plt); 
                    
                abstract_plot* auc = &auc_d["class-" + Tools::ToString(cls -> first)]; 
                auc -> stacked["training"].random_data.push_back(train.auc[cls -> first]); 
                auc -> stacked["validation"].random_data.push_back(valid.auc[cls -> first]); 
                auc -> stacked["evaluation"].random_data.push_back(eval.auc[cls -> first]); 
                auc -> x.label_data[Tools::ToString(x)] = these[x]; 
               
            
                std::string col =  var_name + " :: class - " + Tools::ToString(cls -> first); 
                this -> report.auc_train[col] = train.auc[cls -> first];  
                this -> report.auc_valid[col] = valid.auc[cls -> first];  
                this -> report.auc_eval[col] = eval.auc[cls -> first];  
            }

            std::string ctitle = "Confusion Matrix ("; 
            ctitle += var_name + ") @ Epoch: ";
            ctitle += Tools::ToString(these[x]);
            std::string cfname = var_name;
            cfname += "/confusion/";
            cfname += "epoch-" + Tools::ToString(these[x]); 
            std::string xtitle = "Predicted Classification"; 
            std::string ytitle = "Target Classification"; 

            abstract_plot plt;
            this -> dress_abstraction(&plt, ctitle, xtitle, ytitle, cfname, true, false, -1); 
            for (cls = train.tpr.begin(); cls != train.tpr.end(); ++cls){
                std::string name = Tools::ToString(cls -> first);
                std::vector<int> conf; 

                conf = train.confusion[cls -> first]; 
                plt.stacked["training"].sorted_data[name] = std::vector<float>(conf.begin(), conf.end()); 
                
                conf = valid.confusion[cls -> first]; 
                plt.stacked["validation"].sorted_data[name] = std::vector<float>(conf.begin(), conf.end()); 

                conf = eval.confusion[cls -> first]; 
                plt.stacked["evaluation"].sorted_data[name] = std::vector<float>(conf.begin(), conf.end()); 

            } 
            out -> push_back(plt); 
        }

       
        std::map<std::string, abstract_plot>::iterator itb = auc_d.begin(); 
        for (; itb != auc_d.end(); ++itb){
            abstract_plot auc = itb -> second; 

            std::string ctitle = "Area Under Curve (" + var_name + ") for " + itb -> first;
            std::string cfname = "/auc/" + var_name + "/" + itb -> first;
            std::string xtitle = "Epoch"; 
            std::string ytitle = "Area Under Curve"; 

            this -> dress_abstraction(&auc, ctitle, xtitle, ytitle, cfname, false, true, -1); 
            out -> push_back(auc); 
        }
    }
}


void CyMetric::BuildPlots(std::map<std::string, abstract_plot>* output)
{
    std::map<std::string, abstract_plot> tmp = {}; 
    std::map<int, epoch_t>* ep = &(this -> epoch_data); 

    std::vector<int> these_eps = {}; 
    std::map<int, epoch_t>::iterator itr = ep -> begin();
    for (; itr != ep -> end(); ++itr){
        these_eps.push_back(itr -> first); 
    }
    std::sort(these_eps.begin(), these_eps.end()); 
    if (!these_eps.size()){return;} 

    // Get the node data first
    abstract_plot nodes; 
    dress_abstraction(
            &nodes, "Node Distributions of Sample", 
            "Number of Nodes", "Number of Entries", 
            "NodeStatistics", true, false, -1); 

    this -> BuildNodes(&nodes, &((*ep)[these_eps[0]])); 
    (*output)["NodeStats"] = nodes; 

    std::vector<abstract_plot> losses; 
    this -> BuildLoss(&losses, these_eps); 
    for (unsigned int x(0); x < losses.size(); ++x){
        (*output)["loss-" + losses[x].file.filename] = losses[x]; 
    }

    std::vector<abstract_plot> accuracy; 
    this -> BuildAccuracy(&accuracy, these_eps); 
    for (unsigned int x(0); x < accuracy.size(); ++x){
        (*output)["acc-" + accuracy[x].file.filename] = accuracy[x]; 
    }

    std::vector<abstract_plot> roc_curves; 
    this -> BuildROC(&roc_curves, these_eps);
    for (unsigned int x(0); x < roc_curves.size(); ++x){
        (*output)["roc-" + roc_curves[x].file.filename] = roc_curves[x]; 
    }
}
