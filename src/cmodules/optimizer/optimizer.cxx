#include "../abstractions/abstractions.h"
#include "../optimizer/optimizer.h"

namespace Optimizer
{
    CyEpoch::CyEpoch(){}
    CyEpoch::~CyEpoch(){} 

    void CyEpoch::add_epoch(int epoch){this -> epoch = epoch;}
    void CyEpoch::add_kfold(int fold, std::map<std::string, data_t>* var)
    {
        if (!this -> container.count(fold)){
            this -> container[fold] = *var;
            return;
        }
        std::map<std::string, data_t>* map = &(this -> container[fold]); 
        std::map<std::string, data_t>::iterator itr = var -> begin(); 
        for (; itr != var -> end(); ++itr){
            if (!map -> count(itr -> first)){
                (*map)[itr -> first] = itr -> second;
                continue;
            }
            data_t* self = &((*map)[itr -> first]); 
            data_t* other = &(itr -> second); 
            this -> merge_this(&(other -> truth),    &(self -> truth));  
            this -> merge_this(&(other -> pred),     &(self -> pred)); 
            this -> merge_this(&(other -> index),    &(self -> index)); 

            this -> merge_this(&(other -> nodes),    &(self -> nodes)); 
            this -> merge_this(&(other -> loss),     &(self -> loss)); 
            this -> merge_this(&(other -> accuracy), &(self -> accuracy)); 
        }
    } 

    std::map<std::string, metric_t> CyEpoch::metrics()
    {
        std::map<std::string, metric_t> output; 
        std::map<std::string, data_t> tmp; 
        std::map<std::string, data_t>::iterator itd;

        std::map<int, std::map<std::string, data_t>>::iterator itr;
        itr = this -> container.begin();
        output["kfold-summary"] = metric_t();
        for (; itr != this -> container.end(); ++itr){
            std::string name = "kfold-" + Tools::ToString(itr -> first);
            output[name] = metric_t(); 
            std::map<std::string, data_t>* data_map = &(itr -> second);
            itd = data_map -> begin(); 
            for (; itd != data_map -> end(); ++itd){
                data_t* atom_d = &(itd -> second);
                std::string var_n = itd -> first; 

                if (tmp.count(var_n)){}
                else {tmp[var_n] = data_t();}
                tmp[var_n].name = var_n; 

                this -> merge_this(&atom_d -> loss    , &(tmp[var_n].loss));  
                this -> merge_this(&atom_d -> accuracy, &(tmp[var_n].accuracy));  
                this -> merge_this(&atom_d -> truth   , &(tmp[var_n].truth)); 
                this -> merge_this(&atom_d -> pred    , &(tmp[var_n].pred)); 

                this -> merge_this(&atom_d -> truth   , &(output[name].truth[var_n])); 
                this -> merge_this(&atom_d -> pred    , &(output[name].pred[var_n])); 

                this -> make_average(&output[name].acc_average , &atom_d -> accuracy, var_n); 
                this -> make_average(&output[name].loss_average, &atom_d -> loss    , var_n);

                this -> make_nodes(&output[name], &itd -> second); 
                this -> make_nodes(&output["kfold-summary"], atom_d);
            }            
        }
        itd = tmp.begin();
        for(; itd != tmp.end(); ++itd){ 
            this -> make_average(&output["kfold-summary"].acc_average , &tmp[itd -> first].accuracy, itd -> first);
            this -> make_average(&output["kfold-summary"].loss_average, &tmp[itd -> first].loss, itd -> first);
            output["kfold-summary"].truth[itd -> first] = tmp[itd -> first].truth; 
            output["kfold-summary"].pred[itd -> first]  = tmp[itd -> first].pred; 
        }
        return output;
    }

    CyFold::CyFold(){}
    CyFold::~CyFold(){}
    int CyFold::length(){return this -> cached_hashes.size();}
    void CyFold::Import(const folds_t* in)
    {
        this -> cached_hashes[in -> event_hash] = false; 
    }

    std::vector<std::vector<std::string>> CyFold::fetchthis(int batch_size)
    {
        std::vector<std::string> output = {}; 
        std::map<std::string, bool>* cached = &(this -> cached_hashes);
        std::map<std::string, bool>::iterator itr = cached -> begin();
        for (; itr != cached -> end(); ++itr){output.push_back(itr -> first);}
        return Tools::Quantize(output, batch_size); 
    }

    void CyFold::flushthis(std::vector<std::string> inpt)
    {
        std::map<std::string, bool>* cached = &(this -> cached_hashes); 
        for (std::string hash : inpt){
            if (!cached -> count(hash)){continue;}
            (*cached)[hash] = false;
        }
    }

    std::vector<std::string> CyFold::check_this(std::vector<std::string> hashes)
    {
        std::vector<std::string> output; 
        std::map<std::string, bool>* cached = &(this -> cached_hashes); 
        for (std::string hash : hashes){
            if (!cached -> count(hash)){ continue; }
            if (cached -> at(hash)){ continue; }
            output.push_back(hash);
            (*cached)[hash] = true;
        }        
        return output; 
    }





    CyOptimizer::CyOptimizer(){}
    CyOptimizer::~CyOptimizer()
    {
        this -> delete_folds(&(this -> leaveout_test)); 
        this -> delete_folds(&(this -> kfold_eval)); 
        this -> delete_folds(&(this -> kfold_train)); 

        this -> delete_folds(&(this -> epoch_train)); 
        this -> delete_folds(&(this -> epoch_valid)); 
        this -> delete_folds(&(this -> epoch_test)); 
    }

    void CyOptimizer::register_fold(const folds_t* inpt)
    {
        std::map<int, CyFold*>* fold; 
        if (inpt -> test){ fold = &(this -> leaveout_test); }
        else if (inpt -> train){ fold = &(this -> kfold_train); }
        else if (inpt -> evaluation){ fold = &(this -> kfold_eval); }
        else {return;}

        if (fold -> count(inpt -> kfold)){}
        else { (*fold)[inpt -> kfold] = new CyFold(); }
        CyFold* this_fold = fold -> at(inpt -> kfold);
        this_fold -> Import(inpt);
    }

    std::map<std::string, int> CyOptimizer::fold_map()
    {
        std::map<int, CyFold*>* test = &(this -> leaveout_test); 
        std::map<int, CyFold*>::iterator itr = test -> begin();
        std::map<std::string, int> output = {}; 
        for (; itr != test -> end(); ++itr){
            std::string key = "leave-out" + Tools::ToString(itr -> first); 
            output[key] += itr -> second -> length();  
        } 

        std::map<int, CyFold*>* eval = &(this -> kfold_eval); 
        itr = eval -> begin();
        for (; itr != eval -> end(); ++itr){
            std::string key = "eval-k" + Tools::ToString(itr -> first); 
            output[key] += itr -> second -> length();  
        } 

        std::map<int, CyFold*>* train = &(this -> kfold_train); 
        itr = train -> begin();
        for (; itr != train -> end(); ++itr){
            std::string key = "train-k" + Tools::ToString(itr -> first); 
            output[key] += itr -> second -> length();  
        }
        return output;  
    }  

    std::vector<std::vector<std::string>> CyOptimizer::fetch_train(int kfold, int batch_size)
    {
        return this -> fetch_quant(kfold, batch_size, &(this -> kfold_train)); 
    }

    void CyOptimizer::flush_train(std::vector<std::string> hashes, int kfold)
    {
        this -> flush_data(&hashes, &(this -> kfold_train), kfold); 
    }

    std::vector<std::string> CyOptimizer::check_train(std::vector<std::string> hashes, int kfold)
    {
        return this -> check_data(&hashes, &(this -> kfold_train), kfold);
    }

    std::vector<std::vector<std::string>> CyOptimizer::fetch_validation(int kfold, int batch_size)
    {
        return this -> fetch_quant(kfold, batch_size, &(this -> kfold_eval)); 
    }

    void CyOptimizer::flush_validation(std::vector<std::string> hashes, int kfold)
    {
        this -> flush_data(&hashes, &(this -> kfold_eval), kfold); 
    }

    std::vector<std::string> CyOptimizer::check_validation(std::vector<std::string> hashes, int kfold)
    {
        return this -> check_data(&hashes, &(this -> kfold_eval), kfold);
    }

    std::vector<std::vector<std::string>> CyOptimizer::fetch_evaluation(int batch_size)
    {
        return this -> fetch_quant(-1, batch_size, &(this -> leaveout_test)); 
    }

    void CyOptimizer::flush_evaluation(std::vector<std::string> hashes)
    {
        this -> flush_data(&hashes, &(this -> leaveout_test), -1); 
    }

    std::vector<std::string> CyOptimizer::check_evaluation(std::vector<std::string> hashes)
    {
        return this -> check_data(&hashes, &(this -> leaveout_test), -1);
    }

    void CyOptimizer::train_epoch_kfold(int epoch, int kfold, std::map<std::string, data_t>* data)
    {
        CyEpoch* ep = this -> add_epoch(epoch, &(this -> epoch_train)); 
        ep -> add_kfold(kfold, data); 
    }

    void CyOptimizer::validation_epoch_kfold(int epoch, int kfold, std::map<std::string, data_t>* data)
    {
        CyEpoch* ep = this -> add_epoch(epoch, &(this -> epoch_valid)); 
        ep -> add_kfold(kfold, data); 
    }

    void CyOptimizer::evaluation_epoch_kfold(int epoch, int kfold, std::map<std::string, data_t>* data)
    {
        CyEpoch* ep = this -> add_epoch(epoch, &(this -> epoch_test)); 
        ep -> add_kfold(kfold, data); 
    }

}

