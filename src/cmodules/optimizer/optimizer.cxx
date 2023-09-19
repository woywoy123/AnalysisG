#include "../abstractions/abstractions.h"
#include "../optimizer/optimizer.h"

namespace Optimizer
{
    CyFold::CyFold(){}
    CyFold::~CyFold(){}
    void CyFold::Import(const folds_t* in)
    {
        this -> cached_hashes[in -> event_hash] = false; 
    }

    int CyFold::length()
    {
        return this -> cached_hashes.size();
    }

    std::vector<std::vector<std::string>> CyFold::fetchthis(int batch_size)
    {
        std::vector<std::string> output = {}; 
        std::map<std::string, bool>* cached = &(this -> cached_hashes);
        std::map<std::string, bool>::iterator itr;

        for (itr = cached -> begin(); itr != cached -> end(); ++itr){
            output.push_back(itr -> first); 
        }
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
}

