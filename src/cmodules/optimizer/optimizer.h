#include "../abstractions/abstractions.h"
#include "../abstractions/cytypes.h"
#include <cmath>

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace Optimizer
{
    class CyEpoch
    {
        public:
            CyEpoch();
            ~CyEpoch();
            void add_epoch(int epoch); 
            void add_kfold(int fold,  std::map<std::string, data_t>* var);
            std::map<std::string, data_t> release_kfold(int kfold);
            std::map<std::string, metric_t> metrics();

            int epoch = 0; 
            std::map<int, std::map<std::string, data_t>> container = {}; 

        private:
            void merge_this(
                    std::vector<std::vector<float>>* in, 
                    std::vector<std::vector<float>>* out)
            {
                out -> insert(out -> end(), in -> begin(), in -> end()); 
            };

            void make_average(
                    std::map<std::string, float>* out, 
                    std::vector<std::vector<float>>* vec, 
                    std::string name)
            {
                float sum = 0; 
                float yhat = 0; 
                float stdev = 0; 
                float up = 0; 
                float down = 0; 
                float len = vec -> size(); 
                for (unsigned int x = 0; x < len; ++x)
                {
                    float val = (vec -> at(x))[0]; 
                    sum += val; 
                    if (!x){ up = val; down = val;}
                    if (val > up){up = val;}
                    if (val < down){down = val;}
                }
                yhat = sum/len; 
                sum = 0; 
                for (unsigned int x = 0; x < len; ++x)
                {
                    float val = (vec -> at(x))[0]; 
                    sum += std::pow(val - yhat, 2); 
                }
                stdev = std::pow(sum/len, 0.5); 
                (*out)[name + "_average"] = yhat; 
                (*out)[name + "_stdev"] = stdev;
                (*out)[name + "_up"] = up; 
                (*out)[name + "_down"] = down; 
            }; 

            void make_nodes(metric_t* output, data_t* inpt)
            {
                for (unsigned int x = 0; x < inpt -> nodes.size(); ++x)
                {
                    unsigned int n = inpt -> nodes[x][0]; 
                    output -> num_nodes["nodes-" + Tools::ToString(n)] += 1; 
                }
            }; 

    };

    class CyFold
    {
        public:
            CyFold(); 
            ~CyFold();
            void Import(const folds_t*); 
            int length(); 
            std::vector<std::vector<std::string>> fetchthis(int batch_size);
            std::vector<std::string> check_this(std::vector<std::string> hashes);
            void flushthis(std::vector<std::string> inpt);
            std::map<std::string, bool> cached_hashes = {}; 
    };


    class CyOptimizer
    {
        public:
            CyOptimizer(); 
            ~CyOptimizer(); 
            void register_fold(const folds_t*);
            std::map<std::string, int> fold_map(); 

            void flush_train(std::vector<std::string> hashes, int kfold);
            void flush_validation(std::vector<std::string> hashes, int kfold);
            void flush_evaluation(std::vector<std::string> hashes);

            std::vector<std::vector<std::string>> fetch_train(int kfold, int batch_size);
            std::vector<std::vector<std::string>> fetch_validation(int kfold, int batch_size);
            std::vector<std::vector<std::string>> fetch_evaluation(int batch_size);
            
            std::vector<std::string> check_train(std::vector<std::string> hashes, int kfold);
            std::vector<std::string> check_validation(std::vector<std::string> hashes, int kfold);
            std::vector<std::string> check_evaluation(std::vector<std::string> hashes);

            void train_epoch_kfold(int epoch, int kfold, std::map<std::string, data_t>* data);
            void validation_epoch_kfold(int epoch, int kfold, std::map<std::string, data_t>* data);
            void evaluation_epoch_kfold(int epoch, int kfold, std::map<std::string, data_t>* data);

            std::map<int, CyFold*> kfold_eval    = {};
            std::map<int, CyFold*> kfold_train   = {};
            std::map<int, CyFold*> leaveout_test = {}; 

            std::map<int, CyEpoch*> epoch_train = {};
            std::map<int, CyEpoch*> epoch_valid  = {};
            std::map<int, CyEpoch*> epoch_test  = {}; 

            std::vector<int> use_folds = {};

        private:
            void flush_data(
                    std::vector<std::string>* hashes, 
                    std::map<int, CyFold*>* folds, int kfold)
            {
                if (!folds -> count(kfold)){return;}
                folds -> at(kfold) -> flushthis(*hashes); 
            };

            std::vector<std::vector<std::string>> fetch_quant(
                    int kfold, int batch_size, 
                    std::map<int, CyFold*>* folds)
            {
                if (!folds -> count(kfold)){ return {}; }
                CyFold* f = folds -> at(kfold); 
                return f -> fetchthis(batch_size); 
            }; 

            std::vector<std::string> check_data(
                    std::vector<std::string>* hashes, 
                    std::map<int, CyFold*>* fold, int kfold)
            {
                if (!fold -> count(kfold)){ return {}; }
                CyFold* f = fold -> at(kfold); 
                return f -> check_this(*hashes); 
            }; 

            CyEpoch* add_epoch(int epoch, std::map<int, CyEpoch*>* inpt)
            {
                if (!inpt -> count(epoch)){
                    (*inpt)[epoch] = new CyEpoch(); 
                    (*inpt)[epoch] -> add_epoch(epoch);
                }
                return inpt -> at(epoch); 
            }; 

            template <typename G>
            void delete_folds(std::map<int, G*>* fold)
            {
                typename std::map<int, G*>::iterator itr; 
                itr = fold -> begin(); 
                for (; itr != fold -> end(); ++itr){
                    delete itr -> second; 
                }
            }; 



    }; 

}
#endif
