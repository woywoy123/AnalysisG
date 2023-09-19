#include "../abstractions/cytypes.h"

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace Optimizer
{
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

            std::vector<std::vector<std::string>> fetch_train(int kfold, int batch_size);
            std::vector<std::string> check_train(std::vector<std::string> hashes, int kfold);
            void flush_train(std::vector<std::string> hashes, int kfold);

            std::vector<std::vector<std::string>> fetch_validation(int kfold, int batch_size);
            std::vector<std::string> check_validation(std::vector<std::string> hashes, int kfold);
            void flush_validation(std::vector<std::string> hashes, int kfold);

            std::map<int, CyFold*> leaveout_test = {}; 
            std::map<int, CyFold*> kfold_eval = {};
            std::map<int, CyFold*> kfold_train = {};
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

            void delete_folds(std::map<int, CyFold*>* fold)
            {
                std::map<int, CyFold*>::iterator itr; 
                itr = fold -> begin(); 
                for (; itr != fold -> end(); ++itr){
                    delete itr -> second; 
                }
            }; 

            std::vector<std::string> check_data(
                    std::vector<std::string>* hashes, 
                    std::map<int, CyFold*>* fold, int kfold)
            {
                if (!fold -> count(kfold)){ return {}; }
                CyFold* f = fold -> at(kfold); 
                return f -> check_this(*hashes); 
            }; 




    }; 

}
#endif
