#include "../../Templates/tools/tools.h"
#include "metadata.h"

namespace SampleTracer
{
    CyMetaData::CyMetaData(){}
    CyMetaData::~CyMetaData(){}

    void CyMetaData::addsamples(int index, int range, std::string sample)
    {
        this -> inputfiles[index] = sample; 
        this -> inputrange[index] = range; 
    }

    void CyMetaData::addconfig(std::string key, std::string val)
    {
        this -> config[key] = val; 
    }

    void CyMetaData::hashing()
    {
        this -> hash = Hashing(ToString(this -> dsid)); 
    }
    
    std::string CyMetaData::IndexToSample(int index)
    {
        std::map<int, std::string>::iterator it; 
        std::map<int, std::string>* x = &(this -> inputfiles); 
        for (it = x -> begin(); it != x -> end(); ++it)
        {
            if (index >= this -> inputrange[it -> first]){continue;}
            return it -> second;
        }
        return ""; 
    }

    void CyMetaData::processkeys(std::vector<std::string> keys, unsigned int num)
    {      
        auto scanthis = [](
                std::vector<std::string>* req, std::vector<std::string>* tokens,
                std::vector<std::string>* res, std::string key)
        {
            for (unsigned int i(0); i < req -> size(); ++i)
            {
                std::string tmp = req -> at(i); 
                if (!count(key, tmp)){ continue; }
                for (unsigned x(0); x < tokens -> size(); ++x)
                {
                    std::string tok = tokens -> at(x);
                    if (!count(tok, tmp + ";") && tmp != tok){continue;}
                    res -> push_back(tmp); 
                    res -> push_back(tok); 
                    return; 
                }
            }
        }; 
            
        auto search = [&scanthis](
                std::vector<std::string>* r_tree, 
                std::vector<std::string>* r_branch, 
                std::vector<std::string>* r_leaf, 
                std::vector<std::string> keys, std::vector<Collect>* tree)
        {
            for (unsigned int i(0); i < keys.size(); ++i)
            {
                std::vector<std::string> _tree = {}; 
                std::vector<std::string> _branch = {}; 
                std::vector<std::string> _leaf = {};  
                
                std::vector<std::string> tokens = split(keys.at(i), "/");
                scanthis(r_tree  , &tokens, &_tree  , keys.at(i));      
                scanthis(r_branch, &tokens, &_branch, keys.at(i));        
                scanthis(r_leaf  , &tokens, &_leaf  , keys.at(i)); 

                bool lx = _leaf.size() > 0; 
                bool bx = _branch.size() > 0; 
                bool tx = _tree.size() > 0; 

                Collect* tr = &(tree -> at(i));
                tr -> valid = lx + bx + tx; 
                tr ->  lf_requested = (lx) ? _leaf[0]   : ""; 
                tr ->  lf_matched   = (lx) ? _leaf[1]   : ""; 
                tr ->  lf_path      = (lx) ? join(&tokens, 1, -1, "/") : ""; 
 
                tr ->  br_requested = (bx) ? _branch[0] : ""; 
                tr ->  br_matched   = (bx) ? _branch[1] : ""; 
                
                tr ->  tr_requested = (tx) ? _tree[0] : ""; 
                tr ->  tr_matched   = (tx) ? _tree[1] : ""; 

                tokens.clear();
            }
        }; 

        std::vector<std::string>* tr = &(this -> req_trees); 
        std::vector<std::string>* br = &(this -> req_branches); 
        std::vector<std::string>* lf = &(this -> req_leaves); 

        std::vector<std::vector<std::string>> quant = Quantize(keys, this -> chunks); 
        std::vector<std::vector<Collect>*> check = {}; 
        std::vector<std::thread*> jbs = {};

        for (unsigned int i(0); i < quant.size(); ++i)
        {
            Collect x;
            std::vector<Collect>* f = new std::vector<Collect>(quant[i].size(), x);
            check.push_back(f);
            
            search(tr, br, lf, quant[i], f); 
            std::thread* p = new std::thread(search, tr, br, lf, quant[i], f);
            jbs.push_back(p); 
        }
        for (std::thread* t : jbs){ t -> join(); delete t; }
        jbs = {}; 
        for (unsigned int i(0); i < check.size(); ++i)
        {
            std::vector<Collect> vec = *check[i]; 
            for (unsigned int x(0); x < vec.size(); ++x)
            {
                Collect* col = &vec[x]; 
                std::string tr_get = col -> tr_requested; 
                std::string br_get = col -> br_requested; 
                std::string lf_get = col -> lf_requested; 
                if (!col -> valid){continue;}

                if (!this -> trees.count(tr_get) && tr_get.size())
                {
                    Tree tmp; 
                    tmp.size = num;  
                    tmp.requested = col -> tr_requested; 
                    tmp.matched   = col -> tr_matched; 
                    this -> trees[tr_get] = tmp; 
                }
                
                if (!this -> branches.count(tr_get) && br_get.size())
                {
                    Branch tmp; 
                    tmp.requested = col -> br_requested; 
                    tmp.matched   = col -> br_matched; 
                    this -> branches[br_get] = tmp; 
                }

                if (!this -> leaves.count(lf_get) && lf_get.size())
                {
                    Leaf tmp; 
                    tmp.requested = col -> lf_requested; 
                    tmp.matched   = col -> lf_matched; 
                    tmp.path      = col -> lf_path; 
                    this -> leaves[lf_get] = tmp; 
                }


                // Do the linking here 
                if (lf_get.size())
                {
                    Leaf* lfc = &leaves[lf_get]; 
                    if (br_get.size()){ this -> branches[br_get].leaves.push_back(lfc); }
                    if (tr_get.size()){ this ->    trees[tr_get].leaves.push_back(lfc); }
                    lfc -> branch_name = br_get; 
                    lfc -> tree_name = tr_get; 
                }

                if (br_get.size())
                {
                    Branch* brc = &branches[br_get]; 
                    if (tr_get.size()){ this -> trees[tr_get].branches.push_back(brc); }
                    brc -> tree_name = tr_get; 
                }
            }
            delete check[i]; 
        }
    }

    void CyMetaData::FindMissingKeys()
    {
        std::vector<std::string> r_trees = this -> req_trees; 
        std::vector<std::string> r_branches = this -> req_branches; 
        std::vector<std::string> r_leaves = this -> req_leaves; 
       
        for (unsigned int x(0); x < r_trees.size(); ++x)
        {
            std::string tree = r_trees[x]; 
            if (this -> trees.count(tree)){continue;}
            this -> mis_trees.push_back(tree);  
        }

        for (unsigned int x(0); x < r_branches.size(); ++x)
        {
            std::string branch = r_branches[x]; 
            if (this -> branches.count(branch)){continue;}
            this -> mis_branches.push_back(branch);  
        }

        for (unsigned int x(0); x < r_leaves.size(); ++x)
        {
            std::string leaf = r_leaves[x]; 
            if (this -> leaves.count(leaf)){continue;}
            this -> mis_leaves.push_back(leaf);  
        }
        
        this -> req_trees = {}; 
        std::map<std::string, Tree>::iterator it_tr = this -> trees.begin(); 
        for (; it_tr != this -> trees.end(); ++it_tr)
        {
            this -> req_trees.push_back(it_tr -> first); 
        }

        this -> req_branches = {}; 
        std::map<std::string, Branch>::iterator it_br = this -> branches.begin();
        for (; it_br != this -> branches.end(); ++it_br)
        {
            this -> req_branches.push_back(it_br -> first); 
        }

        this -> req_leaves = {}; 
        std::map<std::string, Leaf>::iterator it_lf = this -> leaves.begin();
        for (; it_lf != this -> leaves.end(); ++it_lf)
        {
            this -> req_leaves.push_back(it_lf -> first); 
        }
    }

    std::map<std::string, int> CyMetaData::GetLength()
    {
        std::map<std::string, Tree>::iterator it; 
        std::map<std::string, int> index = {}; 
        for (it = this -> trees.begin(); it != this -> trees.end(); ++it)
        {
            index[it -> first] = it -> second.size; 
        }
        return index; 
    }

    std::map<std::string, std::vector<std::string>> CyMetaData::MakeGetter()
    {
        std::map<std::string, Tree> trees = this -> trees;  
        std::map<std::string, Tree>::iterator it; 
        std::map<std::string, std::vector<std::string>> output = {}; 
        for (it = trees.begin(); it != trees.end(); ++it)
        {
            output[it -> first] = {}; 
            for (unsigned int x(0); x < it -> second.leaves.size(); ++x)
            {
                Leaf* _lf = it -> second.leaves[x]; 
                std::string get = _lf -> path; 
                output[it -> first].push_back(get); 
            }
        }
        return output; 
    }
}
