#ifndef EVENT_H
#define EVENT_H
#include <iostream>
#include <map>

struct ExportEventTemplate 
{
    std::map<std::string, std::string> leaves; 
    std::map<std::string, std::string> branches; 
    std::map<std::string, std::string> trees;
 
    std::string event_tree; 
    std::string event_tagging; 
    std::string event_name; 
    std::string commit_hash; 
    std::string pickle_string; 
    std::string event_hash; 
    std::string pickled_data; 
    std::string ROOT; 

    double weight; 
    bool cached; 
    bool deprecated;

    int event_index; 
};

namespace CyTemplate
{
    class CyEventTemplate
    {
        public: 
            CyEventTemplate(); 
            ~CyEventTemplate(); 
            ExportEventTemplate MakeMapping(); 
            void ImportEventData(ExportEventTemplate event);

            double weight = 1; 
            int event_index = -1; 
            
            bool cached = false; 
            bool deprecated = false; 

            std::string event_tree = ""; 
            std::string event_tagging = ""; 
            std::string event_name = ""; 
            std::string commit_hash = ""; 
            std::string pickle_string = ""; 
            std::string ROOT = ""; 
            
            std::string Hash(); 
            void Hash(std::string input); 
            void addleaf(std::string key, std::string leaf); 
            void addbranch(std::string key, std::string branch); 
            void addtree(std::string key, std::string tree); 
            
            bool operator == (CyEventTemplate* ev);
        
            std::map<std::string, std::string> leaves = {}; 
            std::map<std::string, std::string> branches = {};  
            std::map<std::string, std::string> trees = {}; 

        private:
            std::string event_hash = ""; 
    }; 
}
#endif
