#ifndef META_META_H
#define META_META_H

#include <structs/meta.h>
#include <structs/folds.h>

#include <structs/property.h>
#include <rapidjson/document.h>
#include <notification/notification.h>
#include <tools/tools.h>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TH1F.h>

class analysis; 

class meta: 
    public tools,
    public notification
{
    public:
        meta(); 
        ~meta(); 

        const folds_t* get_tags(std::string hash); 

        void scan_data(TObject* obj); 
        void scan_sow(TObject* obj); 
        void parse_json(std::string inpt); 
        std::string hash(std::string fname); 

        rapidjson::Document* rpd = nullptr;
        std::string metacache_path; 
        meta_t meta_data; 

        cproperty<bool, meta> isMC; 
        cproperty<bool, meta> found; 

        cproperty<double, meta> eventNumber; 
        cproperty<double, meta> event_index; 
        cproperty<double, meta> totalSize; 
        cproperty<double, meta> kfactor;  
        cproperty<double, meta> ecmEnergy; 
        cproperty<double, meta> genFiltEff; 
        cproperty<double, meta> completion; 
        cproperty<double, meta> beam_energy; 

        cproperty<double, meta> cross_section_nb; 
        cproperty<double, meta> cross_section_fb; 
        cproperty<double, meta> cross_section_pb; 

        cproperty<double, meta> campaign_luminosity;  
        cproperty<double, meta> sum_of_weights; 

        cproperty<unsigned int, meta> dsid;                                    
        cproperty<unsigned int, meta> nFiles; 
        cproperty<unsigned int, meta> totalEvents; 
        cproperty<unsigned int, meta> datasetNumber;

        cproperty<std::string, meta> derivationFormat; 
        cproperty<std::string, meta> AMITag; 
        cproperty<std::string, meta> generators; 
        cproperty<std::string, meta> identifier; 
        cproperty<std::string, meta> DatasetName; 
        cproperty<std::string, meta> prodsysStatus; 
        cproperty<std::string, meta> dataType; 
        cproperty<std::string, meta> version; 
        cproperty<std::string, meta> PDF; 
        cproperty<std::string, meta> AtlasRelease; 
        cproperty<std::string, meta> principalPhysicsGroup; 
        cproperty<std::string, meta> physicsShort; 
        cproperty<std::string, meta> generatorName; 
        cproperty<std::string, meta> geometryVersion; 
        cproperty<std::string, meta> conditionsTag; 
        cproperty<std::string, meta> generatorTune; 
        cproperty<std::string, meta> amiStatus; 
        cproperty<std::string, meta> beamType; 
        cproperty<std::string, meta> productionStep; 
        cproperty<std::string, meta> projectName; 
        cproperty<std::string, meta> statsAlgorithm; 
        cproperty<std::string, meta> genFilterNames; 
        cproperty<std::string, meta> file_type; 
        cproperty<std::string, meta> sample_name;  
        cproperty<std::string, meta> logicalDatasetName;  
        cproperty<std::string, meta> campaign; 

        cproperty<std::vector<std::string>, meta> keywords; 
        cproperty<std::vector<std::string>, meta> weights; 
        cproperty<std::vector<std::string>, meta> keyword; 
        cproperty<std::vector<std::string>, meta> fileGUID; 

        cproperty<std::vector<int>, meta> events; 
        cproperty<std::vector<int>, meta> run_number; 
        cproperty<std::vector<double>, meta> fileSize; 

        cproperty<std::map<int, int>, meta> inputrange; 
        cproperty<std::map<int, std::string>, meta> inputfiles; 

        cproperty<std::map<std::string, int>, meta> LFN; 
        cproperty<std::map<std::string, weights_t>, meta> misc;  

        cproperty<std::map<std::string, std::string>, meta> config;
 
    private:
        friend analysis; 

        void compiler(); 
        std::vector<folds_t>* folds = nullptr; 

        float parse_float(std::string key, TTree* tr);
        std::string parse_string(std::string key, TTree* tr); 

        void static get_isMC(bool*, meta*);                                 
        void static get_found(bool*, meta*);                                 
        void static get_eventNumber(double*, meta*);                               
        void static get_event_index(double*, meta*);                               
        void static get_totalSize(double*, meta*);                               
        void static get_kfactor(double*, meta*);                               
        void static get_ecmEnergy(double*, meta*);                               
        void static get_genFiltEff(double*, meta*);                               
        void static get_completion(double*, meta*);                               
        void static get_beam_energy(double*, meta*);     

        void static get_cross_section_pb(double*, meta*);
        void static get_cross_section_nb(double*, meta*);
        void static get_cross_section_fb(double*, meta*);

        void static get_campaign_luminosity(double*, meta*);                 
        void static get_dsid(unsigned int*, meta*);                              
        void static get_nFiles(unsigned int*, meta*);                        
        void static get_totalEvents(unsigned int*, meta*);                   
        void static get_datasetNumber(unsigned int*, meta*);                   
        void static get_derivationFormat(std::string*, meta*);                    
        void static get_AMITag(std::string*, meta*);                         
        void static get_generators(std::string*, meta*);                     
        void static get_identifier(std::string*, meta*);                     
        void static get_DatasetName(std::string*, meta*);                    
        void static get_prodsysStatus(std::string*, meta*);                    
        void static get_dataType(std::string*, meta*);                       
        void static get_version(std::string*, meta*);                        
        void static get_PDF(std::string*, meta*);                            
        void static get_AtlasRelease(std::string*, meta*);                    
        void static get_principalPhysicsGroup(std::string*, meta*);                    
        void static get_physicsShort(std::string*, meta*);                    
        void static get_generatorName(std::string*, meta*);                    
        void static get_geometryVersion(std::string*, meta*);                    
        void static get_conditionsTag(std::string*, meta*);                    
        void static get_generatorTune(std::string*, meta*);                    
        void static get_amiStatus(std::string*, meta*);                      
        void static get_beamType(std::string*, meta*);                       
        void static get_productionStep(std::string*, meta*);                    
        void static get_projectName(std::string*, meta*);                    
        void static get_statsAlgorithm(std::string*, meta*);                    
        void static get_genFilterNames(std::string*, meta*);                    
        void static get_file_type(std::string*, meta*);                      
        void static get_sample_name(std::string*, meta*);                    
        void static get_logicalDatasetName(std::string*, meta*);                    
        void static get_campaign(std::string*, meta*);                       
        void static get_keywords(std::vector<std::string>*, meta*);          
        void static get_weights(std::vector<std::string>*, meta*);           
        void static get_keyword(std::vector<std::string>*, meta*);           
        void static get_fileGUID(std::vector<std::string>*, meta*);          
        void static get_events(std::vector<int>*, meta*);                    
        void static get_run_number(std::vector<int>*, meta*);                
        void static get_fileSize(std::vector<double>*, meta*);               
        void static get_inputrange(std::map<int, int>*, meta*);              
        void static get_inputfiles(std::map<int, std::string>*, meta*);      
        void static get_LFN(std::map<std::string, int>*, meta*);             
        void static get_misc(std::map<std::string, weights_t>*, meta*);      
        void static get_config(std::map<std::string, std::string>*, meta*);
        void static get_sum_of_weights(double*, meta*); 
}; 

#endif                                                                                                  
