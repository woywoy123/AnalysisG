.. _meta-t:

Meta Data Structures
--------------------

.. cpp:struct:: weights_t

    .. cpp:var:: int dsid 

    .. cpp:var:: bool isAFII

    .. cpp:var:: std::string generator

    .. cpp:var:: std::string ami_tag

    .. cpp:var:: float total_events_weighted

    .. cpp:var:: float total_events

    .. cpp:var:: float processed_events

    .. cpp:var:: float processed_events_weighted

    .. cpp:var:: float processed_events_weighted_squared

    .. cpp:var:: std::map<std::string, float> hist_data


.. cpp:struct:: meta_t 

    .. cpp:var:: unsigned int dsid                             

    .. cpp:var:: bool isMC                                     

    .. cpp:var:: std::string derivationFormat                  

    .. cpp:var:: std::map<int, std::string> inputfiles         

    .. cpp:var:: std::map<std::string, std::string> config     
    
    .. cpp:var:: std::string AMITag                            
    
    .. cpp:var:: std::string generators                        
    
    .. cpp:var:: std::map<int, int> inputrange                 
   
    .. cpp:var:: double eventNumber                            
    
    .. cpp:var:: double event_index                            
    
    .. cpp:var:: bool found                                    
    
    .. cpp:var:: std::string DatasetName                       
    
    .. cpp:var:: double totalSize                              
    
    .. cpp:var:: double kfactor                                
    
    .. cpp:var:: double ecmEnergy                              
    
    .. cpp:var:: double genFiltEff                             
    
    .. cpp:var:: double completion                             
    
    .. cpp:var:: double beam_energy                            
    
    .. cpp:var:: double crossSection                           
    
    .. cpp:var:: double crossSection_mean                      
    
    .. cpp:var:: double campaign_luminosity                    
    
    .. cpp:var:: unsigned int nFiles                           
    
    .. cpp:var:: unsigned int totalEvents                      
    
    .. cpp:var:: unsigned int datasetNumber                    
    
    .. cpp:var:: std::string identifier                        
    
    .. cpp:var:: std::string prodsysStatus                     
    
    .. cpp:var:: std::string dataType                          
    
    .. cpp:var:: std::string version                           
    
    .. cpp:var:: std::string PDF                               
    
    .. cpp:var:: std::string AtlasRelease                      
    
    .. cpp:var:: std::string principalPhysicsGroup             
    
    .. cpp:var:: std::string physicsShort                      
    
    .. cpp:var:: std::string generatorName                     
    
    .. cpp:var:: std::string geometryVersion                   
    
    .. cpp:var:: std::string conditionsTag                     
    
    .. cpp:var:: std::string generatorTune                     
    
    .. cpp:var:: std::string amiStatus                         
    
    .. cpp:var:: std::string beamType                          

    .. cpp:var:: std::string productionStep                    

    .. cpp:var:: std::string projectName                       

    .. cpp:var:: std::string statsAlgorithm                    

    .. cpp:var:: std::string genFilterNames                    

    .. cpp:var:: std::string file_type                         

    .. cpp:var:: std::string sample_name                       

    .. cpp:var:: std::string logicalDatasetName                

    .. cpp:var:: std::string campaign                          

    .. cpp:var:: std::vector<std::string> keywords             

    .. cpp:var:: std::vector<std::string> weights              

    .. cpp:var:: std::vector<std::string> keyword              

    .. cpp:var:: std::vector<int> events                       

    .. cpp:var:: std::vector<int> run_number                   

    .. cpp:var:: std::vector<double> fileSize                  

    .. cpp:var:: std::vector<std::string> fileGUID             

    .. cpp:var:: std::map<std::string, int> LFN                

    .. cpp:var:: std::map<std::string, weights_t> misc         





