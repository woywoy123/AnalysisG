#ifndef PLOTTING_H
#define PLOTTING_H

#include <structs/property.h>
#include <tools/tools.h>
#include <iostream>
#include <string.h>
#include <vector>

class plotting: public tools
{
    public:
        plotting(); 
        ~plotting(); 
        std::string build_path(); 

        // io
        std::string extension = ".png"; 
        std::string filename = "untitled"; 
        std::string output_path = "./Figures"; 

        // meta data
        float x_min = 0; 
        float y_min = 0; 

        float x_max = 1; 
        float y_max = 1; 

        int x_bins = 100; 
        int y_bins = 100; 


        // cosmetics
        std::string style = "ROOT"; 
        std::string title = "untitled"; 
        std::string ytitle = "y-axis"; 
        std::string xtitle = "x-axis"; 

        // fonts
        int font_size = 8; 
        int axis_size = 8; 
        int legend_size = 8; 
        int title_size = 8; 
        bool use_latex = true; 

        // scaling
        float xscaling = 1.5; 
        float yscaling = 1.25; 
        bool auto_scale = true; 

        // data containers
        std::vector<float> x_data = {};
        std::vector<float> y_data = {}; 
        std::vector<float> weights = {}; 



}; 

#endif 
