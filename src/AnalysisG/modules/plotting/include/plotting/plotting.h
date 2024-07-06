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
        float get_max(std::string dim); 
        float get_min(std::string dim); 
        float sum_of_weights(); 


        // io
        std::string extension = ".png"; 
        std::string filename = "untitled"; 
        std::string output_path = "./Figures"; 

        // meta data
        float x_min = 0; 
        float y_min = 0; 

        float x_max = 0; 
        float y_max = 0; 

        int x_bins = 100; 
        int y_bins = 100; 
        bool errors = false; 

        // cosmetics
        std::string style = "ROOT"; 
        std::string title = "untitled"; 
        std::string ytitle = "y-axis"; 
        std::string xtitle = "x-axis"; 
        std::string histfill = "fill"; 
        bool stack   = false; 
        bool density = false;
        bool x_logarithmic = false; 
        bool y_logarithmic = false; 
        float line_width = 1; 
        float alpha      = 0.4; 
        float x_step     = -1; 
        float y_step     = -1; 

        // fonts
        float font_size = 10; 
        float axis_size = 12.5; 
        float legend_size = 10; 
        float title_size = 10; 
        bool use_latex = true; 

        // scaling
        int dpi = 400; 
        float xscaling = 1.25*6.4; 
        float yscaling = 1.25*4.8; 
        bool auto_scale = true; 

        // data containers
        std::vector<float> x_data = {};
        std::vector<float> y_data = {}; 
        std::vector<float> weights = {}; 
        float cross_section = -1; 
        float integrated_luminosity = 140.1; //fb-1


}; 

#endif 
