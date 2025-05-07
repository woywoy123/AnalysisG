/**
 * @file plotting.h
 * @brief Defines the `plotting` class for creating and managing visualizations.
 *
 * This file contains the declaration of the `plotting` class, which provides
 * functionality for creating various types of visualizations from analysis data,
 * such as histograms, scatter plots, ROC curves, and other physics-specific plots.
 * The class inherits from both `tools` and `notification` to leverage utility functions
 * and messaging capabilities.
 */

#ifndef PLOTTING_H
#define PLOTTING_H

#include <notification/notification.h>
#include <structs/property.h>
#include <tools/tools.h>
#include <map>
#include <string>
#include <tuple>
#include <vector>

/**
 * @struct roc_t
 * @brief Structure for storing Receiver Operating Characteristic (ROC) curve data.
 *
 * Holds the true positive rates (TPR), false positive rates (FPR), and thresholds
 * for ROC curve plotting, as well as the calculated Area Under the Curve (AUC).
 */
struct roc_t {
    int cls = 0; 
    int kfold = 0; 
    std::string model = ""; 
    std::vector<std::vector<int>>*     truth = nullptr;
    std::vector<std::vector<double>>* scores = nullptr; 
};

/**
 * @class plotting
 * @brief Provides functionality for creating and managing visualizations.
 *
 * The plotting class handles various aspects of data visualization, including
 * creating plots, managing output formats and paths, computing statistics for plotting,
 * and specialized physics visualization techniques like ROC curves for classifier evaluation.
 * It inherits from both tools (for utility functions) and notification (for messaging).
 */
class plotting: 
    public tools, 
    public notification
{
    public:
        /**
         * @brief Constructor for the `plotting` class.
         * Initializes a new plotting instance with default settings.
         */
        plotting(); 
        /**
         * @brief Destructor for the `plotting` class.
         * Cleans up any resources used by the plotting system.
         */
        ~plotting(); 

        /**
         * @brief Builds a complete path for output files based on configured settings.
         * @return The constructed path as a string.
         */
        std::string build_path(); 
        /**
         * @brief Gets the maximum value for a specified dimension.
         * @param dim The dimension to find the maximum for (e.g., "x", "y", "pt").
         * @return The maximum value as a float.
         */
        float get_max(std::string dim); 
        /**
         * @brief Gets the minimum value for a specified dimension.
         * @param dim The dimension to find the minimum for (e.g., "x", "y", "pt").
         * @return The minimum value as a float.
         */
        float get_min(std::string dim); 
        /**
         * @brief Calculates the sum of weights for normalization.
         * @return The sum of weights as a float.
         */
        float sum_of_weights(); 
        /**
         * @brief Builds error bars or uncertainty bands for plots.
         */
        void build_error(); 
        /**
         * @brief Calculates the mean and standard deviation of a data vector.
         * @param data Pointer to the vector of data values.
         * @return A tuple containing the mean and standard deviation.
         */
        std::tuple<float, float> mean_stdev(std::vector<float>* data);

        /**
         * @brief Builds a Receiver Operating Characteristic (ROC) curve from classification results.
         * @param name The name of the ROC curve (for identification and labeling).
         * @param kfold The k-fold cross-validation index, if applicable.
         * @param labels Pointer to a vector of true labels (0 or 1).
         * @param scores Pointer to a vector of classifier scores/probabilities.
         */
        void build_ROC(std::string name, int kfold, std::vector<int>* labels, std::vector<std::vector<double>>* scores); 
        /**
         * @brief Gets the computed ROC curve data.
         * @return A vector of roc_t structures containing the ROC curve information.
         */
        std::vector<roc_t> get_ROC();  

        // io
        std::string extension = ".pdf"; ///< File extension for output files (e.g., ".pdf", ".png"). Default is ".pdf".
        std::string filename = "untitled"; ///< Base filename for output files. Default is "untitled".
        std::string output_path = "./Figures"; ///< Directory path for output files. Default is "./Figures".

        // meta data
        float x_min = 0; ///< Minimum value for x-axis. Default is 0.
        float y_min = 0; ///< Minimum value for y-axis. Default is 0.

        float x_max = 0; ///< Maximum value for x-axis. Default is 0 (auto-determined if not set).
        float y_max = 0; ///< Maximum value for y-axis. Default is 0 (auto-determined if not set).

        int x_bins = 100; ///< Number of bins for x-axis. Default is 100.
        int y_bins = 100; ///< Number of bins for y-axis. Default is 100.
        bool errors = false; ///< Flag to enable/disable error bars. Default is false.
        bool counts = false; ///< Flag to enable/disable counts display. Default is false.

        // cosmetics
        std::string style = "ROOT"; ///< Plot style. Default is "ROOT".
        std::string title = "untitled"; ///< Plot title. Default is "untitled".
        std::string ytitle = "y-axis"; ///< Y-axis title. Default is "y-axis".
        std::string xtitle = "x-axis"; ///< X-axis title. Default is "x-axis".
        std::string histfill = "fill"; ///< Histogram fill style. Default is "fill".
        std::string overflow = "sum"; ///< Overflow handling. Default is "sum".
        std::string marker = "."; ///< Marker style. Default is ".".
        std::string hatch = ""; ///< Hatch style. Default is empty.
        std::string linestyle = "-"; ///< Line style. Default is "-".

        std::string color = ""; ///< Default color for plots. Default is empty.
        std::vector<std::string> colors = {}; ///< List of colors for multi-series plots.

        bool stack   = false; ///< Flag to enable/disable stacking. Default is false.
        bool density = false; ///< Flag to enable/disable density normalization. Default is false.

        bool x_logarithmic = false; ///< Flag to enable/disable logarithmic x-axis. Default is false.
        bool y_logarithmic = false; ///< Flag to enable/disable logarithmic y-axis. Default is false.

        float line_width = 0.1; ///< Line width for plots. Default is 0.1.
        float cap_size   = 1.0; ///< Cap size for error bars. Default is 1.0.
        float alpha      = 0.4; ///< Transparency level for plots. Default is 0.4.
        float x_step     = -1; ///< Step size for x-axis. Default is -1 (auto-determined).
        float y_step     = -1; ///< Step size for y-axis. Default is -1 (auto-determined).

        // fonts
        float font_size = 10; ///< Font size for text. Default is 10.
        float axis_size = 12.5; ///< Font size for axis labels. Default is 12.5.
        float legend_size = 10; ///< Font size for legend. Default is 10.
        float title_size = 10; ///< Font size for title. Default is 10.
        bool use_latex = true; ///< Flag to enable/disable LaTeX rendering. Default is true.

        // scaling
        int dpi = 400; ///< Dots per inch for output resolution. Default is 400.
        float xscaling = 1.25*6.4; ///< Scaling factor for x-axis. Default is 1.25*6.4.
        float yscaling = 1.25*4.8; ///< Scaling factor for y-axis. Default is 1.25*4.8.
        bool auto_scale = true; ///< Flag to enable/disable auto-scaling. Default is true.

        // data containers
        std::vector<float> x_data = {}; ///< Data for x-axis.
        std::vector<float> y_data = {}; ///< Data for y-axis.

        std::map<std::string, std::map<int, std::vector<std::vector<double>>*>> roc_data = {}; ///< ROC data storage.
        std::map<std::string, std::map<int, std::vector<std::vector<int>>*>>      labels = {}; ///< Labels storage.

        std::vector<float> y_error_up   = {}; ///< Upper error values for y-axis.
        std::vector<float> y_error_down = {}; ///< Lower error values for y-axis.

        std::unordered_map<std::string, float> x_labels = {}; ///< Labels for x-axis.
        std::unordered_map<std::string, float> y_labels = {}; ///< Labels for y-axis.

        std::vector<float> variable_x_bins = {}; ///< Variable binning for x-axis.
        std::vector<float> variable_y_bins = {}; ///< Variable binning for y-axis.

        std::vector<float> weights = {}; ///< Weights for data points.
        float cross_section = -1; ///< Cross-section value. Default is -1.
        float integrated_luminosity = 140.1; ///< Integrated luminosity in fb-1. Default is 140.1.
    
    private: 
        /**
         * @brief Generates a 2D vector of specified dimensions and initializes it with zeros.
         * @tparam g The data type of the elements in the vector.
         * @param x The number of rows.
         * @param y The number of columns.
         * @return A pointer to the generated 2D vector.
         */
        template <typename g>
        std::vector<std::vector<g>>* generate(size_t x, size_t y){
            typename std::vector<g> v(y, 0); 
            return new std::vector<std::vector<g>>(x, v);
        }
}; 

#endif
