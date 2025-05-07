/**
 * @file base.h
 * @brief Defines fundamental data structures and type handling for the AnalysisG framework.
 *
 * This file contains declarations for basic structures and functions that handle
 * data type identification, conversion, and management across the framework.
 * It provides essential functionality for type safety and data management in
 * a heterogeneous, physics data analysis environment.
 */

#ifndef STRUCTS_BASE_H ///< Start of include guard for STRUCTS_BASE_H.
#define STRUCTS_BASE_H ///< Definition of STRUCTS_BASE_H to signify the header has been included.

#include <string> ///< Include standard string library for string manipulation.
#include <vector> ///< Include standard vector container for sequence storage.
#include <map> ///< Include standard map container for key-value associations.
#include <structs/enums.h> ///< Include enumerations used in the framework.

/**
 * @brief Builds a dictionary for a specific data type.
 * @param _name Full name of the data type.
 * @param _shrt Short alias for the data type.
 * 
 * This function registers a data type with the ROOT dictionary system,
 * enabling serialization and deserialization of custom objects.
 */
void buildDict(std::string _name, std::string _shrt);

/**
 * @brief Registers an include path with the interpreter.
 * @param pth Path to register.
 * @param is_abs Flag indicating if the path is absolute (true) or relative (false).
 *
 * This function adds a directory to the interpreter's include path,
 * allowing it to find header files during runtime compilation.
 */
void registerInclude(std::string pth, bool is_abs);

/**
 * @brief Builds a Precompiled Module (PCM) file for a specific module.
 * @param name Name of the module.
 * @param incl Include path for the module.
 * @param exl Flag indicating if the module should be excluded from certain operations.
 *
 * This function generates a PCM file which speeds up loading and interpretation
 * of C++ code in the ROOT framework.
 */
void buildPCM(std::string name, std::string incl, bool exl);

/**
 * @brief Initiates the build of all dictionaries and PCM files.
 *
 * This function triggers the generation of all necessary dictionaries and PCM files
 * for the types registered in the framework. This is typically called during initialization.
 */
void buildAll();

/**
 * @brief Counts occurrences of a substring within a string.
 * @param str Pointer to the string to search within.
 * @param sub The substring to search for.
 * @return The number of occurrences of the substring.
 */
int count(const std::string* str, const std::string sub);

/**
 * @class bsc_t
 * @brief Basic structure class that provides type translation and buffer management.
 *
 * This class serves as a foundation for data type handling in the framework.
 * It provides functionality to translate ROOT types to internal enumerations,
 * manage string buffers, and format type information as strings.
 */
class bsc_t {
public:
    /**
     * @brief Constructor for the bsc_t class.
     * 
     * Initializes a new basic structure instance with default settings.
     */
    bsc_t();
    
    /**
     * @brief Destructor for the bsc_t class.
     * 
     * Cleans up resources used by the basic structure instance.
     */
    ~bsc_t();
    
    /**
     * @brief Translates a ROOT type string to an internal data enumeration.
     * @param root_str Pointer to the string containing the ROOT type.
     * @return The corresponding data_enum value.
     *
     * This method parses a ROOT type string (e.g., "vector<float>") and returns
     * the corresponding internal data_enum value for type-safe operations.
     */
    data_enum root_type_translate(std::string* root_str);
    
    /**
     * @brief Represents the object's state as a string.
     * @return A string representation of the object.
     *
     * This method returns a human-readable string representation of the object's
     * current state, useful for debugging or logging.
     */
    std::string as_string();
    
    /**
     * @brief Scans and returns the current buffer content.
     * @return A string containing the current buffer content.
     *
     * This method returns the current content of the internal buffer,
     * which is used for accumulating textual data during operations.
     */
    std::string scan_buffer();
    
    /**
     * @brief Clears the internal buffer.
     *
     * This method resets the internal buffer to an empty state,
     * typically called after processing the buffer content.
     */
    void flush_buffer();
    
    std::string buffer; ///< Internal buffer for accumulating textual data.
};

#endif // STRUCTS_BASE_H ///< End of include guard for STRUCTS_BASE_H.
