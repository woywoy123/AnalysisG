/**
 * @file tools.h
 * @brief Defines the `tools` class that provides general utility functions for the AnalysisG framework.
 *
 * This file contains the declaration of the `tools` class, which serves as a base class for
 * many components in the AnalysisG framework. It provides a wide range of utility functions for
 * file operations, string manipulations, container operations, type handling, and more.
 * Inheriting from this class grants access to these commonly used functionalities.
 */

#ifndef TOOLS_TOOLS_H ///< Start of include guard for TOOLS_TOOLS_H.
#define TOOLS_TOOLS_H ///< Definition of TOOLS_TOOLS_H to signify the header has been included.

#include <iostream> ///< Include standard I/O library for console output.
#include <string> ///< Include standard string library for string manipulation.
#include <vector> ///< Include standard vector container for sequence storage.
#include <cstdint> ///< Include fixed-width integer type definitions.
#include <map> ///< Include standard map container for key-value associations.

/**
 * @class tools
 * @brief Provides a wide range of utility functions used throughout the framework.
 *
 * The `tools` class serves as a toolkit, offering methods for file operations,
 * string manipulations, container operations, type information retrieval, and more.
 * Many classes in the AnalysisG framework inherit from `tools` to gain access to these
 * common functionalities, promoting code reuse and consistency.
 */
class tools
{
public:
    /**
     * @brief Constructor for the `tools` class.
     * Initializes a new tools instance.
     */
    tools();
    
    /**
     * @brief Destructor for the `tools` class.
     * Cleans up resources used by the tools instance.
     */
    ~tools();  

    // File and path operations
    /**
     * @brief Creates a directory path, including parent directories if needed.
     * @param path The directory path to create.
     */
    static void create_path(std::string path); 
    
    /**
     * @brief Deletes a directory path and its contents.
     * @param path The directory path to delete.
     */
    static void delete_path(std::string path); 
    
    /**
     * @brief Checks if a path refers to an existing file.
     * @param path The path to check.
     * @return True if the path is a file and exists, false otherwise.
     */
    static bool is_file(std::string path); 
    
    /**
     * @brief Renames a file or directory.
     * @param start The original path.
     * @param target The new path.
     */
    static void rename(std::string start, std::string target); 
    
    /**
     * @brief Converts a relative path to an absolute path.
     * @param path The path to convert.
     * @return The absolute version of the path.
     */
    static std::string absolute_path(std::string path); 
    
    /**
     * @brief Lists files in a directory, optionally filtered by extension.
     * @param path The directory to search.
     * @param ext File extension filter. If empty, all files are listed.
     * @return A vector of filenames in the directory.
     */
    static std::vector<std::string> ls(std::string path, std::string ext = ""); 

    // String operations
    /**
     * @brief Converts a double value to a string with standard precision.
     * @param val The double value to convert.
     * @return The string representation of the value.
     */
    static std::string to_string(double val); 
    
    /**
     * @brief Converts a double value to a string with specified precision.
     * @param val The double value to convert.
     * @param prec The number of decimal places to consider.
     * @return The string representation of the value.
     */
    static std::string to_string(double val, int prec); 
    
    /**
     * @brief Replaces all occurrences of a substring in a string.
     * @param in Pointer to the string to modify.
     * @param repl_str The substring to replace.
     * @param repl_with The replacement string.
     */
    static void replace(std::string* in, std::string repl_str, std::string repl_with); 
    
    /**
     * @brief Checks if a string contains a specific substring.
     * @param inpt Pointer to the string to check.
     * @param trg The substring to search for.
     * @return True if the substring is found, false otherwise.
     */
    static bool has_string(std::string* inpt, std::string trg); 
    
    /**
     * @brief Checks if a string ends with a specific substring.
     * @param inpt Pointer to the string to check.
     * @param val The substring to check for at the end.
     * @return True if the string ends with the substring, false otherwise.
     */
    static bool ends_with(std::string* inpt, std::string val); 
    
    /**
     * @brief Checks if a vector of strings contains a specific string.
     * @param data Pointer to the vector to search.
     * @param trg The string to search for.
     * @return True if the string is found in the vector, false otherwise.
     */
    static bool has_value(std::vector<std::string>* data, std::string trg); 
    
    /**
     * @brief Splits a string into a vector of substrings based on a delimiter.
     * @param inpt The input string to split.
     * @param del The delimiter.
     * @return A vector of substrings.
     */
    std::vector<std::string> split(std::string inpt, std::string del); 
    
    /**
     * @brief Removes whitespace from the beginning and end of a string.
     * @param s The string to trim.
     * @return The trimmed string.
     */
    std::string trim(std::string s); 
    
    /**
     * @brief Gets the type name of a variable as a string (for debugging).
     * @tparam T The type to get the name of.
     * @return The name of the type as a string.
     */
    template <typename T>
    std::string type_name(){
        std::string tn = typeid(T).name(); ///< Gets the type name via RTTI.
        return tn; ///< Returns the type name.
    }

    // Container operations
    /**
     * @brief Empties a vector and all objects it points to.
     * @tparam T The type of pointers in the vector.
     * @param inpt Pointer to the vector to empty.
     */
    template <typename T>
    void static flush(std::vector<T*>* inpt){
        for (size_t t(0); t < inpt -> size(); ++t){ ///< Loop through all elements in the vector.
            if (!(*inpt)[t]){continue;} ///< Skip nullptr entries.
            delete (*inpt)[t]; ///< Delete the object the pointer points to.
            (*inpt)[t] = nullptr; ///< Set the pointer to nullptr to avoid dangling pointers.
        }
        inpt -> clear(); ///< Clear the vector.
    }
    
    /**
     * @brief Sums the elements of a vector.
     * @tparam g The type of elements in the vector.
     * @param inpt Pointer to the vector to sum.
     * @return The sum of all elements.
     */
    template <typename g>
    g sum(std::vector<g>* inpt){
        g out = 0; ///< Initialize the output variable.
        for (size_t t(0); t < inpt -> size(); ++t){out += (*inpt)[t];} ///< Add each element to the total.
        return out; ///< Return the sum.
    }
    
    /**
     * @brief Fills a vector with pointers from another vector based on a selection index.
     * @tparam g The type of objects being pointed to.
     * @param out Pointer to the output vector to fill.
     * @param src Pointer to the source vector to select from.
     * @param trg Pointer to a vector of indices for selection.
     */
    template <typename g>
    static void put(std::vector<g*>* out, std::vector<g*>* src, std::vector<int>* trg){
        out -> clear(); ///< Clear the output vector.
        out -> reserve(trg -> size()); ///< Reserve space for the output elements.
        for (size_t x(0); x < trg -> size(); ++x){ ///< Loop through each selected index.
            g* v = (*src)[(*trg)[x]]; ///< Get the object at the selected index.
            out -> push_back(v); ///< Add it to the output vector.
            v -> in_use = 1; ///< Mark the object as in use.
        }
    }
    
    /**
     * @brief Extracts a specific property from a vector of objects into a new vector.
     * @tparam C The type of objects in the source vector.
     * @tparam T The type of the property to extract.
     * @param prop Pointer to the output vector to fill with property values.
     * @param src Pointer to the source vector of objects.
     * @param get_prop Function pointer to extract the property from an object.
     */
    template <typename C, typename T>
    static void extract(std::vector<T>* prop, std::vector<C*>* src, void (*get_prop)(T*, C*)){
        prop -> clear(); ///< Clear the properties vector.
        prop -> reserve(src -> size()); ///< Reserve space for the property values.
        for (size_t x(0); x < src -> size(); ++x){ ///< Loop through each object in the source vector.
            T t_prop = {}; ///< Create a default-initialized property variable.
            get_prop(&t_prop, (*src)[x]); ///< Call the provided function to extract the property.
            prop -> push_back(t_prop); ///< Add the property value to the output vector.
        }
    }
}; 

#endif // TOOLS_TOOLS_H ///< End of include guard for TOOLS_TOOLS_H.
