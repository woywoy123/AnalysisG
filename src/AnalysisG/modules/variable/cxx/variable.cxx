/**
 * @file variable.cxx
 * @brief Implementation of the variable class methods.
 *
 * This file contains methods for handling variables and their properties.
 * It implements functionality for variable manipulation and conversions.
 */

#include "variable.h"

// Implementation of the variable class methods

/**
 * @brief Sets the variable name.
 * 
 * @param name The name to set.
 */
void variable::set_name(std::string name) {
    this->name = name;
}

/**
 * @brief Gets the variable name.
 * 
 * @return The name of the variable.
 */
std::string variable::get_name() {
    return this->name;
}