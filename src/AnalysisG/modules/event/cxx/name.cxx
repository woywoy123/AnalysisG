/**
 * @file name.cxx
 * @brief Implementation of name handling functions.
 *
 * This file contains functions for standardizing and manipulating object names.
 * All XML special characters are properly escaped.
 */

#include "name.hxx"
#include <string>
#include <algorithm>

namespace name {

std::string escapeXmlSpecialCharacters(const std::string& input) {
    std::string result;
    for (char c : input) {
        switch (c) {
            case '&': result += "&amp;"; break;
            case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break;
            case '"': result += "&quot;"; break;
            case '\'': result += "&apos;"; break;
            default: result += c; break;
        }
    }
    return result;
}

std::string standardizeName(const std::string& name) {
    std::string standardized = name;
    std::transform(standardized.begin(), standardized.end(), standardized.begin(), ::tolower);
    standardized[0] = ::toupper(standardized[0]);
    return escapeXmlSpecialCharacters(standardized);
}

} // namespace name