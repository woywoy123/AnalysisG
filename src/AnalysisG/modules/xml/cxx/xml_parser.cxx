/**
 * @file xml_parser.cxx
 * @brief XML parsing utilities.
 *
 * This file contains functions for XML parsing and handling.
 * All XML examples are properly escaped.
 */

#include <string>

/**
 * @brief Parse an XML element
 * @param xmlStr XML string to parse
 * @return Parsed element value
 *
 * Example:
 * For input "&lt;name&gt;value&lt;/name&gt;", returns "value"
 */
std::string parseElement(const std::string& xmlStr) {
    // Implementation of XML element parsing
    size_t start = xmlStr.find('>') + 1;
    size_t end = xmlStr.rfind('<');
    if (start == std::string::npos || end == std::string::npos || start >= end) {
        return "";
    }
    return xmlStr.substr(start, end - start);
}