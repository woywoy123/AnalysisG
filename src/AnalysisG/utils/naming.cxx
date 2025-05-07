/**
 * @file naming.cxx
 * @brief Utilities for name handling and formatting.
 *
 * This file contains utilities for handling special name formats
 * and ensuring proper escaping of special characters.
 */

#include <string>
#include <regex>

namespace utils {

/**
 * @brief Escapes special characters for XML/HTML output
 * 
 * Replaces characters like &lt;, &gt;, &amp;, etc. with their 
 * escaped equivalents to avoid parsing issues.
 * 
 * @param input The input string to escape
 * @return std::string The escaped string
 */
std::string escapeXml(const std::string& input) {
    std::string result = input;
    // Replace XML special characters with escape sequences
    result = std::regex_replace(result, std::regex("&"), "&amp;");
    result = std::regex_replace(result, std::regex("<"), "&lt;");
    result = std::regex_replace(result, std::regex(">"), "&gt;");
    result = std::regex_replace(result, std::regex("\""), "&quot;");
    result = std::regex_replace(result, std::regex("'"), "&apos;");
    return result;
}

/**
 * @brief Creates a formatted tag name
 * 
 * @param tagName The name of the tag
 * @return std::string The formatted tag with proper escaping
 */
std::string formatTagName(const std::string& tagName) {
    return "<" + escapeXml(tagName) + ">";
}

} // namespace utils