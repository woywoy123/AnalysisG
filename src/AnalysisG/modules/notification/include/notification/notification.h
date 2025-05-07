/**
 * @file notification.h
 * @brief Defines the `notification` class for logging and messaging functionality.
 *
 * This file contains the declaration of the `notification` class, which provides
 * standardized logging and messaging capabilities throughout the AnalysisG framework.
 * It enables components to output status updates, warnings, errors, and debug
 * information with consistent formatting and control.
 */

#ifndef NOTIFICATION_NOTIFICATION_H ///< Start of include guard for NOTIFICATION_NOTIFICATION_H.
#define NOTIFICATION_NOTIFICATION_H ///< Definition of NOTIFICATION_NOTIFICATION_H to signify the header has been included.

#include <string> ///< Include standard string library for string manipulation.
#include <iostream> ///< Include standard I/O library for console output.

/**
 * @class notification
 * @brief Provides logging and messaging functionality with various severity levels.
 *
 * The `notification` class serves as a base class for many components in the
 * AnalysisG framework and provides a consistent interface for message output.
 * It supports different severity levels (INFO, WARNING, ERROR, DEBUG) and can
 * be configured to suppress certain types of messages or redirect output.
 */
class notification
{
public:
    /**
     * @brief Constructor for the `notification` class.
     * Initializes a new notification instance with default settings.
     */
    notification(); 
    
    /**
     * @brief Destructor for the `notification` class.
     * Cleans up any resources used by the notification system.
     */
    ~notification(); 

    /**
     * @brief Sets the prefix used in log messages from this instance.
     * @param prefix The string to use as a prefix for log messages.
     */
    void set_prefix(std::string prefix); 
    
    /**
     * @brief Returns the prefix currently used in log messages.
     * @return The current prefix string.
     */
    std::string get_prefix(); 

    /**
     * @brief Outputs an info message.
     * @param msg The message to output.
     * @param newline Whether to append a newline (default: true).
     */
    void Info(std::string msg, bool newline = true); 
    
    /**
     * @brief Outputs a warning message.
     * @param msg The warning message to output.
     * @param newline Whether to append a newline (default: true).
     */
    void Warning(std::string msg, bool newline = true);
    
    /**
     * @brief Outputs an error message.
     * @param msg The error message to output.
     * @param newline Whether to append a newline (default: true).
     */
    void Error(std::string msg, bool newline = true); 
    
    /**
     * @brief Outputs a debug message (only in debug mode).
     * @param msg The debug message to output.
     * @param newline Whether to append a newline (default: true).
     */
    void Debug(std::string msg, bool newline = true);
    
    /**
     * @brief Outputs a message with custom formatting.
     * @param msg The message to output.
     * @param tag A custom tag to prepend to the message.
     * @param newline Whether to append a newline (default: true).
     */
    void Message(std::string msg, std::string tag = "", bool newline = true); 

    /**
     * @brief Enables or disables debug mode.
     * @param dbg Boolean flag, set to true to enable debug messages, false to disable.
     */
    void set_debug_mode(bool dbg); 
    
    /**
     * @brief Returns the current status of debug mode.
     * @return Boolean indicating whether debug mode is enabled (true) or disabled (false).
     */
    bool get_debug_mode(); 

    /**
     * @brief Sets whether warnings should be suppressed.
     * @param sw Boolean flag, set to true to suppress warnings, false to show them.
     */
    void set_suppress_warning(bool sw); 
    
    /**
     * @brief Returns the current warning suppression status.
     * @return Boolean indicating whether warnings are suppressed (true) or shown (false).
     */
    bool get_suppress_warning(); 

    /**
     * @brief Sets whether info messages should be suppressed.
     * @param si Boolean flag, set to true to suppress info messages, false to show them.
     */
    void set_suppress_info(bool si); 
    
    /**
     * @brief Returns the current info message suppression status.
     * @return Boolean indicating whether info messages are suppressed (true) or shown (false).
     */
    bool get_suppress_info(); 

protected:
    std::string prefix = ""; ///< Prefix used in log messages from this instance. Initialized as empty string.
    bool debug_mode = false; ///< Flag indicating whether debug mode is enabled. Initialized as false.
    bool suppress_warnings = false; ///< Flag indicating whether warnings are suppressed. Initialized as false.
    bool suppress_info = false; ///< Flag indicating whether info messages are suppressed. Initialized as false.
}; 

#endif // NOTIFICATION_NOTIFICATION_H ///< End of include guard for NOTIFICATION_NOTIFICATION_H.
