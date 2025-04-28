# Tools Module

@brief Utility functions and helper classes for the framework

## Overview

The Tools module provides a collection of utility functions and helper classes that are used throughout the framework. These include mathematical operations, string manipulations, and file handling utilities.

## Key Components

### tools

Base class providing common utility functions.

```cpp
class tools
{
    // ...existing code...
    std::string to_lower(std::string input); // Converts a string to lowercase
    std::string to_upper(std::string input); // Converts a string to uppercase
    std::string trim(std::string input); // Trims whitespace from a string
    void write_to_file(std::string filename, std::string content); // Writes content to a file
    // ...existing code...
};
```

## Usage Example

```cpp
// Create a tools object
tools* util = new tools();

// Convert a string to lowercase
std::string lower = util->to_lower("HELLO");

// Trim whitespace from a string
std::string trimmed = util->trim("  hello world  ");

// Write content to a file
util->write_to_file("output.txt", "This is a test.");
```

## Advanced Features

- **String Manipulations**: Perform common string operations such as trimming and case conversion.
- **File Handling**: Read and write files efficiently.
- **Integration**: Use utility functions across different modules for consistency.