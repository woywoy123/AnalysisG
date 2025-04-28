# Meta Module

@brief Metadata management and analysis framework

## Overview

The Meta module provides tools for managing and analyzing metadata associated with datasets and events. It supports operations such as parsing, storing, and retrieving metadata for physics analyses. This module is essential for tracking dataset properties and ensuring reproducibility in analyses.

## Key Components

### meta

Class for managing metadata and interfacing with analysis tools.

```cpp
class meta: public tools, public notification
{
    // ...existing code...
    void scan_data(TObject* obj); // Scans data from a ROOT object
    void scan_sow(TObject* obj);  // Scans sum of weights from a ROOT object
    void parse_json(std::string inpt); // Parses metadata from a JSON file
    std::string hash(std::string fname); // Generates a hash for a given filename
    // ...existing code...
};
```

### meta_t

Struct for storing metadata attributes.

```cpp
struct meta_t {
    unsigned int dsid; // Dataset ID
    bool isMC; // Flag indicating if the dataset is Monte Carlo
    std::string derivationFormat; // Format of the dataset derivation
    std::map<int, std::string> inputfiles; // Map of input file IDs to filenames
    std::map<std::string, std::string> config; // Configuration key-value pairs
    // ...existing code...
};
```

## Usage Example

```cpp
// Create a meta object
meta* metadata = new meta();

// Parse metadata from a JSON file
metadata->parse_json("input.json");

// Scan data from a ROOT object
metadata->scan_data(root_object);

// Retrieve metadata attributes
std::string hash = metadata->hash("filename");

// Access dataset ID and format
unsigned int dsid = metadata->dsid;
std::string format = metadata->derivationFormat;
```

## Advanced Features

- **Metadata Parsing**: Parse metadata from JSON files and ROOT objects.
- **Attribute Management**: Manage and retrieve metadata attributes efficiently.
- **Integration with Analysis**: Seamlessly integrate with the analysis framework for metadata-driven workflows.
- **Hashing**: Generate unique hashes for filenames to ensure data integrity.