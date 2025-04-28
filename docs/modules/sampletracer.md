# SampleTracer Module

@brief Sample management and tracing framework

## Overview

The SampleTracer module provides tools for managing and tracing samples in an analysis workflow. It supports operations such as adding metadata, retrieving events, and managing sample containers. This module is critical for organizing and accessing datasets efficiently during analysis.

## Key Components

### sampletracer

Class for managing and tracing samples.

```cpp
class sampletracer
{
    bool add_meta_data(meta* meta_, std::string filename); // Adds metadata to a sample
    meta* get_meta_data(std::string filename); // Retrieves metadata for a given sample
    std::vector<event_template*> get_events(std::string label); // Retrieves events by label
    bool add_event(event_template* ev, std::string label); // Adds an event to a specific label
};
```

## Usage Example

```cpp
// Create a sampletracer object
sampletracer* tracer = new sampletracer();

// Add metadata to a sample
tracer->add_meta_data(metadata, "sample_file");

// Retrieve metadata for a sample
meta* metadata = tracer->get_meta_data("sample_file");

// Retrieve events for a specific label
std::vector<event_template*> events = tracer->get_events("label");

// Add an event to a label
tracer->add_event(event, "signal");
```

## Advanced Features

- **Sample Management**: Add and retrieve metadata for samples.
- **Event Tracing**: Retrieve events associated with specific labels.
- **Integration with Metadata**: Seamlessly integrate with the Meta module for metadata-driven workflows.
- **Labeling**: Organize events into labeled categories for efficient access.