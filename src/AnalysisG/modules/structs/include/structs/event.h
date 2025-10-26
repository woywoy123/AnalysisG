/**
 * @file event.h
 * @brief Core event data structure for AnalysisG framework
 * @defgroup modules_structs Core Data Structures
 * @{
 */

#ifndef EVENT_STRUCTS_H
#define EVENT_STRUCTS_H

#include <iostream>
#include <string>

/**
 * @struct event_t
 * @brief Fundamental event structure containing metadata and state information
 *
 * This structure represents a single physics event in the AnalysisG framework.
 * It stores essential event-level information including identification, weighting,
 * and provenance tracking.
 *
 * @note This is a POD (Plain Old Data) structure for efficient memory layout
 * and fast access patterns in event processing loops.
 *
 * ## Usage Example
 * @code{.cpp}
 * event_t event;
 * event.name = "ttbar_event_001";
 * event.weight = 1.25;  // Monte Carlo weight
 * event.index = 42;     // Event number
 * event.hash = "unique_event_id";
 * event.tree = "nominal";
 * @endcode
 *
 * @see EventTemplate for the complete event interface
 * @see particle_t for associated particle data structure
 */
struct event_t {
    /**
     * @brief Human-readable event name/identifier
     *
     * Typically set by the analysis framework or user code to identify
     * the event type or classification.
     * @default "" (empty string)
     */
    std::string name = "";

    // ========== State Variables ==========

    /**
     * @brief Monte Carlo event weight
     *
     * Represents the statistical weight of this event in Monte Carlo simulations.
     * For data events, this is typically 1.0. For MC, it combines generator weights,
     * cross-section normalization, and any analysis-specific corrections.
     *
     * @note Must be positive for physically meaningful events
     * @default 1.0
     */
    double weight = 1;

    /**
     * @brief Event index/number within the dataset
     *
     * Sequential identifier for the event in the input file or dataset.
     * Used for event tracking and reproducibility.
     *
     * @note A value of -1 indicates an uninitialized or invalid event
     * @default -1
     */
    long index = -1;

    // ========== Provenance Tracking ==========

    /**
     * @brief Unique hash identifier for the event
     *
     * Cryptographic or deterministic hash used for:
     * - Event deduplication across datasets
     * - Reproducibility tracking
     * - Cache key generation
     *
     * @default "" (empty string)
     */
    std::string hash = "";

    /**
     * @brief Name of the ROOT TTree from which this event was read
     *
     * Records the source tree name (e.g., "nominal", "systematic_up", "systematic_down")
     * for systematic variation tracking.
     *
     * @default "" (empty string)
     */
    std::string tree = "";
};

/** @} */ // end of modules_structs group

#endif
