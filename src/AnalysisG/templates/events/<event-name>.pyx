/**
 * @file <event-name>.pyx
 * @brief Template for defining event classes in the AnalysisG framework.
 */

# distutils: language=c++
# cython: language_level=3

from AnalysisG.events.<event-module>.<event-name> cimport <event-name> ///< Import the C++ event class.
from AnalysisG.core.event_template cimport EventTemplate ///< Import the base event template class.

/**
 * @class <Python-Event>
 * @brief Python wrapper for the C++ event class `<event-name>`.
 */
cdef class <Python-Event>(EventTemplate):

    /**
     * @brief Constructor for the Python event class.
     */
    def __cinit__(self): self.ptr = new <event-name>()

    /**
     * @brief Initializes the Python event class.
     */
    def __init__(self): pass

    /**
     * @brief Deallocates the Python event class.
     */
    def __dealloc__(self): del self.ptr

    # Additional methods and attributes can be defined here.

