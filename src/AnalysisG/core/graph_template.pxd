/**
 * @file graph_template.pxd
 * @brief Provides type definitions and class declarations for graph templates in the AnalysisG framework.
 *
 * This file defines the structure and behavior of graph templates, including methods for managing
 * graph data, edges, and nodes.
 */

# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from AnalysisG.core.event_template cimport event_template

cdef extern from "<templates/graph_template.h>" nogil:
    cdef cppclass graph_template:
        graph_template() except + nogil

        string tree
        string name
        string hash
        double weight
        long index
        bool preselection

        graph_template* build(event_template*) except+ nogil
        bool operator == (graph_template& p) except+ nogil
        void CompileEvent() except+ nogil
        void PreSelection() except+ nogil

/**
 * @namespace GraphTemplate
 * @brief Contains the GraphTemplate class and related utilities for managing graph data.
 */

/**
 * @class GraphTemplate
 * @brief Represents a graph structure with nodes and edges.
 *
 * This class provides methods for adding nodes, edges, and attributes, as well as for
 * retrieving graph properties and performing graph-related computations.
 */
cdef class GraphTemplate:
    cdef graph_template* ptr
    cdef list nodes_ ///< List of nodes in the graph.
    cdef list edges_ ///< List of edges in the graph.
    cdef dict attributes_ ///< Dictionary of attributes associated with the graph.

    /**
     * @brief Adds a node to the graph.
     *
     * @param node The node to add.
     */
    void add_node(int node)

    /**
     * @brief Adds an edge to the graph.
     *
     * @param source The source node of the edge.
     * @param target The target node of the edge.
     */
    void add_edge(int source, int target)

    /**
     * @brief Sets an attribute for the graph.
     *
     * @param key The key of the attribute.
     * @param value The value of the attribute.
     */
    void set_attribute(string key, string value)

    /**
     * @brief Retrieves an attribute from the graph.
     *
     * @param key The key of the attribute.
     * @return The value of the attribute.
     */
    string get_attribute(string key)

    /**
     * @brief Retrieves the number of nodes in the graph.
     *
     * @return The number of nodes.
     */
    int num_nodes()

    /**
     * @brief Retrieves the number of edges in the graph.
     *
     * @return The number of edges.
     */
    int num_edges()

