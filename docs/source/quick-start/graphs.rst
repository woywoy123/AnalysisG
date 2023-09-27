Advanced Usage of GraphTemplate
*******************************

Unlike the ``EventTemplate`` and ``SelectionTemplate`` classes, this template module is rather simplistic by construction, as such it will be a short section. 
Most of the content explained in the :ref:`graph-start` is already considered advanced, however, there are some hidden features that this class has, which was not discussed previously.

.. py:class:: GraphTemplate

    .. py:method:: __scrapecode__(fx, str key) -> Code

        A function which returns a dictionary which actually is a data type called **code_t**.
        This dictionary contains information about the code object and how it is preserved. 
        For more details, see the :ref:`code-types` documentation.

    .. py:method:: AddGraphFeature(fx, str name = "")

        :param function fx: Function used to compute the graph feature.

    .. py:method:: AddNodeFeature(fx, str name = "")

        :param function fx: Function used to compute the node feature.

    .. py:method:: AddEdgeFeature(fx, str name = "")

        :param function fx: Function used to compute the edge feature.

    .. py:method:: AddGraphTruthFeature(fx, str name = "")

        :param function fx: Function used to compute the graph feature.

    .. py:method:: AddNodeTruthFeature(fx, str name = "")

        :param function fx: Function used to compute the node feature.

    .. py:method:: AddEdgeTruthFeature(fx, str name = "")

        :param function fx: Function used to compute the edge feature.

    .. py:method:: AddPreSelection(fx, str name = "")

        :param function fx: Function used to accept or reject this graph

    .. py:method:: SetTopology(fx = None)

        :param function fx: Function used to impose a prior bias on the topology.

    .. py:method:: __buildthis__(str key, str co_hash, bool preselection, list this)

        A method which builds the given feature. 

    .. py:method:: Build()

        A method which builds the graph object.

    .. py:method:: ImportCode(dict inpt)

        A method which imports Code objects to the graph.

    .. py:method:: ParticleToIndex(ParticleTemplate val) -> int

        Returns the index of the given particle object within the computation graph.

    .. py:method:: is_self(inpt) -> bool

        Returns a boolean value on whether the input is of **GraphTemplate** type.

    .. py:method:: ImportMetaData(meta_t meta)

        Import the MetaData object dictionary.

    .. py:method:: Import(graph_t graph)

        Instantiate a graph object from a graph_t dictionary type.

    .. py:method:: clone() -> GraphTemplate

        A function which creates a duplicate of the graph object. 
        This **does not** clone the graph attributes, but rather only creates an empty clone of the given graph. 

    .. py:attribute:: Topology -> list

        Returns the current graph topology.

    .. py:attribute:: Export -> graph_t

        Export the graph object as a dictionary.

    .. py:attribute:: Event

        Requires an **EventTemplate** object or any object which has the appropriate attributes.

    .. py:attribute:: Particles -> list

        A list of particles to compute the topology/graph/node from.

    .. py:attribute:: self_loops -> bool

        Connect nodes to themselves, i.e. the edge-index tensor will have values with `i = j`.

    .. py:attribute:: code_owner -> bool

        A special attribute used to indicate whether the C++ backend should be owner of the code objects.

    .. py:attribute:: code -> dict[str, Code]

        Returns a dictionary with the Code objects used to construct the graph.

    .. py:attribute:: index -> int

        An index used to track which event the graph is being computed from. 

    .. py:attribute:: Errors -> dict

        Outputs a dictionary with errors encountered during graph construction.

    .. py:attribute:: PreSelectionMetric -> dict

        Outputs information about the **PreSelection** function's impact on graphs.

    .. py:attribute:: Train -> bool

        Assign the graph for training sample.

    .. py:attribute:: Eval -> bool

        Assign the graph to evaluation sample.

    .. py:attribute:: Validation -> bool

        Assign the graph to validation sample.

    .. py:attribute:: EmptyGraph -> bool

        Returns a **True** if the event has no particles/event passing the **PreSelection** or **Topology** functions.

    .. py:attribute:: SkipGraph -> bool

        Exclude the graph from training/validation/evaluation.

    .. py:attribute:: Tag -> str

        A variable used to tag the event with some string value. 

    .. py:attribute:: cached -> bool

        Indicates whether this graph has been cached and saved within a HDF5 file.
       

    .. py:attribute:: ROOT -> str
    
        Returns the ROOT filename from which the event was compiled from.

    .. py:attribute:: hash -> str

        Once set, an 18 character long string will be internally generated, which cannot be modified.
        The hash is computed from ``input/<event index>/``, and assigns each event a unique identity such that the tracer can retrieve the specified event.
        If the getter (``self.hash``) has been called prior to the setter (``self.hash = 'something'``), then an empty string is returned.

    .. py:attribute:: Graph -> bool

        Returns a boolean to indicate this graph to be of GraphTemplate type.

    .. py:attribute:: GraphName -> str
        
        Returns the name of this graph type.

    .. py:attribute:: Tree -> str

        Returns the ROOT Tree from which the graph was generated from.


Missing EventTemplate Attribute Behavior
________________________________________

.. code-block:: python 

    from AnalysisG.Templates import GraphTemplate

    class MyGraph(GraphTemplate):

        def __init__(self, Event = None):
            self.Event = Event
            self.Particles += self.Event.ArbitraryParticleName

Consider the code-block above, one might wonder what would happen if the ``Event`` implementation is missing an attribute? 
Generally, this would result in the code crashing and throwing the ``AttributeError`` exception. 
The framework in constructed to account for such instances using a pseudo-event object, which is instantiated when the ``self.Event`` variable is set. 
When the object does not contain the attribute ``ArbitraryParticleName``, the pseudo-event will return an empty list, and thus populate an empty graph (although graph level features would still be included).
This means, the event or event-graph would still be available, but with no particle nodes or edges. 
