Graph
_____

An extension module which aims to reduce the need for writing redundant torch.scatter node and edge aggregation.
Each of the functions highlighted below have the same output type and structure.
These are outlined below:

- clusters: 
    The aggregated sum of the 4-vectors 

- unique_sum: 
    A tensor holding the unique cluster masses for the predicted topology. 
    This can be interpreted as the number of particles reconstructed in the particular graph. 
    For instance, if the event contains n-particles, with n-signature masses, then in an ideal situation n-particles would be contained within this tensor, rather than m-constituent particles (children). 

- reverse_sum: 
    A reverse mapping of unique_sum back to each graph node.

- node_sum: 
    A tensor holding the aggregated node's 4-vector.



.. py:module:: pyc.Graph.Base

        .. py:function:: unique_aggregation(clusters, node_features) -> torch.tensor

            An aggregation function used to sum node features without double summing any particular node pairs. 
            Given the complexity of the operation in nominal PyTorch, this function is exclusive to CUDA tensors.

            :param torch.tensor clusters: Cluster indices used to perform the summation over.
            :param torch.tensor node_features: The feature vector of the node to sum.


.. py:module:: pyc.Graph.Cartesian


        .. py:function:: edge(edge_index, prediction, px, py, pz, e, include_zero) -> dict
            :no-index:

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The edge classification prediction of the neural network.
            :param torch.tensor px: The node's cartesian momentum component in the x-direction.
            :param torch.tensor py: The node's cartesian momentum component in the y-direction.
            :param torch.tensor pz: The node's cartesian momentum component in the z-direction.
            :param torch.tensor e: The node's energy.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}


        .. py:function:: edge(edge_index, prediction, pmc, include_zero) -> dict

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The edge classification prediction of the neural network.
            :param torch.tensor pmc: A compact version of the particle's 4-vector.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}


        .. py:function:: node(edge_index, prediction, px, py, pz, e, include_zero) -> dict
            :no-index:

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The node classification prediction of the neural network.
            :param torch.tensor px: The node's cartesian momentum component in the x-direction.
            :param torch.tensor py: The node's cartesian momentum component in the y-direction.
            :param torch.tensor pz: The node's cartesian momentum component in the z-direction.
            :param torch.tensor e: The node's energy.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}


        .. py:function:: node(edge_index, prediction, pmc, include_zero) -> dict

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The node classification prediction of the neural network.
            :param torch.tensor pmc: A compact version of the particle's 4-vector.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}
 



.. py:module:: pyc.Graph.Polar


        .. py:function:: edge(edge_index, prediction, pt, eta, phi, e, include_zero) -> dict
            :no-index:

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The edge classification prediction of the neural network.
            :param torch.tensor pt: The particle's transverse momentum
            :param torch.tensor eta: The rapidity of the particle
            :param torch.tensor phi: The azimuthal compnent of the particle node
            :param torch.tensor e: The node's energy.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}


        .. py:function:: edge(edge_index, prediction, pmu, include_zero) -> dict

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The edge classification prediction of the neural network.
            :param torch.tensor pmu: A compact version of the particle's 4-vector.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}


        .. py:function:: node(edge_index, prediction, pt, eta, phi, e, include_zero) -> dict
            :no-index:

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The node classification prediction of the neural network.
            :param torch.tensor pt: The particle's transverse momentum
            :param torch.tensor eta: The rapidity of the particle
            :param torch.tensor phi: The azimuthal compnent of the particle node
            :param torch.tensor e: The node's energy.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}


        .. py:function:: node(edge_index, prediction, pmu, include_zero) -> dict

            :param torch.tensor edge_index: The graph topology (this is usually known as the edge_index)
            :param torch.tensor prediction: The node classification prediction of the neural network.
            :param torch.tensor pmu: A compact version of the particle's 4-vector.
            :param bool include_zero: Whether to include predictions where the MLP predicts 0. 

            :return dict[torch.tensor]: {
                                            clusters : torch.tensor, unique_sum : torch.tensor, 
                                            reverse_sum : torch.tensor, node_sum : torch.tensor}
 



