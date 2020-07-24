import numpy as np


class Cluster:
    def __init__(self, rows=None, cols=None):
        self.rows = rows if rows is not None else set()
        self.cols = cols if cols is not None else set()


class Node:
    def __init__(self, layer, detections=None, remainders=None):
        # Index of the layer (track) in the network
        self.layer = layer
        # List of detection indices considered down to current node
        self.detections = detections if detections is not None else set()
        # List of remaining detection indices to be considered up to the next layer
        self.remainders = remainders if remainders is not None else set()
        # Index of the node when added to the network. This is set by the network and
        # should not be edited.
        self.ind = None


class Net:
    def __init__(self, nodes, edges=None):
        for n_i, node in enumerate(nodes):
            node.ind = n_i
        self._nodes = nodes
        self.edges = edges if edges is not None else dict()

    @property
    def num_nodes(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    def add_node(self, node, parent, identity):
        # Set the node index
        node.ind = len(self.nodes)
        # Add node to graph
        self.nodes.append(node)
        # Create edge from parent to child
        self.edges[(parent, node)] = {identity}

    def get_parents(self, node):
        return [edge[0] for edge in self.edges if edge[1] == node]

    def get_children(self, node):
        return [edge[1] for edge in self.edges if edge[0] == node]


def gen_clusters(v_matrix):
    """ Cluster tracks into groups that sharing detections

    Parameters
    ----------
    validation_matrix: :class:`np.array`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).

    Returns
    -------
    list of :class:`Cluster` objects
        A list of :class:`Cluster` objects, where each cluster contains the indices of the rows
        (tracks) and columns (detections) pertaining to the cluster
    """
    # Initiate parameters
    num_rows, num_cols = np.shape(v_matrix)  # Number of tracks

    # Form clusters of tracks sharing measurements
    unassoc_rows = set()
    clusters = set()

    # Iterate over all rows
    for row_ind in range(num_rows):

        # Extract valid column indices
        v_cols = set(np.flatnonzero(v_matrix[row_ind, :]))

        # If there exist valid columns
        if len(v_cols):

            # Check if matched measurements are members of any clusters
            matched_clusters = [cluster for cluster in clusters
                                if any([v_ind in cluster.cols for v_ind in v_cols])]

            num_matched_clusters = len(matched_clusters)

            if num_matched_clusters == 0:
                # If not matched with any clusters, create a new one.
                new_cluster = Cluster({row_ind}, v_cols)
                clusters.add(new_cluster)
            else:
                # Start from last cluster, joining each one with the previous
                # and removing the former.
                for matched_cluster_ind in range(num_matched_clusters - 2, -1, -1):
                    matched_clusters[matched_cluster_ind].rows |= \
                        matched_clusters[matched_cluster_ind+1].rows
                    matched_clusters[matched_cluster_ind].cols |= \
                        matched_clusters[matched_cluster_ind + 1].cols
                    clusters.remove(matched_clusters[matched_cluster_ind + 1])
                    matched_clusters.pop()

                # Finally, join with associated track.
                matched_clusters[0].rows |= {row_ind}
                matched_clusters[0].cols |= v_cols
        else:
            unassoc_rows.add(row_ind)

    return list(clusters), list(unassoc_rows)


def calc_validation_and_likelihood_matrices(tracks, detections, hypotheses):
    """ Compute the validation and likelihood matrices

    Parameters
    ----------
    list of :class:`Track`
        Current tracked objects
    detections : list of :class:`Detection`
        Retrieved measurements
    hypotheses: dict
        Key value pairs of tracks with associated detections

    Returns
    -------
    :class:`np.array`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).
    :class:`np.array`
        A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
        likelihoods for all combinations of tracks and detections. The first column corresponds
        to the null hypothesis.
    """
    num_tracks = len(tracks)
    num_detections = len(detections)

    # Construct validation and likelihood matrices
    # Both matrices have shape (num_tracks, num_detections + 1), where the first column
    # corresponds to the null hypothesis.
    likelihood_matrix = np.zeros((num_tracks, num_detections + 1))
    for i, track in enumerate(tracks):
        for hyp in hypotheses[track]:
            if not hyp:
                likelihood_matrix[i, 0] = hyp.weight
            else:
                j = next(d_i for d_i, detection in enumerate(detections)
                         if hyp.measurement == detection)
                likelihood_matrix[i, j + 1] = hyp.weight
    validation_matrix = likelihood_matrix > 0
    return validation_matrix, likelihood_matrix


def construct_ehm_net(validation_matrix):
    """ Construct the EHM net as per Section 3.1 of [1]_

    Parameters
    ----------
    validation_matrix: :class:`np.array`
        An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
        (aka. valid) associations between tracks and detections. The first column corresponds
        to the null hypothesis (hence contains all ones).

    Returns
    -------
    : Net
        The constructed net object

    """
    num_tracks = validation_matrix.shape[0]

    # Initialise net
    root_node = Node(layer=-1)  # Root node is at layer -1
    net = Net([root_node])

    # A layer in the network is created for each track (not counting the root-node layer)
    num_layers = num_tracks
    for i in range(num_layers):

        # Get list of nodes in previous layer
        parent_nodes = [node for node in net.nodes if node.layer == i - 1]

        # Get indices of hypothesised detections for the track
        v_detections = set(np.flatnonzero(validation_matrix[i, :]))

        # For all nodes in previous layer
        for parent in parent_nodes:

            # Exclude any detections already considered by parent nodes (always include null)
            v_detections_m1 = (v_detections - parent.detections) | {0}

            # Iterate over valid detections
            for j in v_detections_m1:

                # Get list of nodes in current layer
                child_nodes = [node for node in net.nodes if node.layer == i]

                # Compute remainders up to next layer (i+1)
                remainders = set()
                for ii in range(i + 1, num_layers):
                    remainders |= set(np.flatnonzero(validation_matrix[ii, :]))
                remainders -= (parent.detections | {j}) - {0}

                # Find valid nodes in current layer that have the same remainders
                v_children = [child for child in child_nodes if remainders == child.remainders]

                # If layer is empty or no valid nodes exist, add new node
                if not len(v_children) or not len(child_nodes):
                    # Create new node
                    detections = parent.detections | {j}
                    child = Node(layer=i, detections=detections, remainders=remainders)
                    # Add node to net
                    net.add_node(child, parent, j)
                else:
                    # Simply add new edge or update existing one
                    for child in v_children:
                        if (parent, child) in net.edges:
                            net.edges[(parent, child)].add(j)
                        else:
                            net.edges[(parent, child)] = {j}
                        child.detections |= parent.detections | {j}
    return net


def compute_association_probabilities(net, likelihood_matrix):
    """ Compute the joint association weights, as described in Section 3.3 of [1]_

    Parameters
    ----------
    net: Net
        A net object representing the valid joint association hypotheses
        (see :meth:`~.JPDAwithEHM._construct_ehm_net` and Section 3.1 of [1]_)
    likelihood_matrix: :class:`np.array`
        A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
        likelihoods for all combinations of tracks and detections. The first column corresponds
        to the null hypothesis.

    Returns
    -------
    :class:`np.array`
        A matrix of shape (num_tracks, num_detections + 1) containing the normalised
        association probabilities for all combinations of tracks and detecrtons. The first
        column corresponds to the null hypothesis.
    """
    num_tracks, num_detections = likelihood_matrix.shape
    num_nodes = net.num_nodes

    # Compute p_D (Downward-pass) - Eq. (22) of [1]
    p_D = np.zeros((num_nodes, ))
    p_D[0] = 1
    for child in net.nodes[1:]:
        c_i = child.ind
        parents = net.get_parents(child)
        for parent in parents:
            p_i = parent.ind
            ids = list(net.edges[(parent, child)])
            p_D[c_i] += np.sum(likelihood_matrix[child.layer, ids]*p_D[p_i])

    # Compute p_U (Upward-pass) - Eq. (23) of [1]
    p_U = np.zeros((num_nodes, ))
    p_U[-1] = 1
    for parent in reversed(net.nodes[:-1]):
        p_i = parent.ind
        children = net.get_children(parent)
        for child in children:
            c_i = child.ind
            ids = list(net.edges[(parent, child)])
            p_U[p_i] += np.sum(likelihood_matrix[child.layer, ids]*p_U[c_i])

    # Compute p_DT - Eq. (21) of [1]
    p_DT = np.zeros((num_detections, num_nodes))
    for child in net.nodes:
        c_i = child.ind
        v_edges = {edge: ids for edge, ids in net.edges.items() if edge[1] == child}
        for edge, ids in v_edges.items():
            p_i = edge[0].ind
            for j in ids:
                p_DT[j, c_i] += p_D[p_i]

    # Compute p_T - Eq. (20) of [1]
    p_T = np.ones((num_detections, num_nodes))
    p_T[:, 0] = 0
    for node in net.nodes[1:]:
        n_i = node.ind
        for j in range(num_detections):
            p_T[j, n_i] = p_U[n_i]*likelihood_matrix[node.layer, j]*p_DT[j, n_i]

    # Compute association weights - Eq. (15) of [1]
    a_matrix = np.zeros(likelihood_matrix.shape)
    for i in range(num_tracks):
        node_inds = [n_i for n_i, node in enumerate(net.nodes) if node.layer == i]
        for j in range(num_detections):
            a_matrix[i, j] = np.sum(p_T[j, node_inds])
        # Normalise
        a_matrix[i, :] = a_matrix[i, :]/np.sum(a_matrix[i, :])

    return a_matrix
