# -*- coding: utf-8 -*-
import numpy as np

from .base import DataAssociator
from ..base import Property, Base
from ..hypothesiser import Hypothesiser
from ..hypothesiser.probability import PDAHypothesiser
from ..types.detection import MissedDetection
from ..types.hypothesis import (
    SingleProbabilityHypothesis, ProbabilityJointHypothesis)
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
import itertools


class PDA(DataAssociator):
    """Probabilistic Data Association (PDA)

    Given a set of detections and a set of tracks, each track has a
    probability that it is associated to each specific detection.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # Ensure association probabilities are normalised
        for track, hypothesis in hypotheses.items():
            hypothesis.normalise_probabilities(total_weight=1)

        return hypotheses


class JPDA(DataAssociator):
    r"""Joint Probabilistic Data Association (JPDA)

    Given a set of Detections and a set of Tracks, each Detection has a
    probability that it is associated with each specific Track. Rather than
    associate specific Detections/Tracks, JPDA calculates the new state of a
    Track based on its possible association with ALL Detections.  The new
    state is a Gaussian Mixture, reduced to a single Gaussian.
    If

    .. math::

          prob_{association(Detection, Track)} <
          \frac{prob_{association(MissedDetection, Track)}}{gate\ ratio}

    then Detection is assumed to be outside Track's gate, and the probability
    of association is dropped from the Gaussian Mixture.  This calculation
    takes place in the function :meth:`enumerate_JPDA_hypotheses`.
    """

    hypothesiser = Property(
        PDAHypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # enumerate the Joint Hypotheses of track/detection associations
        joint_hypotheses = \
            self.enumerate_JPDA_hypotheses(tracks, hypotheses)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from JointHypotheses
        new_hypotheses = dict()

        for track in tracks:

            single_measurement_hypotheses = list()

            # record the MissedDetection hypothesis for this track
            prob_misdetect = Probability.sum(
                joint_hypothesis.probability
                for joint_hypothesis in joint_hypotheses
                if not joint_hypothesis.hypotheses[track].measurement)

            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    hypotheses[track][0].prediction,
                    MissedDetection(timestamp=time),
                    measurement_prediction=hypotheses[track][0].measurement_prediction,
                    probability=prob_misdetect))

            # record hypothesis for any given Detection being associated with
            # this track
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    continue
                pro_detect_assoc = Probability.sum(
                    joint_hypothesis.probability
                    for joint_hypothesis in joint_hypotheses
                    if joint_hypothesis.hypotheses[track].measurement is hypothesis.measurement)

                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=pro_detect_assoc))

            result = MultipleHypothesis(single_measurement_hypotheses, True, 1)

            new_hypotheses[track] = result

        return new_hypotheses

    @classmethod
    def enumerate_JPDA_hypotheses(cls, tracks, multihypths):

        joint_hypotheses = list()

        if not tracks:
            return joint_hypotheses

        # perform a simple level of gating - all track/detection pairs for
        # which the probability of association is a certain multiple less
        # than the probability of missed detection - detection is outside the
        # gating region, association is impossible
        possible_assoc = list()

        for track in tracks:
            track_possible_assoc = list()
            for hypothesis in multihypths[track]:
                # Always include missed detection (gate ratio < 1)
                track_possible_assoc.append(hypothesis)
            possible_assoc.append(track_possible_assoc)

        # enumerate all valid JPDA joint hypotheses
        enum_JPDA_hypotheses = (
            joint_hypothesis
            for joint_hypothesis in itertools.product(*possible_assoc)
            if cls.isvalid(joint_hypothesis))

        # turn the valid JPDA joint hypotheses into 'JointHypothesis'
        for joint_hypothesis in enum_JPDA_hypotheses:
            local_hypotheses = {}

            for track, hypothesis in zip(tracks, joint_hypothesis):
                local_hypotheses[track] = \
                    multihypths[track][hypothesis.measurement]

            joint_hypotheses.append(
                ProbabilityJointHypothesis(local_hypotheses))

        # normalize ProbabilityJointHypotheses relative to each other
        sum_probabilities = Probability.sum(hypothesis.probability
                                            for hypothesis in joint_hypotheses)
        for hypothesis in joint_hypotheses:
            hypothesis.probability /= sum_probabilities

        return joint_hypotheses

    @staticmethod
    def isvalid(joint_hypothesis):

        # 'joint_hypothesis' represents a valid joint hypothesis if
        # no measurement is repeated (ignoring missed detections)

        measurements = set()
        for hypothesis in joint_hypothesis:
            measurement = hypothesis.measurement
            if not measurement:
                pass
            elif measurement in measurements:
                return False
            else:
                measurements.add(measurement)

        return True


class JPDAwithEHM(JPDA):
    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        new_hypotheses = self._run_EHM(tracks, detections, hypotheses, time)

        return new_hypotheses

    def _run_EHM(self, tracks, detections, hypotheses, time):

        track_list = list(tracks)
        detection_list = list(detections)

        net, l_matrix = self._construct_EHM_net(track_list, detection_list, hypotheses)
        a_matrix = self._compute_association_weights(net, l_matrix)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from JointHypotheses
        new_hypotheses = dict()

        for i, track in enumerate(track_list):

            single_measurement_hypotheses = list()

            prob_misdetect = Probability(a_matrix[i, 0])

            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    hypotheses[track][0].prediction,
                    MissedDetection(timestamp=time),
                    measurement_prediction=hypotheses[track][0].measurement_prediction,
                    probability=prob_misdetect))

            # record hypothesis for any given Detection being associated with
            # this track
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    continue

                j = next(i+1 for i, detection in enumerate(detection_list)
                         if hypothesis.measurement == detection)

                pro_detect_assoc = Probability(a_matrix[i, j])

                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=pro_detect_assoc))

            result = MultipleHypothesis(single_measurement_hypotheses, True, 1)

            new_hypotheses[track] = result

        return new_hypotheses

    def _construct_EHM_net(self, tracks, detections, hypotheses):

        num_tracks = len(tracks)
        num_detections = len(detections)

        # Construct validation and likelihood matrices
        # Both matrices have shape (num_tracks, num_detections + 1), where the first column
        # corresponds to the null hypothesis.
        l_matrix = np.zeros((num_tracks, num_detections + 1))
        for i, track in enumerate(tracks):
            for hyp in hypotheses[track]:
                if not hyp:
                    l_matrix[i, 0] = hyp.weight
                else:
                    j = next(d_i for d_i, detection in enumerate(detections)
                             if hyp.measurement == detection)
                    l_matrix[i, j+1] = hyp.weight
        v_matrix = l_matrix > 0

        class Node(Base):
            track_ind = Property(int, doc="Index of track to which the node relates",
                                 default=-1)
            detections = Property(set, doc="Set of detection indices examined up to that node",
                                    default=None)
            remainders = Property(set, doc="Set of remainder detection indices",
                                  default=None)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.detections is None:
                    self.detections = set()
                if self.remainders is None:
                    self.remainders = set()

        class Net(Base):
            nodes = Property([Node], doc="List of nodes in the net")
            edges = Property(dict, doc="Dictionary whose keys are in the form (p_i, c_i), where "
                                       "p_i and c_i are the parent and child node indices"
                                       "respectively. The value of each entry contains the set of"
                                       "detection indices that relate the two nodes.",
                             default=dict())

            @property
            def num_nodes(self):
                return len(self.nodes)

            def add_node(self, node, parent_ind, detections):

                # Add node to graph
                self.nodes.append(node)

                # Create edge from parent to child
                child_ind = len(self.nodes) - 1
                net.edges[(parent_ind, child_ind)] = detections

        # Initialise net
        root_node = Node()
        net = Net([root_node])

        # A layer in the network is created for each track (not counting the root-node layer)
        num_layers = num_tracks
        for i in range(num_layers):

            # Get list of (index, node) of nodes in previous layer
            parent_nodes = [(n_i, node) for n_i, node in enumerate(net.nodes)
                            if node.track_ind == i-1]

            # Get indices of measurements associated with track
            v_detection_inds = set(np.nonzero(v_matrix[i, :])[0])

            # For all nodes in previous layer
            for p_i, parent in parent_nodes:

                # Get measurements to consider
                v_detection_inds_m1 = v_detection_inds - parent.detections | {0}

                # Iterate over measurements
                for j in v_detection_inds_m1:

                    # Get list of (index, node) of nodes in current layer
                    child_nodes = [(n_i, node) for n_i, node in enumerate(net.nodes)
                                   if node.track_ind == i]

                    # If layer is empty, add new node
                    if not len(child_nodes):

                        # Create new node
                        child = Node(i, parent.detections | {j}, {p_i})

                        # Compute remainders
                        remainders = set()
                        for ii in range(i + 1, num_layers):
                            remainders |= set(np.nonzero(v_matrix[ii, :])[0])
                        child.remainders = remainders - (child.detections - {0})

                        # Add node to net
                        net.add_node(child, p_i, {j})

                    else:
                        # Compute remainders (i-1)
                        remainders_im1 = set()
                        for ii in range(i + 1, num_layers):
                            remainders_im1 |= set(np.nonzero(v_matrix[ii, :])[0])
                        remainders_im1 -= (parent.detections | {j}) - {0}

                        matched = False

                        # For all nodes in current layer
                        for c_i, child in child_nodes:

                            # If the node's list of remainders is equal to previous remainders
                            if remainders_im1 == child.remainders:
                                # Simply add new edge and update parent/child relationships
                                if (p_i, c_i) in net.edges:
                                    net.edges[(p_i, c_i)].add(j)
                                else:
                                    net.edges[(p_i, c_i)] = {j}
                                child.detections |= parent.detections | {j}

                                matched = True

                        if not matched:

                            # Create new node
                            child = Node(i, parent.detections | {j}, {p_i})

                            # Compute remainders
                            remainders = set()
                            for ii in range(i + 1, num_layers):
                                remainders |= set(np.nonzero(v_matrix[ii, :])[0])
                            child.remainders = remainders - (child.detections - {0})

                            # Add node to net
                            net.add_node(child, p_i, {j})

        return net, l_matrix

    def _compute_association_weights(self, net, l_matrix):
        num_tracks, num_detections = l_matrix.shape
        num_nodes = net.num_nodes

        # Compute p_D (Forward-pass)
        p_D = np.zeros((num_nodes, ))
        p_D[0] = 1
        for c_i, child in enumerate(net.nodes):
            if not c_i:
                continue
            parents = [edge[0] for edge in net.edges if edge[1] == c_i]
            for p_i in parents:
                p_D_m1 = p_D[p_i]
                for j in net.edges[(p_i, c_i)]:
                    p_D[c_i] += l_matrix[child.track_ind, j]*p_D_m1

        # Compute p_U (Backward-pass)
        p_U = np.zeros((num_nodes, ))
        p_U[-1] = 1
        for p_i, parent in reversed(list(enumerate(net.nodes))):
            if p_i == net.num_nodes-1:
                continue
            children = [edge[1] for edge in net.edges if edge[0] == p_i]
            for c_i in children:
                child = net.nodes[c_i]
                p_U_p1 = p_U[c_i]
                if (p_i, c_i) in net.edges:
                    for j in net.edges[(p_i, c_i)]:
                        p_U[p_i] += l_matrix[child.track_ind, j]*p_U_p1

        # Compute p_DT
        p_DT = np.zeros((num_detections, num_nodes))
        for c_i, child in enumerate(net.nodes):
            for j in range(num_detections):
                valid_pis = [edge[0] for edge, value in net.edges.items()
                             if (j in value and edge[1] == c_i)]

                for p_i in valid_pis:
                    p_D_m1 = p_D[p_i]
                    p_DT[j, c_i] += p_D_m1

        # Compute p_T
        p_T = np.ones((num_detections, num_nodes))
        p_T[:, 0] = 0
        for n_i, node in enumerate(net.nodes):
            if not n_i:
                continue
            for j in range(num_detections):
                p_T[j, n_i] = p_U[n_i]*l_matrix[node.track_ind, j]*p_DT[j, n_i]

        # Compute beta
        beta = np.zeros(l_matrix.shape)
        for i in range(num_tracks):
            node_inds = [n_i for n_i, node in enumerate(net.nodes) if node.track_ind == i]
            for j in range(num_detections):
                beta[i, j] = np.sum(p_T[j, node_inds])
            # Normalise
            beta[i, :] = beta[i, :]/np.sum(beta[i, :])

        return beta
