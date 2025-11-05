
from karateclub import DeepWalk, Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import textdistance

import networkx as nx

import warnings
from typing import Literal, List, Optional, Union
from tqdm import tqdm

import numpy as np
from collections import Counter
import math
import copy

from langchain_google_vertexai import VertexAIEmbeddings

class Eval:
    """ Base class for evaluating knowledge graphs
    """
    def __init__(self, kg_gt: nx.MultiDiGraph, kg_pred: nx.MultiDiGraph, embed: Optional[bool] = False) -> None:
        """ initialization of Eval class

        Args:
            kg_gt (nx.MultiDiGraph): Ground truth knowledge graph that will be loaded with the class initialization. Defaults to None.
            kg_pred (nx.MultiDiGraph): Predicted knowledge graph that will be loaded with the class initialization. Defaults to None.
            embed (Optional[bool], optional): Wether or not the nodes names of given knowledge graphs will be embedded. Defaults to None.
        """
        # initialize class with given knowledge graphs or create new ones
        self.kg_gt = kg_gt
        self.kg_pred = kg_pred
 
        self.embedded_pred_nodes = None
        self.embedded_gt_nodes = None

        # embed nodes of given knowledge graphs
        if embed:
            self.embedded_gt_nodes = self.embed_nodes(self.kg_gt)
            self.embedded_pred_nodes = self.embed_nodes(self.kg_pred)
        
    def embed_nodes(self, kg: nx.MultiDiGraph) -> dict[str, List[float]]:
        """ returns all node names in the provided graph as embeddings 

        Args:
            kg (nx.MultiDiGraph): the graph of which the node names will be embedded

        Returns:
            dict[str, List[float]]: a dictionary comprising of the node name as key and the embedding as value
        """
        out = {}
        for node_name in tqdm(list(kg.nodes), desc="Embedding nodes...", leave=False):
            if len(node_name) > 0:
                embeddings = VertexAIEmbeddings(model_name="text-embedding-004", api_transport="rest", location="europe-west4")
                embedded_node = embeddings.embed_query(node_name)
                out[node_name] = embedded_node
        return out

    def deep_walk(self, kg_type: Literal["ground_truth", "prediction"], directed: bool=True) -> List[float]:
        """Structually embedds the node names of the provided graph using the deepwalk algorithm

        Args:
            kg_type (Literal[&quot;ground_truth&quot;, &quot;prediction&quot;]): The type of knowledge graph that is structually embedded on the node level
            directed (bool, optional): Wether or not length descrepancy between the two graphs node count should be factored in. Defaults to True.

        Returns:
            List[float]: List of structually embedded nodes
        """
        if kg_type == "ground_truth":
            graph = self.kg_gt
        else:
            graph = self.kg_pred
        
        if not directed:
            graph = graph.to_undirected()

        node_mapping = {node: i for i, node in enumerate(graph)}

        indexed_graph = nx.relabel_nodes(graph, node_mapping)

        model = DeepWalk(epochs=30, walk_number=50, walk_length=80)
        model.fit(indexed_graph)
        return model.get_embedding().tolist()
    
    def node2vec(self, kg_type: Literal["ground_truth", "prediction"], directed: bool=True) -> List[List[float]]:
        """Structually embedds the node names of the provided graph using the node2vec algorithm

        Args:
            kg_type (Literal[&quot;ground_truth&quot;, &quot;prediction&quot;]): The type of knowledge graph that is structually embedded on the node level
            directed (bool, optional): Wether or not length descrepancy between the two graphs node count should be factored in. Defaults to True.

        Returns:
            List[float]: List of structually embedded nodes
        """
        if kg_type == "ground_truth":
            graph = self.kg_gt
        else:
            graph = self.kg_pred
        
        if not directed:
            graph = graph.to_undirected()

        node_mapping = {node: i for i, node in enumerate(graph)}

        indexed_graph = nx.relabel_nodes(graph, node_mapping)

        model = Node2Vec(epochs=30, walk_number=50, walk_length=80)
        model.fit(indexed_graph)
        
        return model.get_embedding().tolist()

    def jaccard_similarity_coefficient(self, node_name_1: str, node_name_2: str, threshold: float) -> float:
        """calculates the jaccard similarity coefficient of two given node names

        Args:
            node_name_1 (str): name of the first node that should be checked for similarity
            node_name_2 (str): name of the second node that should be checked for similarity
            threshold (float): threshold at which two nodes are considered equal

        Returns:
            float: jaccard similarity coefficient of the given nodes
        """
        if threshold == 1:
            return textdistance.jaccard.similarity(node_name_1, node_name_2)
        else:
            if textdistance.jaccard.similarity(node_name_1, node_name_2) >= threshold:
                return 1
            else:
                return 0

    def overlap_coefficient(self, node_name_1: str, node_name_2: str, threshold: float) -> float:
        """calculates the overlap coefficient of two given node names

        Args:
            node_name_1 (str): name of the first node that should be checked for similarity
            node_name_2 (str): name of the second node that should be checked for similarity
            threshold (float): threshold at which two nodes are considered equal

        Returns:
            float: overlap coefficient of the given nodes
        """
        # Calculate overlap coefficient
        if threshold == 1:
            return textdistance.overlap.similarity(node_name_1, node_name_2)
        else:
            if textdistance.overlap.similarity(node_name_1, node_name_2) >= threshold:
                return 1
            else:
                return 0
    
    def levenshtein_similarity_coefficient(self, node_name_1: str, node_name_2: str, threshold: float) -> float:
        """calculates the levenshtein similarity coefficient of two given node names

        Args:
            node_name_1 (str): name of the first node that should be checked for similarity
            node_name_2 (str): name of the second node that should be checked for similarity
            threshold (float): threshold at which two nodes are considered equal

        Returns:
            float: levenshtein similarity coefficient of the given nodes
        """
        distance = textdistance.levenshtein.distance(node_name_1, node_name_2)
        similarity_percentage = (1 - distance / max(len(node_name_1), len(node_name_2)))
        if threshold == 1:
            return similarity_percentage
        else:
            if similarity_percentage >= threshold:
                return 1
            else:
                return 0
    
    def dice_coefficient(self, node_name_1: str, node_name_2: str, threshold: float) -> float:
        """calculates the dice coefficient of two given node names

        Args:
            node_name_1 (str): name of the first node that should be checked for similarity
            node_name_2 (str): name of the second node that should be checked for similarity
            threshold (float): threshold at which two nodes are considered equal

        Returns:
            float: dice coefficient of the given nodes
        """
        

        if threshold == 1:
            return textdistance.sorensen_dice.similarity(node_name_1, node_name_2)
        else:
            if textdistance.sorensen_dice.similarity(node_name_1, node_name_2) >= threshold:
                return 1
            else:
                return 0

    def jaro_winkler_similarity_coefficient(self, node_name_1: str, node_name_2: str, threshold: float) -> float:
        """calculates the jaro winkler similarity coefficient of two given node names

        Args:
            node_name_1 (str): name of the first node that should be checked for similarity
            node_name_2 (str): name of the second node that should be checked for similarity
            threshold (float): threshold at which two nodes are considered equal
            
        Returns:
            float: jaro winkler similarity coefficient of the given nodes
        """
        if threshold == 1:
            return textdistance.jaro_winkler.similarity(node_name_1, node_name_2)
        else:
            if textdistance.jaro_winkler.similarity(node_name_1, node_name_2) >= threshold:
                return 1
            else:
                return 0
        
    def cosine_similarity_coefficient(self, node_vector_1, node_vector_2, threshold: float) -> float:
        """calculates the cosine similarity coefficient of two given node names

        Args:
            node_vector_1 (str): name of the first node that should be checked for similarity
            node_vector_2 (str): name of the second node that should be checked for similarity
            threshold (float): threshold at which two nodes are considered equal

        Returns:
            float: cosine similarity coefficient of the given nodes
        """
        vectors = (node_vector_1, node_vector_2)

        if threshold == 1:
            return cosine_similarity(vectors)[0, 1]
        else:
            if (cosine_similarity(vectors)[0, 1]) >= threshold:
                return 1
            else:
                return 0

    def get_max_similarity_node(self, gt_node: Union[str, List[float]], pred_nodes: List[Union[List[str], List[List[float]]]], 
                                  similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                                             'levenshtein_similarity', 'dice_coefficient', 
                                                             'jaro_winkler_similarity', 'cosine_similarity'],
                                  threshold: float) -> tuple[str, float]:
        """ Identifies the most similar node name from the predicted knowledge graph for a given ground truth node name using the specified similarity metric.

        Args:
            node_gt (str): name of the node from the ground truth knowledge graph
            pred_nodes (List[str]): list of nodes names from the predicted knowledge graph that are checked for similarity to the gt_node
            similarity_metric (str): type of metric to be used for the similarity calculation 
            threshold (float): threshold at which two nodes are considered equal

        Raises:
            Exception: similarity metric is not supported!

        Returns:
            tuple[str, float]: name of the most similar node, calculated similarity coefficient 
        """
        similar_node, max_similarity_coefficient = "", -math.inf

        for pred_node in pred_nodes:
            if similarity_metric == "jaccard_similarity_coefficient":
                similarity_coefficient = self.jaccard_similarity_coefficient(node_name_1=gt_node, node_name_2=pred_node, threshold=threshold)
            elif similarity_metric == "overlap_coefficient":
                similarity_coefficient = self.overlap_coefficient(node_name_1=gt_node, node_name_2=pred_node,threshold=threshold)
            elif similarity_metric == "levenshtein_similarity":
                similarity_coefficient = self.levenshtein_similarity_coefficient(node_name_1=gt_node, node_name_2=pred_node,threshold=threshold)
            elif similarity_metric == "dice_coefficient":
                similarity_coefficient = self.dice_coefficient(node_name_1=gt_node, node_name_2=pred_node, threshold=threshold)
            elif similarity_metric == "jaro_winkler_similarity":
                similarity_coefficient = self.jaro_winkler_similarity_coefficient(node_name_1=gt_node, node_name_2=pred_node, threshold=threshold)
            elif similarity_metric == "cosine_similarity":
                similarity_coefficient = self.cosine_similarity_coefficient(node_vector_1=gt_node, node_vector_2=pred_node, threshold=threshold)
            else:
                raise Exception(f"{similarity_metric} is not supported!")
            
            if similarity_coefficient > max_similarity_coefficient:
                similar_node, max_similarity_coefficient = pred_node, similarity_coefficient
        
        return similar_node, max_similarity_coefficient

    def kg_node_stats(self, kg_type: Literal["ground_truth", "prediction"]) -> tuple[int, dict]:
        """calculates basic stats regarding the nodes of a knowledge graph

        Args:
            type (Literal[&quot;ground_truth&quot;, &quot;prediction&quot;]): the type of knowledge graph that will be evaluated

        Returns:
            tuple[int, dict]: returns overall node count and the node count by assigned type
        """

        if kg_type == "ground_truth":
            kg = self.kg_gt
        else:
            kg = self.kg_pred

        node_count = len(kg.nodes)

        type_count = dict(Counter(item[1]['type'] for item in kg.nodes(data=True) if item[1] != {}))
        
        return node_count, type_count

    def kg_relationship_stats(self, kg_type: Literal["ground_truth", "prediction"]) -> tuple[int, dict, dict, dict]:
        """calculates basic stats regarding the relationships of a knowledge graph

        Args:
            type (Literal[&quot;ground_truth&quot;, &quot;prediction&quot;]):the type of knowledge graph that will be evaluated

        Returns:
            tuple[int, dict, dict, dict]: returns the overall relationship count, relationship count by source entity, count by relation type, relationship count by target entity
        """
        if kg_type == "ground_truth":
            kg = self.kg_gt
        else:
            kg = self.kg_pred

        relationship_count = len(kg.edges)

        source_count = dict(Counter(item[0] for item in kg.edges(data=True)))

        relation_count = dict(Counter(item[2]['relation'] for item in kg.edges(data=True)))

        target_count = dict(Counter(item[1] for item in kg.edges(data=True)))

        return relationship_count, source_count, relation_count, target_count

    def node_similarity_coefficient(self,
        similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                    'levenshtein_similarity', 'dice_coefficient', 
                                    'jaro_winkler_similarity', 'cosine_similarity'], 
        threshold: float = 1,
        compute_diff: bool = True,
        show_matches: bool = False,
        show_unmatched: bool = False) -> float:
        """calculates the node similarity coefficient (NSC) between the ground truth knowledge graph and the predicted knowledge graph

        Args:
            similarity_metric (str): type of metric to be used for the node name similarity calculation 
            threshold (float, optional): threshold at which two nodes are considered to be equal. Defaults to 1 
            compute_diff (bool, optional): wether excessive nodes in the predicted knowledge graph should influence the computed NSC score. Defaults to True.
            show_matches (bool, optional): wether or not all matched nodes will be printed. Defaults to False.
            show_unmatched (bool, optional): wether or not all unmatched nodes will be printed. Defaults to False.

        Returns:
            float: The calculated node similarity coefficient based on the specified similarity metric

        Raises:
            Warning: Matching the ground truth knowledge graph to the predicted knowledge graph can lead to inaccurate similarity scores! To get more accurate results set compute_diff=True
        """
        # wether to use nodes in string form or embedded
        if similarity_metric == "cosine_similarity":
            # embedd nodes if necessary
            if not self.embedded_gt_nodes:
                self.embedded_gt_nodes = self.embed_nodes(self.kg_gt)
            if not self.embedded_pred_nodes:
                self.embedded_pred_nodes = self.embed_nodes(self.kg_pred)

            pred_nodes = list(copy.deepcopy(self.embedded_pred_nodes).values())
            gt_nodes = list(copy.deepcopy(self.embedded_gt_nodes).values())
        else:
            pred_nodes = list(copy.deepcopy(self.kg_pred.nodes))
            gt_nodes = list(copy.deepcopy(self.kg_gt.nodes))

        # warning if length difference is computetd everytime
        if len(gt_nodes) > len(pred_nodes) and compute_diff is False:
            warnings.warn("Unable to avoid computing the length difference. Amount of predicted nodes is smaller than the ground truth! To get results set compute_diff=True")
            return f"Can not calculate a node similarity coefficient based on the {similarity_metric} due to parameter conflicts!"

        similarity_coefficient_sum, matches = self.calculate_node_similarity_coefficient(gt_nodes=gt_nodes, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)

        if similarity_metric == "cosine_similarity":
            for entry in matches:
                unvectorized_gt_node = next((key for key, value in self.embedded_gt_nodes.items() if value == entry["ground_truth"]), None)
                unvectorized_pred_node = next((key for key, value in self.embedded_pred_nodes.items() if value == entry["prediction"]), None)
                entry["ground_truth"] = unvectorized_gt_node
                entry["prediction"] = unvectorized_pred_node

        # wether missing nodes or excessive nodes in the predicted knowledge graph should influence the accuracy score
        if compute_diff:
            diff = abs(len(list(self.kg_gt.nodes)) - len(list(self.kg_pred.nodes)))
            node_similarity_coefficient = similarity_coefficient_sum / (len(list(self.kg_gt.nodes)) + diff)
        else:
            node_similarity_coefficient = similarity_coefficient_sum / min(len(list(self.kg_gt.nodes)), len(list(self.kg_pred.nodes)))

        if show_matches:
            print("Matched nodes: " + str(matches))

        if show_unmatched:
            predictions = [item['prediction'] for item in matches]
            unmatched = [item for item in self.kg_pred.nodes if item not in predictions]
            print("Unmatched nodes: " + str(unmatched))
        
        return f"Calculated a node similarity coefficient of {node_similarity_coefficient} based on the {similarity_metric}!"

    def calculate_node_similarity_coefficient(self, 
        gt_nodes: Union[List[str], List[List[float]]], 
        pred_nodes: Union[List[str], List[List[float]]],
        similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                    'levenshtein_similarity', 'dice_coefficient', 
                                    'jaro_winkler_similarity', 'cosine_similarity'], 
        threshold: float=1) -> float:
        """ Calculates the similarity between the ground truth and the prediction based on the level of the node names

        Args:
            gt_nodes (Union[List[str], List[List[float]]]): List of ground truth nodes either as strings or embeddings.
            pred_nodes (Union[List[str], List[List[float]]]): List of predicted nodes either as strings or embeddings.
            similarity_metric (Literal[&#39;jaccard_similarity_coefficient&#39;, &#39;overlap_coefficient&#39;, &#39;levenshtein_similarity&#39;, &#39;dice_coefficient&#39;, &#39;jaro_winkler_similarity&#39;, &#39;cosine_similarity&#39;]): Similarity metric used to compute node similarity
            threshold (float, optional): threshold at which two nodes are considered equal. Defaults to 1.

        Raises:
            NotImplementedError: Raise error in case this method is not implemented by a sub class

        Returns:
            float: the calculated NSC based on the selected similarity metric
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_kg_matrix(self,
                        gt_nodes: Union[List[str], List[List[float]]], 
                        pred_nodes: Union[List[str], List[List[float]]],
                        similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                                    'levenshtein_similarity', 'dice_coefficient', 
                                                    'jaro_winkler_similarity', 'cosine_similarity'],
                        threshold: float=1) -> tuple[np.ndarray, np.ndarray]:
        """Creates a adjecency matrix out of the two provided lists of nodes

        Args:
            gt_nodes (Union[List[str], List[List[float]]]): List of ground truth nodes either as strings or embeddings.
            pred_nodes (Union[List[str], List[List[float]]]): List of predicted nodes either as strings or embeddings.
            similarity_metric (Literal[&#39;jaccard_similarity_coefficient&#39;, &#39;overlap_coefficient&#39;, &#39;levenshtein_similarity&#39;, &#39;dice_coefficient&#39;, &#39;jaro_winkler_similarity&#39;, &#39;cosine_similarity&#39;]): Similarity metric used to compute node similarity
            threshold (float, optional): threshold at which two nodes are considered equal. Defaults to 1.

        Raises:
            NotImplementedError: Raise error in case this method is not implemented by a sub class

        Returns: 
            tuple[np.ndarray, np.ndarray]: the adjacency matrix of the ground truth graph and the mapped predicted graph as an adjacency matrix
        """
        raise NotImplementedError("Subclasses must implement this method")

    def adjacency_matrix_similarity(self,
                                    similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                                               'levenshtein_similarity', 'dice_coefficient', 
                                                               'jaro_winkler_similarity', 'cosine_similarity'],
                                    threshold: float=1) -> float:
        """calculates the adjacency matrix similarity (AMS) between the ground truth knowledge graph and the predicted knowledge graph

        Args:
            similarity_metric (str): type of metric to be used for the node similarity calculation
            threshold (float, optional): threshold at which two nodes are considered equal. Defaults to 1 

        Returns:
            float: the calculated adjacency matrix similarity
        """
        # wether to use nodes in string form or embedded
        if similarity_metric == "cosine_similarity":
            # embedd nodes if necessary
            if not self.embedded_gt_nodes:
                self.embedded_gt_nodes = self.embed_nodes(self.kg_gt)
            if not self.embedded_pred_nodes:
                self.embedded_pred_nodes = self.embed_nodes(self.kg_pred)

            gt_nodes = list(copy.deepcopy(self.embedded_gt_nodes).values())
            pred_nodes = list(copy.deepcopy(self.embedded_pred_nodes).values())
        else:
            gt_nodes = list(copy.deepcopy(self.kg_gt.nodes))
            pred_nodes = list(copy.deepcopy(self.kg_pred.nodes))

        # create adjacency matrix of ground truth and mapped predicted graph
        gt_matrix, pred_matrix = self.create_kg_matrix(gt_nodes=gt_nodes, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)

        weighted_correlation = 0
        total_row_weight = 0

        for i in range(gt_matrix.shape[0]):
            # calculate the ground truth row weight
            if np.sum(gt_matrix) == 0:
                row_weight = 0
            else:
                row_weight = np.sum(gt_matrix[i]) / np.sum(gt_matrix)
            total_row_weight += row_weight
            

            # calculate row similarity using the pearsonr coeficient
            if np.std(pred_matrix[i]) == 0 or np.std(gt_matrix[i]) == 0:
                correlation = 0
            else:
                correlation = abs(pearsonr(pred_matrix[i], gt_matrix[i])[0])

            weighted_correlation += row_weight * correlation

        if total_row_weight == 0:
            return 0 

        adjacency_matrix_similarity = (weighted_correlation / total_row_weight)

        return f"Calculated a adjacency matrix similarity of {adjacency_matrix_similarity} based on the {similarity_metric}!"
    
    def structural_node_similarity_coefficient(self, embedding_type: Literal["node2vec", "deep_walk"], directed: bool=True, compute_diff: bool=True) -> float:
        """Calculates the node similarity coefficient using structually embedded nodes.

        Args:
            embedding_type (Literal[&quot;node2vec&quot;, &quot;deep_walk&quot;]): Defines wich algorithm to use to create node embeddings
            directed (bool, optional): Wether the graph should be computed using directed or undirected edges. Defaults to True.
            compute_diff (bool, optional): Wether or not length descrepancy between the two graphs node count should be factored in. Defaults to True.

        Returns:
            float: calculated NSC based on structually embedded nodes
        """
        if embedding_type == "node2vec":
            kge_gt = self.node2vec("ground_truth", directed=directed)
            kge_pred = self.node2vec("prediction", directed=directed)
        else:
            kge_gt = self.deep_walk("ground_truth", directed=directed)
            kge_pred = self.deep_walk("prediction", directed=directed)
        
        similarity_coefficient_sum, _ = self.calculate_node_similarity_coefficient(gt_nodes=kge_gt, pred_nodes=kge_pred, similarity_metric="cosine_similarity")

        # wether missing nodes or excessive nodes in the predicted knowledge graph should influence the accuracy score
        if compute_diff:
            diff = abs(len(list(self.kg_gt.nodes)) - len(list(self.kg_pred.nodes)))
            node_similarity_coefficient = similarity_coefficient_sum / (len(list(self.kg_gt.nodes)) + diff)
        else:
            node_similarity_coefficient = similarity_coefficient_sum / min(len(list(self.kg_gt.nodes)), len(list(self.kg_pred.nodes)))

        return f"Calculated a node similarity coefficient of {round(node_similarity_coefficient, 3)} based on the {embedding_type} embedding!"

class Metrics(Eval):
    """Class for the evaluation of knowledge graphs. 
       Metrics: 
       - node similarity coefficient (NSC)
       - adjacency matrix similarity (AMS)

       Source: arxiv.org/abs/2408.14397
    """
    def create_kg_matrix(self,
                            gt_nodes: Union[List[str], List[List[float]]], 
                            pred_nodes: Union[List[str], List[List[float]]],
                            similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                                      'levenshtein_similarity', 'dice_coefficient', 
                                                      'jaro_winkler_similarity', 'cosine_similarity'],
                            threshold: float) -> tuple[np.array, np.array]:
        """maps the nodes of the predicted knowledge graph to the ground truth knowledge graph and transforms both to adjacency matrices

        Args:
            gt_nodes (Union[List[str], List[List[float]]]): list of ground truth node names in either string or embedded format
            pred_nodes (Union[List[str], List[List[float]]]): list of predicted node names in either string or embedded format
            similarity_metric (Literal[&#39;jaccard_similarity_coefficient&#39;, &#39;overlap_coefficient&#39;, &#39;levenshtein_similarity&#39;, &#39;dice_coefficient&#39;, &#39;jaro_winkler_similarity&#39;, &#39;cosine_similarity&#39;]): similarity metric used to determine matches between ground truth node names and predicted node names
            threshold (float): threshold at which two nodes are considered equal.

        Returns:
            tuple[np.array, np.array]: adjacency matrix of the ground truth knowledge graph and the adjacency matrix of the similarity mapped graph
        """        
        # map ground truth node to predicted nodes
        mapped_nodes = []
        for gt_node in tqdm(gt_nodes, desc="Mapping predicted nodes...", leave=False):
            if similarity_metric == "cosine_similarity":
                similar_node, _ = self.get_max_similarity_node(gt_node=gt_node, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)

                # convert embedding back to a string
                similar_node = next((key for key, value in self.embedded_pred_nodes.items() if value == similar_node), None)
            else:
                similar_node, _ = self.get_max_similarity_node(gt_node=gt_node, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)
            mapped_nodes.append(similar_node)

        # create adjacency matrices initialized with 0
        n = len(self.kg_gt.nodes)
        gt_matrix = np.zeros((n, n))
        pred_matrix = np.zeros((n, n))

        # map node name to an index number
        gt_node_to_index = {gt_node: i for i, gt_node in enumerate(self.kg_gt.nodes)}
        pred_node_to_index = {pred_node: i for i, pred_node in enumerate(mapped_nodes)}

        # create adjacency matrix for ground truth knowledge graph
        for triplet in tqdm(list(self.kg_gt.edges(data=True)), desc="Creating ground truth kg adjacency matrix...", leave=False):
            source, relation, target = triplet[0], triplet[2]["relation"], triplet[1]
            if source in gt_node_to_index.keys() and target in gt_node_to_index.keys():
                source_idx = gt_node_to_index[source]
                target_idx = gt_node_to_index[target]
            
                if relation:
                    gt_matrix[source_idx, target_idx] = 1
        
        # create adjacency matrix for predicted knowledge graph based on mapped nodes
        for triplet in tqdm(list(self.kg_pred.edges(data=True)), desc="Creating predicted kg adjacency matrix...", leave=False):
            source, relation, target = triplet[0], triplet[2]["relation"], triplet[1]
            if source in pred_node_to_index.keys() and target in pred_node_to_index.keys():
                source_idx = pred_node_to_index[source]
                target_idx = pred_node_to_index[target]
                
                if relation:
                    pred_matrix[source_idx, target_idx] = 1

        return gt_matrix, pred_matrix

    def calculate_node_similarity_coefficient(self, 
        gt_nodes: Union[List[str],List[List[float]]], 
        pred_nodes: Union[List[str],List[List[float]]],
        similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                    'levenshtein_similarity', 'dice_coefficient', 
                                    'jaro_winkler_similarity', 'cosine_similarity'], 
        threshold: float=1) -> float:
        """matches the ground truth node names to the most similar node name in the prediction and calculates the average similarity across all matched nodes

        Args:
            gt_nodes (Union[List[str], List[List[float]]]): list of ground truth node names in either string or embedded format
            pred_nodes (Union[List[str], List[List[float]]]): list of predicted node names in either string or embedded format
            similarity_metric (Literal[&#39;jaccard_similarity_coefficient&#39;, &#39;overlap_coefficient&#39;, &#39;levenshtein_similarity&#39;, &#39;dice_coefficient&#39;, &#39;jaro_winkler_similarity&#39;, &#39;cosine_similarity&#39;]): similarity metric used to determine matches between ground truth node names and predicted node names
            threshold (float): threshold at which two nodes are considered equal.

        Returns:
            float: The calculated node similarity coefficient score

        Raises:
            Warning: Matching the ground truth knowledge graph to the predicted knowledge graph can lead to inaccurate similarity scores! To get more accurate results set compute_diff=True
        """

        matches = []
        similarity_coefficient_sum = 0
        count = 0

        # map ground truth nodes to predicted nodes
        for gt_node in tqdm(gt_nodes, desc="Mapping similar nodes...", leave=False):
            # case if there are fewer prediction nodes than ground truth
            if count == (self.kg_pred.nodes):                  
                break
            similar_node, max_similarity_coefficient = self.get_max_similarity_node(gt_node=gt_node, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)
            similarity_coefficient_sum += max_similarity_coefficient
            matches.append({"ground_truth": gt_node, "prediction": similar_node, "similarity": max_similarity_coefficient})
            count += 1

        return similarity_coefficient_sum, matches
        
class MetricsOptimized(Eval):
    def create_kg_matrix(self,
                            gt_nodes: Union[List[str], List[List[float]]], 
                            pred_nodes: Union[List[str], List[List[float]]],
                            similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                                      'levenshtein_similarity', 'dice_coefficient', 
                                                      'jaro_winkler_similarity', 'cosine_similarity'],
                            threshold: float) -> tuple[np.array, np.array]:
        """maps the nodes of the predicted knowledge graph to the ground truth knowledge graph and transforms both to adjacency matrices

        Args:
            gt_nodes (Union[List[str], List[List[float]]]): list of ground truth node names in either string or embedded format
            pred_nodes (Union[List[str], List[List[float]]]): list of predicted node names in either string or embedded format
            similarity_metric (Literal[&#39;jaccard_similarity_coefficient&#39;, &#39;overlap_coefficient&#39;, &#39;levenshtein_similarity&#39;, &#39;dice_coefficient&#39;, &#39;jaro_winkler_similarity&#39;, &#39;cosine_similarity&#39;]): similarity metric used to determine matches between ground truth node names and predicted node names
            threshold (float): threshold at which two nodes are considered equal.

        Returns:
            tuple[np.array, np.array]: adjacency matrix of the ground truth knowledge graph and the adjacency matrix of the similarity mapped graph
        """  
        # create adjacency matrices initialized with 0
        n = len(gt_nodes)
        gt_matrix = np.zeros((n, n))
        pred_matrix = np.zeros((n, n))
        # map ground truth node to predicted nodes
        mapped_nodes = []
        truth_nodes = []
        while True:
            if len(gt_nodes) == 0:
                break
            candidates = []

            # get most similar predicted node for each ground truth node
            for gt_node in gt_nodes:
                similar_node, max_similarity_coefficient = self.get_max_similarity_node(gt_node=gt_node, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)
                candidates.append({"gt_node":gt_node, "similar_node": similar_node, "max_similarity_coefficient": max_similarity_coefficient})

            # count how often each similar predicted node occurs
            if similarity_metric == "cosine_similarity":
                mapping = dict(Counter(tuple(item['similar_node']) for item in candidates))
            else:
                mapping = dict(Counter(item['similar_node'] for item in candidates))
            
            # avoid error in case of shorter prediction than ground truth
            if len(mapping) == 1 and () in mapping.keys():
                break
            if len(mapping) == 1 and '' in mapping.keys():
                break

            # for each extracted similar node 
            for item in mapping:
                if isinstance(item, tuple):
                    item = list(item)
                
                # get all candidates that have the similar predicted node in common
                similar_nodes = [entry for entry in candidates if entry['similar_node'] == item]

                # get the candidate with the highest similarity of the ground truth and the predicted node
                max_sim_dict = max(similar_nodes, key=lambda x: x['max_similarity_coefficient'])

                # add candidate ground truth and prediction to proper lists before removal from all ground truth and predicted nodes
                if similarity_metric == "cosine_similarity":
                    truth_nodes.append(next((k for k, v in self.embedded_gt_nodes.items() if v == max_sim_dict["gt_node"]), None))
                    mapped_nodes.append(next((k for k, v in self.embedded_pred_nodes.items() if v == max_sim_dict["similar_node"]), None))
                else:
                    truth_nodes.append(max_sim_dict["gt_node"])
                    mapped_nodes.append(max_sim_dict["similar_node"])

                # remove the canditate with the highest similarity
                gt_nodes.remove(max_sim_dict["gt_node"])
                pred_nodes.remove(max_sim_dict["similar_node"])

        # map node name to an index number
        gt_node_to_index = {gt_node: i for i, gt_node in enumerate(truth_nodes)}
        pred_node_to_index = {pred_node: i for i, pred_node in enumerate(mapped_nodes)}

        # create adjacency matrix for ground truth knowledge graph
        for triplet in tqdm(list(self.kg_gt.edges(data=True)), desc="Creating ground truth kg adjacency matrix...", leave=False):
            source, relation, target = triplet[0], triplet[2]["relation"], triplet[1]
            if source in gt_node_to_index.keys() and target in gt_node_to_index.keys():
                source_idx = gt_node_to_index[source]
                target_idx = gt_node_to_index[target]
            
                if relation:
                    gt_matrix[source_idx, target_idx] = 1
        
        # create adjacency matrix for predicted knowledge graph based on mapped nodes
        for triplet in tqdm(list(self.kg_pred.edges(data=True)), desc="Creating predicted kg adjacency matrix...", leave=False):
            source, relation, target = triplet[0], triplet[2]["relation"], triplet[1]
            if source in pred_node_to_index.keys() and target in pred_node_to_index.keys():
                source_idx = pred_node_to_index[source]
                target_idx = pred_node_to_index[target]
                
                if relation:
                    pred_matrix[source_idx, target_idx] = 1
        return gt_matrix, pred_matrix
    
    def calculate_node_similarity_coefficient(self, 
        gt_nodes: Union[List[str],List[List[float]]], 
        pred_nodes: Union[List[str],List[List[float]]],
        similarity_metric: Literal['jaccard_similarity_coefficient', 'overlap_coefficient', 
                                    'levenshtein_similarity', 'dice_coefficient', 
                                    'jaro_winkler_similarity', 'cosine_similarity'], 
        threshold: float=1,) -> float:
        """matches the ground truth node names to the most similar node name in the prediction avoiding duplicated assignment of predicted node names and calculates the average similarity across all matched nodes

        Args:
            gt_nodes (Union[List[str], List[List[float]]]): list of ground truth node names in either string or embedded format
            pred_nodes (Union[List[str], List[List[float]]]): list of predicted node names in either string or embedded format
            similarity_metric (Literal[&#39;jaccard_similarity_coefficient&#39;, &#39;overlap_coefficient&#39;, &#39;levenshtein_similarity&#39;, &#39;dice_coefficient&#39;, &#39;jaro_winkler_similarity&#39;, &#39;cosine_similarity&#39;]): similarity metric used to determine matches between ground truth node names and predicted node names
            threshold (float): threshold at which two nodes are considered equal.

        Returns:
            float: The calculated node similarity coefficient score

        Raises:
            Warning: Matching the ground truth knowledge graph to the predicted knowledge graph can lead to inaccurate similarity scores! To get more accurate results set compute_diff=True
        """
        
        matches = []
        similarity_coefficient_sum = 0
        while True:
            if len(gt_nodes) == 0:
                break
            candidates = []

            # get most similar predicted node for each ground truth node
            for gt_node in gt_nodes:
                similar_node, max_similarity_coefficient = self.get_max_similarity_node(gt_node=gt_node, pred_nodes=pred_nodes, similarity_metric=similarity_metric, threshold=threshold)
                candidates.append({"gt_node":gt_node, "similar_node": similar_node, "max_similarity_coefficient": max_similarity_coefficient})
            
            # count how often each similar predicted node occurs    
            if similarity_metric == "cosine_similarity":
                mapping = dict(Counter(tuple(item['similar_node']) for item in candidates))
            else:
                mapping = dict(Counter(item['similar_node'] for item in candidates))

            # avoid error in case of shorter prediction than ground truth
            if len(mapping) == 1 and () in mapping.keys():
                break
            if len(mapping) == 1 and '' in mapping.keys():
                break


            # for each extracted similar node 
            for item in mapping:
                if isinstance(item, tuple):
                    item = list(item)
                # get all candidates that have the similar predicted node in common
                similar_nodes = [entry for entry in candidates if entry['similar_node'] == item]

                # get the candidate with the highest similarity of the ground truth and the predicted node
                max_sim_dict = max(similar_nodes, key=lambda x: x['max_similarity_coefficient'])

                matches.append({"ground_truth": max_sim_dict["gt_node"], "prediction": max_sim_dict["similar_node"], "similarity": max_sim_dict["max_similarity_coefficient"]})

                # remove the canditate with the highest similarity
                gt_nodes.remove(max_sim_dict["gt_node"])
                pred_nodes.remove(max_sim_dict["similar_node"])

                similarity_coefficient_sum += max_sim_dict["max_similarity_coefficient"]

        return similarity_coefficient_sum, matches