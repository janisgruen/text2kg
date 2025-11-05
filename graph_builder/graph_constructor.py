import networkx as nx
import pickle
import warnings

class KnowledgeGraph:
    """ class to facilitate the knowledge graph creation using networkx
    """
    def __init__(self) -> None:
        """ initialization of the class
        """
        self.graph = nx.MultiDiGraph()

    def __check_node_structure(self, nodes: dict) -> None:
        """ Checks if the dictionary that contains the nodes has a valid structure

        Args:
            nodes (dict): Dictionary containing a list of nodes

        Raises:
            InvalidStructureError: Missing 'nodes' key.
            InvalidStructureError: 'nodes' should be a list.
            InvalidStructureError: Each item in 'nodes' should be a dictionary.
        """
        if "nodes" not in nodes.keys():
            raise InvalidStructureError("Missing 'nodes' key.")
        
        if not isinstance(nodes["nodes"], list):
            raise InvalidStructureError("'nodes' should be a list.")
        
        if not all(isinstance(node, dict) for node in nodes["nodes"]):
            raise InvalidStructureError("Each item in 'nodes' should be a dictionary.")
    
    def __check_relations_structure(self, relations: dict) -> None:
        """ Checks if the dictionary that contains relationships has a valid structure

        Args:
            relationships (dict): Dictionary containing relationships

        Raises:
            InvalidStructureError: Missing 'relationships' key.
            InvalidStructureError: 'relationships' should be a list.
            InvalidStructureError: Each item in 'relationships' should be a dictionary.
        """
        if "relations" not in relations.keys():
            raise InvalidStructureError("Missing 'relationships' key.")
        
        if not isinstance(relations["relations"], list):
            raise InvalidStructureError("'relations' should be a list.")
        
        if not all(isinstance(relations, dict) for relations in relations["relations"]):
            raise InvalidStructureError("Each item in 'relations' should be a dictionary.")

    def __check_kg_structure(self, kg: dict) -> None:
        """checks if the given knowledge graph has the correct structure:
        {"nodes": [{...}, {...}, ...], "relations": [{...}, {...}, ...]}
        
        Args:
            kg (dict): the knowledge graph that will be validated

        Raises:
            InvalidStructureError: Missing 'nodes' key.
            InvalidStructureError: Missing 'relations' key.
            InvalidStructureError: 'nodes' should be a list.
            InvalidStructureError: Each item in 'nodes' should be a dictionary.
            InvalidStructureError: 'relations' should be a list.
            InvalidStructureError: Each item in 'relations' should be a dictionary.
        """    
        if "nodes" not in kg.keys():
            raise InvalidStructureError("Missing 'nodes' key.")
        
        if "relations" not in kg.keys():
            raise InvalidStructureError("Missing 'relations' key.")
        
        self.__check_node_structure(nodes=kg["nodes"])
        self.__check_relations_structure(nodes=kg["relations"])

    def addNodes(self, nodes: dict) -> None:
        """ Adds all nodes to the graph that are given in the nodes dictionary

        Args:
            nodes (dict): Nodes that will be added to the graph
        """
        self.__check_node_structure(nodes=nodes)
        for node in nodes["nodes"]:
            if "name" in node.keys():
                if "type" not in node.keys():
                    self.addNode(name=node["name"], type="NaN")
                else:
                    self.addNode(name=node["name"], type=node["type"])

    def addNode(self, name: str, type: str) -> None:
        """ Adds a single node to the graph 

        Args:
            name (str): name of the node to be added
            type (str): type of the node that will be added as additional parameter
        """
        if len(name) > 0:
            if not self.graph.has_node(n=name):
                self.graph.add_node(node_for_adding=name)
            
            # if no type is given assign type to NaN
            if type == {}:
                type = "NaN"

            if "type" not in self.graph.nodes[name]:
                self.graph.nodes[name]["type"] = type

    def addEdges(self, edges: dict, multi_edges: bool=False) -> None:
        """ Adds all edges to the graph that are given in the eges dictionary

        Args:
            edges (dict): Edges that will be added to the graph
            multi_edges (bool): Wether or not there can be multiple edges between two nodes. Defaults to False.
        """
        self.__check_relations_structure(relations=edges)
        for edge in edges["relations"]:
            if "source" in edge.keys() and "relation" in edge.keys() and "target" in edge.keys():
                self.addEdge(source=edge["source"], relation=edge["relation"], target=edge["target"], multi_edge=multi_edges)

    def addEdge(self, source: str, relation: str, target: str, multi_edge: bool=False) -> None:
        """ Adds a single edge to the graph

        Args:
            source (str): Name of the source node that is connected by the relationship
            relation (str): Type of the relationship between the source and the target node 
            target (str): Name of the target node that is connected by the relationship
            multi_edges (bool): Wether or not there can be multiple edges between two nodes. Defaults to False.
        """
        if len(source) > 0 and len(relation) > 0 and len(target) > 0:
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(u_for_edge=source, relation=relation, v_for_edge=target)
            elif multi_edge:
                self.graph.add_edge(u_for_edge=source, relation=relation, v_for_edge=target)

    def construct_from_dict(self, kg: dict, refine: bool = True) -> None:
        """ constructs a knowledge graph based on the given dictionary

        Args:
            kg (dict): Knowledge graph in a dictionary form
            refine (bool, optional): Wether or not nodes will be deleted that do not occur in any relationship. Defaults to True.
        """
        # check for correct knowledge graph structure
        self.__check_kg_structure(kg=kg)

        nodes, relationships = kg.items()

        # add nodes to knowledge graph
        if len(nodes) > 0:
            for node in nodes[1]:
                self.addNode(name=node["name"], type=node["type"])

        # add edges to the knowledge graph
        if len(relationships) > 0:
            for relationship in relationships[1]:
                if all(key in relationship.keys() for key in ("source", "relation", "target")):
                    self.addEdge(source=relationship["source"], relation=relationship["relation"], target=relationship["target"])

        # remove all nodes that are not present in any relationship
        if refine:
            self.refine()   

    def refine(self) -> None:
        """ Refines the knowledge graph by removing every node that is not present in any relationship
        """
        nodes_to_remove = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        self.graph.remove_nodes_from(nodes_to_remove)
    
    def merge_nodes(self, node_to_keep: str, node_to_merge: str) -> None:
        """ Merges node_to_merge into node_to_keep together with all the node´s edges 

        Args:
            node_to_keep (str): The node to be maintained in the graph
            node_to_merge (str): The node to merge into the other node
        """
        # get all neighbors of the node_to_merge
        for neighbor, edge_data in self.graph[node_to_merge].items():
            if neighbor != node_to_keep:
                self.graph.add_edge(u_for_edge=node_to_keep, v_for_edge=neighbor, **edge_data)
            else:
                warnings.warn(f"Warning: Edge between {node_to_keep} and {node_to_merge} will be lost to prevent self-loops!")
        self.graph.remove_node(n=node_to_merge)

    def get_connected_neighbours(self, node: str, radius: int=1, directed: bool=False) -> nx.MultiDiGraph:
        """ returns a subgraph containing all nodes within a specified distance from a given starting node

        Args:
            node (str): The starting node from which the distance is measured
            radius (int, optional): The maximum distance from the starting node to include nodes in the subgraph. Defaults to 1.
            directed (bool, optional): Wether or not the graph will be treated as directed. Defaults to False.

        Returns:
            nx.MultiDiGraph: A subgraph that contains all nodes within the distance including all edges connecting these nodes
        """
        if directed:
            subgraph = nx.ego_graph(G=self.graph, n=node, radius=radius, undirected=False)
        else:
            subgraph = nx.ego_graph(G=self.graph, n=node, radius=radius, undirected=True)

        return subgraph
    
    def get_subgraph(self, nodes: list[str]) -> nx.MultiDiGraph:
        """ Extracts a subgraph based on the given nodes list with all edges between these nodes

        Args:
            nodes (list[str]): list of node names that the subgraph should contain

        Returns:
            nx.MultiDiGraph: subgraph containing the given nodes and edges that connect those
        """
        return self.graph.subgraph(nodes=nodes)

    def load(self, file_path: str) -> None:
        """ imports an existing knowledge graph from a pickle file

        Args:
            file_path (str): the file path to the pickle file of the knowledge graph that will be loaded

        Raises:
            ValueError: Given file is not from required file type .pickle or .pkl!
            Exception: Knowledge graph import failed!
        """

        # Check for valid file extension
        if not (file_path.endswith('.pkl') or file_path.endswith('.pickle')):
            raise ValueError("Given file is not from required file type .pickle or .pkl!")
        
        # check if the file can be loaded as a pickle
        try:
            with open(file_path, 'rb') as f:
                self.graph = pickle.load(f)

        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            raise Exception("Knowledge graph import failed!")

    def save(self, file_path: str) -> None:
        """ saves the knowledge graph to the given file path

        Args:
            file_path (str): path to pickle file where the knowledge graph will be saved

        Raises:
            ValueError: File path must end with .pkl or .pickle!
        """
        # Check for valid file extension
        if not (file_path.endswith('.pkl') or file_path.endswith('.pickle')):
            raise ValueError("File path must end with .pkl or .pickle!")

        # Save the knowledge graph to the file
        with open(file_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Data has been successfully saved to {file_path}.")

    def get_graph(self) -> nx.MultiDiGraph:
        """ returns the entire knowledge graph

        Returns:
            nx.MultiDiGraph: The knowledge graph that is used by the class 
        """
        return self.graph

class InvalidStructureError(Exception):
    """Custom exception raised when the structure of a knowledge graph is invalid."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)