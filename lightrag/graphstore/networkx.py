
import os
import networkx as nx
from dataclasses import dataclass
from typing import List, Union, Tuple
from lightrag.utils import logger
from lightrag.base import BaseGraphStorage

@dataclass
class NetworkXStorage(BaseGraphStorage):
    def __post_init__(self):
        self._graph_xml_file = os.path.join(
            self.global_config['working_dir'], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = self.load_nx_graph(file_name=self._graph_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graph_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()

    @staticmethod
    def load_nx_graph(file_name: Union[str, None]) -> nx.Graph:
        """Read graph from .graphxml file"""
        if file_name is None:
            return None
        if os.path.exists(path=file_name):
            return nx.read_graphml(file_name)
        return None
    
    @staticmethod
    def write_nx_graph(file_name: str, G: nx.Graph) -> None:
        """Write graph to .graphxml file"""
        logger.info(f"Writing graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        nx.write_graphml(G, file_name)
    
    async def has_node(self, node_id: str) -> bool:
        """Check if node exsits from node_id"""
        return self._graph.has_node(n=node_id)
    
    async def get_all_nodes(self) -> list:
        """Get all nodes from graph"""
        return self._graph.nodes
    
    async def has_edge(self, src_node_id: str, tgt_node_id: str) -> bool:
        """Check if edge exists from src_node and tgt_node"""
        return self._graph.has_edge(u=src_node_id, v=tgt_node_id)
    
    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Get node from graph through node_id"""
        return self._graph.nodes.get(node_id)
     
    async def get_degree_node(self, node_id: str) -> int:
        """Get degree node from Graph through node_id"""
        if self.has_node(node_id=node_id):
            return self._graph.degree(node_id)
    
    async def get_degree_between_nodes(self, src_node_id: str, tgt_node_id) -> int:
        """Get degree between 2 nodes from Graph through src_node_id and tgt_node_id"""
        if self.has_node(src_node_id) and self.has_node(tgt_node_id):
            return self._graph.degree(src_node_id) + self._graph.degree(tgt_node_id)

    async def get_edge(self, src_node_id: str, tgt_node_id: str)-> Union[dict, None]:
        """Get edge from graph through src_node_id and tgt_node_id"""
        if self.has_edge(src_node_id, tgt_node_id):
            return self._graph.edges.get((src_node_id, tgt_node_id))
    
    async def get_all_edges(self) -> list:
        """Get all edges from graph"""
        return self._graph.edges

    async def get_node_edges(self, node_id: str) -> List[Tuple[str, str]]:
        """Get edges of node_id from graph"""
        if self.has_node(node_id):
            return list(self._graph.edges(node_id))
        return None
    
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Update a node to graph based on the specificed node_id"""
        self._graph.add_node(node_id, **node_data)
    
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from graph based on the specified node_id"""
        if self.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")
    
    async def upsert_edge(self, src_node_id: str, tgt_node_id: str, edge_data: dict[str, str]) -> None:
        """Update a edge to graph based on the specificed src_node_id and tgt_node_id"""
        self._graph.add_edge(src_node_id, tgt_node_id, **edge_data)
    
    async def delete_edge(self, src_node_id: str, tgt_node_id: str) -> None:
        """Delete a edge from graph based on the specified src_node_id and tgt_node_id"""
        if self.has_edge(src_node_id, tgt_node_id):
            self._graph.remove_edge(u=src_node_id, v=tgt_node_id)
            logger.info(f"Edge between {src_node_id} and {tgt_node_id} deleted from the graph.")
        else:
            logger.warning(f"Edge between {src_node_id} and {tgt_node_id} not found in the graph for deletion.")
    
