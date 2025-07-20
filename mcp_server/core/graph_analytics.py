"""
Graph Analytics and Reasoning System

This module provides comprehensive graph analytics capabilities for knowledge graphs,
including shortest path algorithms, centrality measures, community detection, and
subgraph analysis for enhanced relationship discovery and graph reasoning.
"""

from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import math

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("NetworkX loaded successfully for graph analytics")
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    logger = logging.getLogger(__name__)
    logger.warning("NetworkX not available - using basic graph operations")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.cluster import SpectralClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SpectralClustering = None


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.0
    centrality_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    weight: float = 1.0


@dataclass
class PathResult:
    """Result of path finding operations"""
    path: List[str]
    length: int
    total_weight: float
    edge_types: List[str]
    confidence: float


@dataclass
class Community:
    """Represents a detected community in the graph"""
    id: str
    nodes: Set[str]
    density: float
    modularity: float
    central_nodes: List[str]
    common_themes: List[str] = field(default_factory=list)


class GraphAnalytics:
    """Advanced graph analytics and reasoning engine"""
    
    def __init__(self, use_networkx: bool = True):
        """Initialize the graph analytics engine"""
        self.use_networkx = use_networkx and NETWORKX_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Graph storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # NetworkX graph for advanced operations
        self.nx_graph = None
        if self.use_networkx:
            self.nx_graph = nx.DiGraph()
            self.logger.info("Initialized with NetworkX support")
        else:
            self.logger.info("Initialized with basic graph operations")
        
        # Analytics cache
        self._centrality_cache: Dict[str, Dict[str, float]] = {}
        self._communities_cache: List[Community] = []
        self._cache_valid = False
    
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph"""
        try:
            self.nodes[node.id] = node
            
            if self.use_networkx and self.nx_graph is not None:
                self.nx_graph.add_node(
                    node.id,
                    entity_type=node.entity_type,
                    **node.properties
                )
            
            self._invalidate_cache()
            return True
        except Exception as e:
            self.logger.error(f"Error adding node {node.id}: {e}")
            return False
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph"""
        try:
            # Ensure both nodes exist
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                self.logger.warning(f"Cannot add edge: missing nodes {edge.source_id} or {edge.target_id}")
                return False
            
            edge_key = (edge.source_id, edge.target_id)
            self.edges[edge_key] = edge
            self.adjacency[edge.source_id].add(edge.target_id)
            
            if self.use_networkx and self.nx_graph is not None:
                self.nx_graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relationship_type=edge.relationship_type,
                    weight=edge.weight,
                    confidence=edge.confidence,
                    **edge.properties
                )
            
            self._invalidate_cache()
            return True
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.source_id} -> {edge.target_id}: {e}")
            return False
    
    def _invalidate_cache(self):
        """Invalidate analytics cache when graph changes"""
        self._cache_valid = False
        self._centrality_cache.clear()
        self._communities_cache.clear()
    
    def find_shortest_path(self, source_id: str, target_id: str, 
                          max_length: int = 6) -> Optional[PathResult]:
        """Find shortest path between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        if self.use_networkx and self.nx_graph is not None:
            return self._find_shortest_path_networkx(source_id, target_id)
        else:
            return self._find_shortest_path_basic(source_id, target_id, max_length)
    
    def _find_shortest_path_networkx(self, source_id: str, target_id: str) -> Optional[PathResult]:
        """Find shortest path using NetworkX"""
        try:
            path = nx.shortest_path(self.nx_graph, source_id, target_id, weight='weight')
            total_weight = nx.shortest_path_length(self.nx_graph, source_id, target_id, weight='weight')
            
            # Extract edge types and confidence
            edge_types = []
            min_confidence = 1.0
            
            for i in range(len(path) - 1):
                edge_data = self.nx_graph[path[i]][path[i + 1]]
                edge_types.append(edge_data.get('relationship_type', 'unknown'))
                min_confidence = min(min_confidence, edge_data.get('confidence', 1.0))
            
            return PathResult(
                path=path,
                length=len(path) - 1,
                total_weight=total_weight,
                edge_types=edge_types,
                confidence=min_confidence
            )
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            self.logger.error(f"Error finding path with NetworkX: {e}")
            return None
    
    def _find_shortest_path_basic(self, source_id: str, target_id: str, max_length: int) -> Optional[PathResult]:
        """Find shortest path using basic BFS"""
        if source_id == target_id:
            return PathResult(path=[source_id], length=0, total_weight=0.0, edge_types=[], confidence=1.0)
        
        queue = deque([(source_id, [source_id], 0.0, [], 1.0)])
        visited = {source_id}
        
        while queue:
            current_id, path, weight, edge_types, confidence = queue.popleft()
            
            if len(path) > max_length:
                continue
            
            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id in visited:
                    continue
                
                edge_key = (current_id, neighbor_id)
                edge = self.edges.get(edge_key)
                if not edge:
                    continue
                
                new_path = path + [neighbor_id]
                new_weight = weight + edge.weight
                new_edge_types = edge_types + [edge.relationship_type]
                new_confidence = min(confidence, edge.confidence)
                
                if neighbor_id == target_id:
                    return PathResult(
                        path=new_path,
                        length=len(new_path) - 1,
                        total_weight=new_weight,
                        edge_types=new_edge_types,
                        confidence=new_confidence
                    )
                
                visited.add(neighbor_id)
                queue.append((neighbor_id, new_path, new_weight, new_edge_types, new_confidence))
        
        return None
    
    def find_all_paths(self, source_id: str, target_id: str, 
                      max_length: int = 4, max_paths: int = 10) -> List[PathResult]:
        """Find multiple paths between nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        
        if self.use_networkx and self.nx_graph is not None:
            return self._find_all_paths_networkx(source_id, target_id, max_length, max_paths)
        else:
            return self._find_all_paths_basic(source_id, target_id, max_length, max_paths)
    
    def _find_all_paths_networkx(self, source_id: str, target_id: str, 
                                max_length: int, max_paths: int) -> List[PathResult]:
        """Find all paths using NetworkX"""
        try:
            paths = []
            path_generator = nx.all_simple_paths(self.nx_graph, source_id, target_id, cutoff=max_length)
            
            for i, path in enumerate(path_generator):
                if i >= max_paths:
                    break
                
                # Calculate path metrics
                total_weight = 0.0
                edge_types = []
                min_confidence = 1.0
                
                for j in range(len(path) - 1):
                    edge_data = self.nx_graph[path[j]][path[j + 1]]
                    total_weight += edge_data.get('weight', 1.0)
                    edge_types.append(edge_data.get('relationship_type', 'unknown'))
                    min_confidence = min(min_confidence, edge_data.get('confidence', 1.0))
                
                paths.append(PathResult(
                    path=path,
                    length=len(path) - 1,
                    total_weight=total_weight,
                    edge_types=edge_types,
                    confidence=min_confidence
                ))
            
            # Sort by length then weight
            paths.sort(key=lambda p: (p.length, p.total_weight))
            return paths
        except Exception as e:
            self.logger.error(f"Error finding all paths with NetworkX: {e}")
            return []
    
    def _find_all_paths_basic(self, source_id: str, target_id: str, 
                             max_length: int, max_paths: int) -> List[PathResult]:
        """Find all paths using basic DFS"""
        paths = []
        
        def dfs(current_id: str, path: List[str], weight: float, 
                edge_types: List[str], confidence: float, visited: Set[str]):
            if len(paths) >= max_paths or len(path) > max_length:
                return
            
            if current_id == target_id and len(path) > 1:
                paths.append(PathResult(
                    path=path.copy(),
                    length=len(path) - 1,
                    total_weight=weight,
                    edge_types=edge_types.copy(),
                    confidence=confidence
                ))
                return
            
            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id in visited:
                    continue
                
                edge_key = (current_id, neighbor_id)
                edge = self.edges.get(edge_key)
                if not edge:
                    continue
                
                visited.add(neighbor_id)
                path.append(neighbor_id)
                edge_types.append(edge.relationship_type)
                
                dfs(neighbor_id, path, weight + edge.weight, 
                    edge_types, min(confidence, edge.confidence), visited)
                
                # Backtrack
                path.pop()
                edge_types.pop()
                visited.remove(neighbor_id)
        
        dfs(source_id, [source_id], 0.0, [], 1.0, {source_id})
        
        # Sort by length then weight
        paths.sort(key=lambda p: (p.length, p.total_weight))
        return paths
    
    def calculate_centrality_measures(self, force_recalculate: bool = False) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for all nodes"""
        if self._cache_valid and not force_recalculate and self._centrality_cache:
            return self._centrality_cache
        
        centrality_results = {}
        
        if self.use_networkx and self.nx_graph is not None:
            centrality_results = self._calculate_centrality_networkx()
        else:
            centrality_results = self._calculate_centrality_basic()
        
        # Update node importance scores
        for node_id, centralities in centrality_results.items():
            if node_id in self.nodes:
                # Combined importance score (weighted average of centralities)
                importance = (
                    centralities.get('degree', 0.0) * 0.3 +
                    centralities.get('betweenness', 0.0) * 0.4 +
                    centralities.get('pagerank', 0.0) * 0.3
                )
                self.nodes[node_id].importance_score = importance
                self.nodes[node_id].centrality_scores = centralities
        
        self._centrality_cache = centrality_results
        return centrality_results
    
    def _calculate_centrality_networkx(self) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures using NetworkX"""
        try:
            results = {}
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.nx_graph)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.nx_graph, weight='weight')
            
            # PageRank
            pagerank = nx.pagerank(self.nx_graph, weight='weight')
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(self.nx_graph, distance='weight')
            
            # Combine results
            for node_id in self.nodes:
                results[node_id] = {
                    'degree': degree_centrality.get(node_id, 0.0),
                    'betweenness': betweenness_centrality.get(node_id, 0.0),
                    'pagerank': pagerank.get(node_id, 0.0),
                    'closeness': closeness_centrality.get(node_id, 0.0)
                }
            
            self.logger.info("Calculated centrality measures using NetworkX")
            return results
        except Exception as e:
            self.logger.error(f"Error calculating centrality with NetworkX: {e}")
            return self._calculate_centrality_basic()
    
    def _calculate_centrality_basic(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic centrality measures without NetworkX"""
        results = {}
        n_nodes = len(self.nodes)
        
        if n_nodes == 0:
            return results
        
        # Calculate degree centrality
        for node_id in self.nodes:
            out_degree = len(self.adjacency[node_id])
            in_degree = sum(1 for neighbors in self.adjacency.values() if node_id in neighbors)
            total_degree = out_degree + in_degree
            
            # Normalized degree centrality
            degree_centrality = total_degree / (2 * (n_nodes - 1)) if n_nodes > 1 else 0.0
            
            # Basic PageRank approximation (degree-based)
            pagerank_approx = total_degree / sum(
                len(neighbors) for neighbors in self.adjacency.values()
            ) if any(self.adjacency.values()) else 1.0 / n_nodes
            
            results[node_id] = {
                'degree': degree_centrality,
                'betweenness': 0.0,  # Complex to calculate without NetworkX
                'pagerank': pagerank_approx,
                'closeness': 0.0  # Complex to calculate without NetworkX
            }
        
        self.logger.info("Calculated basic centrality measures")
        return results
    
    def detect_communities(self, algorithm: str = 'spectral', min_size: int = 3) -> List[Community]:
        """Detect communities in the graph"""
        if self._cache_valid and self._communities_cache:
            return self._communities_cache
        
        if self.use_networkx and self.nx_graph is not None:
            communities = self._detect_communities_networkx(algorithm, min_size)
        else:
            communities = self._detect_communities_basic(min_size)
        
        self._communities_cache = communities
        return communities
    
    def _detect_communities_networkx(self, algorithm: str, min_size: int) -> List[Community]:
        """Detect communities using NetworkX algorithms"""
        try:
            communities = []
            
            if algorithm == 'louvain' and hasattr(nx, 'community') and hasattr(nx.community, 'louvain_communities'):
                # Louvain algorithm for modularity optimization
                community_sets = nx.community.louvain_communities(self.nx_graph.to_undirected())
            elif algorithm == 'spectral' and SKLEARN_AVAILABLE:
                # Spectral clustering
                adjacency_matrix = nx.adjacency_matrix(self.nx_graph.to_undirected())
                if adjacency_matrix.shape[0] > 1:
                    n_clusters = min(max(2, len(self.nodes) // 10), 10)
                    clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
                    labels = clustering.fit_predict(adjacency_matrix.toarray())
                    
                    # Group nodes by cluster
                    cluster_groups = defaultdict(set)
                    for i, label in enumerate(labels):
                        node_id = list(self.nodes.keys())[i]
                        cluster_groups[label].add(node_id)
                    
                    community_sets = [nodes for nodes in cluster_groups.values() if len(nodes) >= min_size]
                else:
                    community_sets = []
            else:
                # Fallback to greedy modularity
                community_sets = nx.community.greedy_modularity_communities(self.nx_graph.to_undirected())
            
            # Create Community objects
            for i, community_nodes in enumerate(community_sets):
                if len(community_nodes) < min_size:
                    continue
                
                community_nodes = set(community_nodes)
                
                # Calculate community metrics
                subgraph = self.nx_graph.subgraph(community_nodes)
                density = nx.density(subgraph)
                
                # Find central nodes (highest degree in community)
                degrees = dict(subgraph.degree())
                central_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:3]
                
                communities.append(Community(
                    id=f"community_{i}",
                    nodes=community_nodes,
                    density=density,
                    modularity=0.0,  # Would need full graph modularity calculation
                    central_nodes=central_nodes
                ))
            
            self.logger.info(f"Detected {len(communities)} communities using {algorithm}")
            return communities
        except Exception as e:
            self.logger.error(f"Error detecting communities with NetworkX: {e}")
            return self._detect_communities_basic(min_size)
    
    def _detect_communities_basic(self, min_size: int) -> List[Community]:
        """Detect communities using basic connected components"""
        communities = []
        visited = set()
        
        def dfs_component(start_node: str) -> Set[str]:
            """Find connected component using DFS"""
            component = set()
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                
                visited.add(node)
                component.add(node)
                
                # Add unvisited neighbors
                for neighbor in self.adjacency[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                
                # Also check incoming edges
                for source, targets in self.adjacency.items():
                    if node in targets and source not in visited:
                        stack.append(source)
            
            return component
        
        # Find all connected components
        for node_id in self.nodes:
            if node_id not in visited:
                component = dfs_component(node_id)
                if len(component) >= min_size:
                    # Calculate basic density
                    edges_in_component = 0
                    for source in component:
                        edges_in_component += len(self.adjacency[source].intersection(component))
                    
                    max_edges = len(component) * (len(component) - 1)
                    density = edges_in_component / max_edges if max_edges > 0 else 0.0
                    
                    # Find central nodes (highest degree)
                    node_degrees = {}
                    for node in component:
                        degree = len(self.adjacency[node].intersection(component))
                        # Add incoming edges
                        for source, targets in self.adjacency.items():
                            if node in targets and source in component:
                                degree += 1
                        node_degrees[node] = degree
                    
                    central_nodes = sorted(node_degrees.keys(), 
                                         key=lambda x: node_degrees[x], reverse=True)[:3]
                    
                    communities.append(Community(
                        id=f"component_{len(communities)}",
                        nodes=component,
                        density=density,
                        modularity=0.0,
                        central_nodes=central_nodes
                    ))
        
        self.logger.info(f"Detected {len(communities)} connected components")
        return communities
    
    def extract_subgraph(self, node_ids: Set[str], include_connections: bool = True) -> 'GraphAnalytics':
        """Extract a subgraph containing specified nodes"""
        subgraph = GraphAnalytics(use_networkx=self.use_networkx)
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges
        for (source_id, target_id), edge in self.edges.items():
            if include_connections:
                # Include if either node is in the set
                if source_id in node_ids or target_id in node_ids:
                    if source_id in self.nodes and target_id in self.nodes:
                        if source_id not in subgraph.nodes:
                            subgraph.add_node(self.nodes[source_id])
                        if target_id not in subgraph.nodes:
                            subgraph.add_node(self.nodes[target_id])
                        subgraph.add_edge(edge)
            else:
                # Include only if both nodes are in the set
                if source_id in node_ids and target_id in node_ids:
                    subgraph.add_edge(edge)
        
        return subgraph
    
    def get_node_neighborhood(self, node_id: str, radius: int = 1) -> Set[str]:
        """Get nodes within specified radius of given node"""
        if node_id not in self.nodes:
            return set()
        
        neighborhood = {node_id}
        current_level = {node_id}
        
        for _ in range(radius):
            next_level = set()
            for current_node in current_level:
                # Add outgoing neighbors
                next_level.update(self.adjacency[current_node])
                # Add incoming neighbors
                for source, targets in self.adjacency.items():
                    if current_node in targets:
                        next_level.add(source)
            
            # Remove already visited nodes
            next_level -= neighborhood
            neighborhood.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return neighborhood
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        # Ensure centrality measures are calculated
        centrality_measures = self.calculate_centrality_measures()
        communities = self.detect_communities()
        
        # Calculate graph metrics
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        
        # Node type distribution
        entity_types = defaultdict(int)
        for node in self.nodes.values():
            entity_types[node.entity_type] += 1
        
        # Relationship type distribution
        relationship_types = defaultdict(int)
        for edge in self.edges.values():
            relationship_types[edge.relationship_type] += 1
        
        # Top important nodes
        top_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.importance_score,
            reverse=True
        )[:10]
        
        return {
            'graph_metrics': {
                'total_nodes': n_nodes,
                'total_edges': n_edges,
                'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
                'average_degree': (2 * n_edges) / n_nodes if n_nodes > 0 else 0.0
            },
            'entity_distribution': dict(entity_types),
            'relationship_distribution': dict(relationship_types),
            'communities': {
                'total_communities': len(communities),
                'average_community_size': sum(len(c.nodes) for c in communities) / len(communities) if communities else 0,
                'largest_community_size': max(len(c.nodes) for c in communities) if communities else 0
            },
            'top_important_nodes': [
                {
                    'id': node.id,
                    'entity_type': node.entity_type,
                    'importance_score': node.importance_score,
                    'centrality_scores': node.centrality_scores
                }
                for node in top_nodes
            ],
            'analytics_capabilities': {
                'networkx_available': NETWORKX_AVAILABLE,
                'numpy_available': NUMPY_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE
            }
        }


def create_graph_analytics(use_networkx: bool = True) -> GraphAnalytics:
    """Factory function to create a graph analytics instance"""
    return GraphAnalytics(use_networkx=use_networkx)


if __name__ == "__main__":
    # Example usage
    analytics = create_graph_analytics()
    
    # Create test nodes
    node1 = GraphNode(id="person_1", entity_type="person", properties={"name": "Alice"})
    node2 = GraphNode(id="person_2", entity_type="person", properties={"name": "Bob"})
    node3 = GraphNode(id="org_1", entity_type="organization", properties={"name": "TechCorp"})
    
    # Add nodes
    analytics.add_node(node1)
    analytics.add_node(node2)
    analytics.add_node(node3)
    
    # Create test edges
    edge1 = GraphEdge("person_1", "org_1", "works_for", confidence=0.9, weight=1.0)
    edge2 = GraphEdge("person_2", "org_1", "works_for", confidence=0.8, weight=1.0)
    edge3 = GraphEdge("person_1", "person_2", "knows", confidence=0.7, weight=0.5)
    
    # Add edges
    analytics.add_edge(edge1)
    analytics.add_edge(edge2)
    analytics.add_edge(edge3)
    
    # Test analytics
    print("=== Graph Analytics Test ===")
    
    # Test shortest path
    path = analytics.find_shortest_path("person_1", "person_2")
    if path:
        print(f"Shortest path: {' -> '.join(path.path)}")
        print(f"Path length: {path.length}, Weight: {path.total_weight}")
    
    # Test centrality measures
    centralities = analytics.calculate_centrality_measures()
    print(f"\nCentrality measures calculated for {len(centralities)} nodes")
    
    # Test community detection
    communities = analytics.detect_communities()
    print(f"Detected {len(communities)} communities")
    
    # Get analytics summary
    summary = analytics.get_analytics_summary()
    print(f"\nGraph Summary:")
    print(f"- Nodes: {summary['graph_metrics']['total_nodes']}")
    print(f"- Edges: {summary['graph_metrics']['total_edges']}")
    print(f"- Density: {summary['graph_metrics']['density']:.3f}")
    print(f"- Communities: {summary['communities']['total_communities']}")
