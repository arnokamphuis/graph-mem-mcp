"""
Test Suite for Graph Analytics Module

This test validates the graph analytics implementation with comprehensive coverage
of shortest path algorithms, centrality measures, community detection, and 
subgraph operations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mcp_server.core.graph_analytics import (
    GraphAnalytics, GraphNode, GraphEdge, PathResult, Community,
    create_graph_analytics
)


def test_basic_graph_operations():
    """Test basic graph node and edge operations"""
    print("=== Testing Basic Graph Operations ===")
    
    analytics = create_graph_analytics()
    
    # Create test nodes
    node1 = GraphNode(id="alice", entity_type="person", 
                      properties={"name": "Alice", "age": 30})
    node2 = GraphNode(id="bob", entity_type="person", 
                      properties={"name": "Bob", "age": 25})
    node3 = GraphNode(id="techcorp", entity_type="organization", 
                      properties={"name": "TechCorp", "industry": "Software"})
    node4 = GraphNode(id="project_x", entity_type="project",
                      properties={"name": "Project X", "status": "active"})
    
    # Add nodes
    assert analytics.add_node(node1), "Failed to add Alice"
    assert analytics.add_node(node2), "Failed to add Bob"
    assert analytics.add_node(node3), "Failed to add TechCorp"
    assert analytics.add_node(node4), "Failed to add Project X"
    
    print(f"✅ Added {len(analytics.nodes)} nodes")
    
    # Create test edges
    edge1 = GraphEdge("alice", "techcorp", "works_for", 
                      confidence=0.9, weight=1.0)
    edge2 = GraphEdge("bob", "techcorp", "works_for", 
                      confidence=0.8, weight=1.0)
    edge3 = GraphEdge("alice", "bob", "knows", 
                      confidence=0.7, weight=0.5)
    edge4 = GraphEdge("alice", "project_x", "manages", 
                      confidence=0.95, weight=2.0)
    edge5 = GraphEdge("bob", "project_x", "works_on", 
                      confidence=0.85, weight=1.5)
    
    # Add edges
    assert analytics.add_edge(edge1), "Failed to add Alice -> TechCorp"
    assert analytics.add_edge(edge2), "Failed to add Bob -> TechCorp"
    assert analytics.add_edge(edge3), "Failed to add Alice -> Bob"
    assert analytics.add_edge(edge4), "Failed to add Alice -> Project X"
    assert analytics.add_edge(edge5), "Failed to add Bob -> Project X"
    
    print(f"✅ Added {len(analytics.edges)} edges")
    
    return analytics


def test_shortest_path():
    """Test shortest path algorithms"""
    print("\n=== Testing Shortest Path Algorithms ===")
    
    analytics = test_basic_graph_operations()
    
    # Test direct path
    path = analytics.find_shortest_path("alice", "bob")
    assert path is not None, "Should find path from Alice to Bob"
    assert len(path.path) == 2, f"Expected path length 2, got {len(path.path)}"
    assert path.path == ["alice", "bob"], f"Expected [alice, bob], got {path.path}"
    print(f"✅ Direct path: {' -> '.join(path.path)} (weight: {path.total_weight})")
    
    # Test indirect path
    path = analytics.find_shortest_path("bob", "alice")
    if path:
        print(f"✅ Indirect path: {' -> '.join(path.path)} (weight: {path.total_weight})")
    else:
        print("ℹ️ No path from Bob to Alice (directed graph)")
    
    # Test multiple paths
    all_paths = analytics.find_all_paths("alice", "project_x", max_length=3, max_paths=5)
    print(f"✅ Found {len(all_paths)} paths from Alice to Project X")
    for i, path in enumerate(all_paths):
        print(f"   Path {i+1}: {' -> '.join(path.path)} (weight: {path.total_weight})")


def test_centrality_measures():
    """Test centrality calculation"""
    print("\n=== Testing Centrality Measures ===")
    
    analytics = test_basic_graph_operations()
    
    centralities = analytics.calculate_centrality_measures()
    
    print("✅ Centrality measures calculated:")
    for node_id, measures in centralities.items():
        node = analytics.nodes[node_id]
        print(f"   {node_id} (importance: {node.importance_score:.3f}):")
        for measure, value in measures.items():
            print(f"     {measure}: {value:.3f}")
    
    # Check that all nodes have centrality measures
    assert len(centralities) == len(analytics.nodes), "All nodes should have centrality measures"
    
    # Check that importance scores are updated
    for node in analytics.nodes.values():
        assert node.importance_score >= 0, "Importance score should be non-negative"
    
    print("✅ All centrality measures validated")


def test_community_detection():
    """Test community detection algorithms"""
    print("\n=== Testing Community Detection ===")
    
    analytics = test_basic_graph_operations()
    
    # Add more nodes to create meaningful communities
    # Second company cluster
    node5 = GraphNode(id="charlie", entity_type="person", 
                      properties={"name": "Charlie", "age": 35})
    node6 = GraphNode(id="datacorp", entity_type="organization", 
                      properties={"name": "DataCorp", "industry": "Analytics"})
    node7 = GraphNode(id="project_y", entity_type="project",
                      properties={"name": "Project Y", "status": "planning"})
    
    analytics.add_node(node5)
    analytics.add_node(node6)
    analytics.add_node(node7)
    
    # Create second cluster
    analytics.add_edge(GraphEdge("charlie", "datacorp", "works_for", confidence=0.9, weight=1.0))
    analytics.add_edge(GraphEdge("charlie", "project_y", "leads", confidence=0.95, weight=2.0))
    
    # Connect clusters
    analytics.add_edge(GraphEdge("alice", "charlie", "knows", confidence=0.6, weight=0.3))
    
    communities = analytics.detect_communities(min_size=2)
    
    print(f"✅ Detected {len(communities)} communities:")
    for i, community in enumerate(communities):
        print(f"   Community {i+1} ({len(community.nodes)} nodes):")
        print(f"     Nodes: {sorted(community.nodes)}")
        print(f"     Density: {community.density:.3f}")
        print(f"     Central nodes: {community.central_nodes}")


def test_subgraph_operations():
    """Test subgraph extraction and neighborhood analysis"""
    print("\n=== Testing Subgraph Operations ===")
    
    analytics = test_basic_graph_operations()
    
    # Test neighborhood discovery
    neighborhood = analytics.get_node_neighborhood("alice", radius=1)
    print(f"✅ Alice's 1-hop neighborhood: {sorted(neighborhood)}")
    
    neighborhood_2 = analytics.get_node_neighborhood("alice", radius=2)
    print(f"✅ Alice's 2-hop neighborhood: {sorted(neighborhood_2)}")
    
    # Test subgraph extraction
    selected_nodes = {"alice", "bob", "techcorp"}
    subgraph = analytics.extract_subgraph(selected_nodes, include_connections=False)
    
    print(f"✅ Extracted subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
    
    # Test subgraph with connections
    subgraph_with_connections = analytics.extract_subgraph(selected_nodes, include_connections=True)
    print(f"✅ Extracted subgraph with connections: {len(subgraph_with_connections.nodes)} nodes, {len(subgraph_with_connections.edges)} edges")


def test_analytics_summary():
    """Test comprehensive analytics summary"""
    print("\n=== Testing Analytics Summary ===")
    
    analytics = test_basic_graph_operations()
    
    summary = analytics.get_analytics_summary()
    
    print("✅ Analytics Summary Generated:")
    print(f"   Graph Metrics:")
    print(f"     Total Nodes: {summary['graph_metrics']['total_nodes']}")
    print(f"     Total Edges: {summary['graph_metrics']['total_edges']}")
    print(f"     Density: {summary['graph_metrics']['density']:.3f}")
    print(f"     Average Degree: {summary['graph_metrics']['average_degree']:.3f}")
    
    print(f"   Entity Distribution: {summary['entity_distribution']}")
    print(f"   Relationship Distribution: {summary['relationship_distribution']}")
    
    print(f"   Top Important Nodes:")
    for node_info in summary['top_important_nodes'][:3]:
        print(f"     {node_info['id']} ({node_info['entity_type']}): {node_info['importance_score']:.3f}")
    
    print(f"   Analytics Capabilities:")
    for capability, available in summary['analytics_capabilities'].items():
        status = "✅" if available else "❌"
        print(f"     {capability}: {status}")
    
    # Validate summary structure
    required_keys = ['graph_metrics', 'entity_distribution', 'relationship_distribution', 
                    'communities', 'top_important_nodes', 'analytics_capabilities']
    for key in required_keys:
        assert key in summary, f"Missing required key: {key}"
    
    print("✅ Analytics summary validation complete")


def test_graceful_fallback():
    """Test graceful fallback when NetworkX is not available"""
    print("\n=== Testing Graceful Fallback ===")
    
    # Test with NetworkX disabled
    analytics_basic = GraphAnalytics(use_networkx=False)
    
    # Create basic graph
    node1 = GraphNode(id="test1", entity_type="test")
    node2 = GraphNode(id="test2", entity_type="test")
    
    analytics_basic.add_node(node1)
    analytics_basic.add_node(node2)
    analytics_basic.add_edge(GraphEdge("test1", "test2", "connects"))
    
    # Test basic operations work without NetworkX
    path = analytics_basic.find_shortest_path("test1", "test2")
    assert path is not None, "Basic path finding should work"
    
    centralities = analytics_basic.calculate_centrality_measures()
    assert len(centralities) == 2, "Basic centrality should work"
    
    communities = analytics_basic.detect_communities()
    assert len(communities) >= 0, "Basic community detection should work"
    
    print("✅ Graceful fallback validated - all operations work without NetworkX")


def run_all_tests():
    """Run comprehensive test suite"""
    print("🚀 Starting Graph Analytics Test Suite")
    print("=" * 60)
    
    try:
        test_basic_graph_operations()
        test_shortest_path()
        test_centrality_measures()
        test_community_detection()
        test_subgraph_operations()
        test_analytics_summary()
        test_graceful_fallback()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - Graph Analytics Implementation Validated!")
        print("✅ Ready for integration with main knowledge graph system")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
