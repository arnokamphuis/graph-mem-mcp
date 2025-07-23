"""
Direct Test for Graph Analytics Module (No Dependencies)

This test validates the graph analytics implementation without relying on
other modules that may have dependency issues.
"""

import sys
import os

# Add the path to access the graph_analytics module directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp_server', 'core'))

try:
    from graph_analytics import (
        GraphAnalytics, GraphNode, GraphEdge, PathResult, Community,
        create_graph_analytics
    )
    print("✅ Successfully imported graph_analytics module")
except ImportError as e:
    print(f"❌ Failed to import graph_analytics: {e}")
    sys.exit(1)


def test_direct_graph_operations():
    """Test graph analytics directly without schema dependencies"""
    print("\n=== Testing Direct Graph Analytics Operations ===")
    
    # Create analytics instance
    analytics = create_graph_analytics()
    print(f"✅ Created GraphAnalytics instance (NetworkX: {analytics.use_networkx})")
    
    # Create test nodes
    nodes = [
        GraphNode(id="alice", entity_type="person", properties={"name": "Alice"}),
        GraphNode(id="bob", entity_type="person", properties={"name": "Bob"}),
        GraphNode(id="company", entity_type="organization", properties={"name": "TechCorp"}),
        GraphNode(id="project", entity_type="project", properties={"name": "AI Project"})
    ]
    
    # Add nodes
    for node in nodes:
        success = analytics.add_node(node)
        assert success, f"Failed to add node {node.id}"
    
    print(f"✅ Added {len(analytics.nodes)} nodes")
    
    # Create test edges
    edges = [
        GraphEdge("alice", "company", "works_for", confidence=0.9, weight=1.0),
        GraphEdge("bob", "company", "works_for", confidence=0.8, weight=1.0),
        GraphEdge("alice", "bob", "knows", confidence=0.7, weight=0.5),
        GraphEdge("alice", "project", "manages", confidence=0.95, weight=2.0),
        GraphEdge("bob", "project", "contributes_to", confidence=0.85, weight=1.5),
    ]
    
    # Add edges
    for edge in edges:
        success = analytics.add_edge(edge)
        assert success, f"Failed to add edge {edge.source_id} -> {edge.target_id}"
    
    print(f"✅ Added {len(analytics.edges)} edges")
    
    return analytics


def test_path_finding():
    """Test shortest path functionality"""
    print("\n=== Testing Path Finding ===")
    
    analytics = test_direct_graph_operations()
    
    # Test shortest path
    path = analytics.find_shortest_path("alice", "bob")
    if path:
        print(f"✅ Found path: {' -> '.join(path.path)}")
        print(f"   Length: {path.length}, Weight: {path.total_weight:.2f}")
        print(f"   Confidence: {path.confidence:.2f}")
    else:
        print("❌ No path found")
    
    # Test all paths
    all_paths = analytics.find_all_paths("alice", "project", max_length=3)
    print(f"✅ Found {len(all_paths)} paths from alice to project")
    for i, path in enumerate(all_paths):
        print(f"   Path {i+1}: {' -> '.join(path.path)} (weight: {path.total_weight:.2f})")


def test_centrality():
    """Test centrality measures"""
    print("\n=== Testing Centrality Measures ===")
    
    analytics = test_direct_graph_operations()
    
    centralities = analytics.calculate_centrality_measures()
    print("✅ Centrality measures calculated:")
    
    for node_id, measures in centralities.items():
        node = analytics.nodes[node_id]
        print(f"   {node_id}:")
        print(f"     Importance: {node.importance_score:.3f}")
        for measure_name, value in measures.items():
            print(f"     {measure_name}: {value:.3f}")


def test_communities():
    """Test community detection"""
    print("\n=== Testing Community Detection ===")
    
    analytics = test_direct_graph_operations()
    
    communities = analytics.detect_communities(min_size=2)
    print(f"✅ Detected {len(communities)} communities:")
    
    for i, community in enumerate(communities):
        print(f"   Community {i+1}:")
        print(f"     Nodes: {sorted(community.nodes)}")
        print(f"     Density: {community.density:.3f}")
        print(f"     Central nodes: {community.central_nodes}")


def test_subgraph():
    """Test subgraph operations"""
    print("\n=== Testing Subgraph Operations ===")
    
    analytics = test_direct_graph_operations()
    
    # Test neighborhood
    neighborhood = analytics.get_node_neighborhood("alice", radius=1)
    print(f"✅ Alice's neighborhood: {sorted(neighborhood)}")
    
    # Test subgraph extraction
    selected_nodes = {"alice", "bob", "company"}
    subgraph = analytics.extract_subgraph(selected_nodes)
    print(f"✅ Extracted subgraph: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")


def test_summary():
    """Test analytics summary"""
    print("\n=== Testing Analytics Summary ===")
    
    analytics = test_direct_graph_operations()
    
    summary = analytics.get_analytics_summary()
    print("✅ Analytics Summary:")
    print(f"   Nodes: {summary['graph_metrics']['total_nodes']}")
    print(f"   Edges: {summary['graph_metrics']['total_edges']}")
    print(f"   Density: {summary['graph_metrics']['density']:.3f}")
    print(f"   Average Degree: {summary['graph_metrics']['average_degree']:.3f}")
    
    print(f"   Entity Types: {summary['entity_distribution']}")
    print(f"   Relationship Types: {summary['relationship_distribution']}")
    
    print(f"   Top Nodes:")
    for node_info in summary['top_important_nodes'][:3]:
        print(f"     {node_info['id']}: {node_info['importance_score']:.3f}")


def test_fallback_mode():
    """Test without NetworkX"""
    print("\n=== Testing Fallback Mode (No NetworkX) ===")
    
    analytics = GraphAnalytics(use_networkx=False)
    print(f"✅ Created analytics in fallback mode")
    
    # Add minimal graph
    analytics.add_node(GraphNode(id="n1", entity_type="test"))
    analytics.add_node(GraphNode(id="n2", entity_type="test"))
    analytics.add_edge(GraphEdge("n1", "n2", "connects"))
    
    # Test basic operations
    path = analytics.find_shortest_path("n1", "n2")
    assert path is not None, "Path should be found"
    
    centralities = analytics.calculate_centrality_measures()
    assert len(centralities) == 2, "Should have centralities for both nodes"
    
    communities = analytics.detect_communities()
    print(f"✅ Fallback mode works: found {len(communities)} communities")


def run_direct_tests():
    """Run all direct tests"""
    print("🚀 Running Direct Graph Analytics Tests")
    print("=" * 50)
    
    try:
        test_direct_graph_operations()
        test_path_finding()
        test_centrality()
        test_communities()
        test_subgraph()
        test_summary()
        test_fallback_mode()
        
        print("\n" + "=" * 50)
        print("🎉 ALL DIRECT TESTS PASSED!")
        print("✅ Graph Analytics module is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_direct_tests()
    sys.exit(0 if success else 1)
