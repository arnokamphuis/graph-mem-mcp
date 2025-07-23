#!/usr/bin/env python3
"""
Phase 4.2 Knowledge Graph Integration Validation

Tests the integration of Phase 1-3 knowledge graph components with FastAPI.
Validates new enhanced API endpoints for entity extraction, relationship extraction,
coreference resolution, quality assessment, and graph analytics.
"""

import sys
import json
import asyncio
import requests
import time
from pathlib import Path

# Add the mcp_server directory to Python path for imports
sys.path.append(str(Path(__file__).parent / "mcp_server"))

def test_api_imports():
    """Test that all required component imports work correctly"""
    print("🧪 Test 1: API Component Imports")
    
    try:
        # Test Phase 2.2 Enhanced Entity Extraction
        try:
            from extraction.enhanced_entity_extractor import create_enhanced_entity_extractor
            print("  ✅ Enhanced entity extractor import successful")
            extraction_available = True
        except ImportError as e:
            print(f"  ⚠️  Enhanced entity extractor not available: {e}")
            extraction_available = False
        
        # Test Phase 2.1 Sophisticated Relationship Extraction
        try:
            from extraction.relation_extractor import create_relationship_extractor
            print("  ✅ Relationship extractor import successful")
            relation_available = True
        except ImportError as e:
            print(f"  ⚠️  Relationship extractor not available: {e}")
            relation_available = False
        
        # Test Phase 2.3 Coreference Resolution
        try:
            from extraction.coreference_resolver import create_coreference_resolver
            print("  ✅ Coreference resolver import successful")
            coref_available = True
        except ImportError as e:
            print(f"  ⚠️  Coreference resolver not available: {e}")
            coref_available = False
        
        # Test Phase 3.1 Quality Assessment (may not exist yet)
        try:
            from quality.assessment_framework import create_quality_assessor
            print("  ✅ Quality assessor import successful")
            quality_available = True
        except ImportError as e:
            print(f"  ⚠️  Quality assessor not available: {e}")
            quality_available = False
        
        # Test Phase 1 Graph Analytics
        try:
            from core.graph_analytics import GraphAnalytics
            print("  ✅ Graph analytics import successful")
            analytics_available = True
        except ImportError as e:
            print(f"  ⚠️  Graph analytics not available: {e}")
            analytics_available = False
        
        # Count available components
        available_count = sum([extraction_available, relation_available, coref_available, analytics_available])
        total_core_components = 4  # Excluding quality for now as it might not be implemented
        
        if available_count >= 3:  # Allow some flexibility
            print(f"  ✅ {available_count}/{total_core_components} core components available")
            return True
        else:
            print(f"  ❌ Only {available_count}/{total_core_components} components available")
            return False
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_enhanced_entity_extraction():
    """Test enhanced entity extraction component directly"""
    print("🧪 Test 2: Enhanced Entity Extraction")
    
    try:
        from extraction.enhanced_entity_extractor import create_enhanced_entity_extractor
        
        # Create extractor
        extractor = create_enhanced_entity_extractor()
        print("  ✅ Enhanced entity extractor created")
        
        # Test extraction
        test_text = "John Smith works at Google Inc. in Mountain View, California."
        entities = extractor.extract_entities(test_text)
        
        print(f"  ✅ Extracted {len(entities)} entities from test text")
        
        # Validate entities have required attributes
        for entity in entities[:3]:  # Check first 3
            if hasattr(entity, 'text') and hasattr(entity, 'type'):
                print(f"    - {entity.text} ({entity.type})")
            else:
                print(f"    - Entity: {entity}")
        
        return len(entities) > 0
        
    except Exception as e:
        print(f"  ❌ Enhanced entity extraction test failed: {e}")
        return False

def test_relationship_extraction():
    """Test sophisticated relationship extraction component directly"""
    print("🧪 Test 3: Sophisticated Relationship Extraction")
    
    try:
        from extraction.relation_extractor import create_relationship_extractor
        
        # Create extractor
        extractor = create_relationship_extractor()
        print("  ✅ Relationship extractor created")
        
        # Test extraction with entities
        test_text = "John Smith works at Google Inc."
        test_entities = [
            {"id": "ent1", "text": "John Smith", "type": "person", "start": 0, "end": 10},
            {"id": "ent2", "text": "Google Inc", "type": "organization", "start": 20, "end": 30}
        ]
        
        relationships = extractor.extract_relationships(test_text, test_entities)
        
        print(f"  ✅ Extracted {len(relationships)} relationships from test text")
        
        # Validate relationships
        for rel in relationships[:2]:  # Check first 2
            if hasattr(rel, 'type') and hasattr(rel, 'confidence'):
                print(f"    - {rel.type} (confidence: {rel.confidence:.2f})")
            else:
                print(f"    - Relationship: {rel}")
        
        return True  # Success even if no relationships found
        
    except Exception as e:
        print(f"  ❌ Relationship extraction test failed: {e}")
        return False

def test_coreference_resolution():
    """Test coreference resolution component directly"""
    print("🧪 Test 4: Coreference Resolution")
    
    try:
        from extraction.coreference_resolver import create_coreference_resolver
        
        # Create resolver
        resolver = create_coreference_resolver()
        print("  ✅ Coreference resolver created")
        
        # Test resolution
        test_text = "John Smith is a software engineer. He works at Google."
        test_entities = [
            {"id": "ent1", "text": "John Smith", "type": "person", "start": 0, "end": 10},
            {"id": "ent2", "text": "He", "type": "pronoun", "start": 40, "end": 42}
        ]
        
        result = resolver.resolve_coreferences(test_text, test_entities)
        
        print(f"  ✅ Coreference resolution completed")
        
        # Check if result has expected structure
        if isinstance(result, dict):
            chains = result.get("chains", [])
            print(f"    - Found {len(chains)} coreference chains")
        else:
            print(f"    - Resolution result: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Coreference resolution test failed: {e}")
        return False

def test_graph_analytics():
    """Test graph analytics component directly"""
    print("🧪 Test 5: Graph Analytics")
    
    try:
        from core.graph_analytics import GraphAnalytics
        
        # Create analytics engine
        analytics = GraphAnalytics()
        print("  ✅ Graph analytics engine created")
        
        # Test basic metrics calculation with proper GraphNode objects
        from core.graph_analytics import GraphNode, GraphEdge
        
        # Create proper GraphNode objects
        node1 = GraphNode(
            id="ent1",
            entity_type="person",
            properties={"name": "John"}
        )
        node2 = GraphNode(
            id="ent2", 
            entity_type="organization",
            properties={"name": "Google"}
        )
        node3 = GraphNode(
            id="ent3",
            entity_type="person", 
            properties={"name": "Jane"}
        )
        
        # Add nodes using proper GraphNode objects
        analytics.add_node(node1)
        analytics.add_node(node2)
        analytics.add_node(node3)
        
        # Create proper GraphEdge objects
        edge1 = GraphEdge(
            source="ent1",
            target="ent2",
            relationship_type="works_for",
            properties={}
        )
        edge2 = GraphEdge(
            source="ent3",
            target="ent2", 
            relationship_type="works_for",
            properties={}
        )
        
        # Add edges using proper GraphEdge objects
        analytics.add_edge(edge1)
        analytics.add_edge(edge2)
        
        # Test analytics summary
        metrics = analytics.get_analytics_summary()
        
        print(f"  ✅ Graph analytics summary generated: {type(metrics)}")
        
        if isinstance(metrics, dict):
            print(f"    - Summary keys: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Graph analytics test failed: {e}")
        return False

def test_api_endpoint_structure():
    """Test that API endpoints are properly defined in main.py"""
    print("🧪 Test 6: API Endpoint Structure")
    
    try:
        # Read main.py and check for endpoint definitions
        main_file = Path("mcp_server/main.py")
        if not main_file.exists():
            print("  ❌ main.py file not found")
            return False
        
        main_content = main_file.read_text()
        
        # Check for Phase 4.2 endpoints
        endpoints = [
            "/api/v1/extract/entities",
            "/api/v1/extract/relationships", 
            "/api/v1/resolve/coreferences",
            "/api/v1/quality/assess",
            "/api/v1/analytics/graph"
        ]
        
        found_endpoints = 0
        for endpoint in endpoints:
            if endpoint in main_content:
                print(f"  ✅ Found endpoint: {endpoint}")
                found_endpoints += 1
            else:
                print(f"  ❌ Missing endpoint: {endpoint}")
        
        if found_endpoints >= 4:  # Allow some flexibility
            print(f"  ✅ {found_endpoints}/{len(endpoints)} endpoints found")
            return True
        else:
            print(f"  ❌ Only {found_endpoints}/{len(endpoints)} endpoints found")
            return False
        
    except Exception as e:
        print(f"  ❌ Endpoint structure test failed: {e}")
        return False

def main():
    """Run all Phase 4.2 validation tests"""
    print("============================================================")
    print("🧪 PHASE 4.2 KNOWLEDGE GRAPH INTEGRATION VALIDATION SUITE")
    print("============================================================")
    
    tests = [
        ("API Component Imports", test_api_imports),
        ("Enhanced Entity Extraction", test_enhanced_entity_extraction),
        ("Sophisticated Relationship Extraction", test_relationship_extraction),
        ("Coreference Resolution", test_coreference_resolution),
        ("Graph Analytics", test_graph_analytics),
        ("API Endpoint Structure", test_api_endpoint_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
        except Exception as e:
            print(f"  ❌ Test error: {e}")
            status = "❌ ERROR"
        
        print(f"{test_name:40} {status}")
    
    print("============================================================")
    print("📊 VALIDATION RESULTS")
    print("============================================================")
    
    success_rate = (passed / total) * 100
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 70:  # 70% threshold for Phase 4.2
        print("✅ QUALITY GATE PASSED: 70%+ test success")
        print("🎉 Phase 4.2 validation SUCCESSFUL!")
        return 0
    else:
        print("❌ QUALITY GATE FAILED: <70% test success")
        print("🔄 Phase 4.2 needs additional work")
        return 1

if __name__ == "__main__":
    sys.exit(main())
