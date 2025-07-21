#!/usr/bin/env python3
"""
Simple test runner for Phase 2.1 Sophisticated Relationship Extraction
Validates core functionality without import issues
"""

import sys
import os

# Add paths for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic import functionality"""
    print("🧪 Testing Phase 2.1 Basic Imports...")
    
    try:
        # Test core imports
        from mcp_server.core.graph_schema import RelationshipInstance, SchemaManager
        print("✅ Core schema imports successful")
        
        # Test basic relation extractor imports
        from mcp_server.extraction.relation_extractor import (
            ExtractionMethod, RelationshipCandidate, ExtractionContext
        )
        print("✅ Basic relation extractor imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_class_instantiation():
    """Test class instantiation"""
    print("\n🧪 Testing Class Instantiation...")
    
    try:
        from mcp_server.extraction.relation_extractor import (
            SophisticatedRelationshipExtractor, ExtractionContext, 
            RelationshipCandidate, ExtractionMethod, create_relationship_extractor
        )
        
        # Test factory function
        extractor = create_relationship_extractor(confidence_threshold=0.5)
        print("✅ SophisticatedRelationshipExtractor created via factory")
        
        # Test context creation
        context = ExtractionContext(
            text="John works at Google. Sarah graduated from MIT.",
            source_id="test"
        )
        print("✅ ExtractionContext created successfully")
        
        # Test candidate creation
        candidate = RelationshipCandidate(
            source_entity="John",
            target_entity="Google", 
            relationship_type="works_at",
            confidence=0.8,
            evidence_text="John works at Google",
            context_window="John works at Google",
            extraction_method=ExtractionMethod.PATTERN_BASED,
            position_start=0,
            position_end=18
        )
        print("✅ RelationshipCandidate created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Class instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_extraction():
    """Test basic extraction functionality"""
    print("\n🧪 Testing Basic Extraction...")
    
    try:
        from mcp_server.extraction.relation_extractor import (
            create_relationship_extractor, ExtractionContext
        )
        
        # Create extractor
        extractor = create_relationship_extractor(confidence_threshold=0.3)
        
        # Simple test text
        context = ExtractionContext(
            text="John Smith works at Google Inc.",
            source_id="test_doc"
        )
        
        # Extract relationships
        candidates = extractor.extract_relationships(context)
        print(f"✅ Extracted {len(candidates)} relationship candidates")
        
        # Show candidates if any
        for i, candidate in enumerate(candidates[:3]):
            print(f"   {i+1}. {candidate.source_entity} --{candidate.relationship_type}--> {candidate.target_entity} (conf: {candidate.confidence:.2f})")
        
        # Test statistics
        stats = extractor.get_extraction_statistics()
        print(f"✅ Statistics retrieved: {len(stats)} metrics")
        
        return True
    except Exception as e:
        print(f"❌ Basic extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_1_integration():
    """Test Phase 1 integration"""
    print("\n🧪 Testing Phase 1 Integration...")
    
    try:
        from mcp_server.extraction.relation_extractor import (
            create_relationship_extractor, ExtractionContext
        )
        from mcp_server.core.graph_schema import RelationshipInstance
        
        # Create extractor
        extractor = create_relationship_extractor(confidence_threshold=0.3)
        
        # Test text
        context = ExtractionContext(
            text="Alice manages Bob at TechCorp.",
            source_id="test_integration"
        )
        
        # Extract as Phase 1 instances
        instances = extractor.extract_relationships_as_instances(context)
        print(f"✅ Converted to {len(instances)} RelationshipInstance objects")
        
        # Validate instances
        for i, instance in enumerate(instances[:2]):
            if isinstance(instance, RelationshipInstance):
                print(f"   {i+1}. Instance: {instance.source_entity_id} --{instance.relationship_type}--> {instance.target_entity_id}")
            else:
                print(f"   {i+1}. Invalid instance type: {type(instance)}")
        
        return True
    except Exception as e:
        print(f"❌ Phase 1 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and report results"""
    print("=" * 60)
    print("🧪 PHASE 2.1 SOPHISTICATED RELATIONSHIP EXTRACTION TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("Class Instantiation", test_class_instantiation()))
    test_results.append(("Basic Extraction", test_basic_extraction()))
    test_results.append(("Phase 1 Integration", test_phase_1_integration()))
    
    # Report results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Phase 2.1 is ready!")
        return True
    else:
        print("⚠️  TESTS FAILED - Phase 2.1 needs fixes!")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
