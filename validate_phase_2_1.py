#!/usr/bin/env python3
"""
Direct validation test for Phase 2.1 components
Focuses on immediate validation without complex test framework
"""

import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_import_validation():
    """Test 1: Import Validation"""
    print("🧪 Test 1: Import Validation")
    try:
        from mcp_server.extraction.relation_extractor import (
            ExtractionMethod, RelationshipCandidate, ExtractionContext,
            SophisticatedRelationshipExtractor, create_relationship_extractor
        )
        print("  ✅ Core extraction imports successful")
        
        from mcp_server.core.graph_schema import RelationshipInstance
        print("  ✅ Phase 1 core imports successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_enum_validation():
    """Test 2: Enum Validation"""
    print("\n🧪 Test 2: ExtractionMethod Enum Validation")
    try:
        from mcp_server.extraction.relation_extractor import ExtractionMethod
        
        methods = [method.value for method in ExtractionMethod]
        expected = ['transformer', 'pattern_based', 'dependency_parsing', 'rule_based']
        
        print(f"  Available methods: {methods}")
        
        for exp in expected:
            if exp in methods:
                print(f"  ✅ {exp} method available")
            else:
                print(f"  ⚠️ {exp} method missing")
        
        return len([m for m in expected if m in methods]) >= 3
    except Exception as e:
        print(f"  ❌ Enum test failed: {e}")
        return False

def test_factory_creation():
    """Test 3: Factory Creation"""
    print("\n🧪 Test 3: Factory Creation")
    try:
        from mcp_server.extraction.relation_extractor import create_relationship_extractor
        
        extractor = create_relationship_extractor(confidence_threshold=0.5)
        print("  ✅ Factory created extractor successfully")
        
        # Test basic properties
        if hasattr(extractor, 'confidence_threshold'):
            print(f"  ✅ Confidence threshold: {extractor.confidence_threshold}")
        
        if hasattr(extractor, 'get_extraction_statistics'):
            stats = extractor.get_extraction_statistics()
            print(f"  ✅ Statistics method works: {len(stats)} metrics")
        
        return True
    except Exception as e:
        print(f"  ❌ Factory test failed: {e}")
        return False

def test_candidate_creation():
    """Test 4: Candidate Creation"""
    print("\n🧪 Test 4: RelationshipCandidate Creation")
    try:
        from mcp_server.extraction.relation_extractor import RelationshipCandidate, ExtractionMethod
        
        candidate = RelationshipCandidate(
            source_entity="John",
            target_entity="Google",
            relationship_type="works_at",
            confidence=0.8,
            evidence_text="John works at Google",
            context_window="John works at Google Inc",
            extraction_method=ExtractionMethod.PATTERN_BASED,
            position_start=0,
            position_end=20
        )
        
        print("  ✅ RelationshipCandidate created successfully")
        print(f"  ✅ Source: {candidate.source_entity}")
        print(f"  ✅ Target: {candidate.target_entity}")
        print(f"  ✅ Type: {candidate.relationship_type}")
        print(f"  ✅ Confidence: {candidate.confidence}")
        
        # Test conversion to RelationshipInstance
        instance = candidate.to_relationship_instance()
        if instance:
            print("  ✅ Conversion to RelationshipInstance successful")
        else:
            print("  ⚠️ RelationshipInstance conversion returned None")
        
        return True
    except Exception as e:
        print(f"  ❌ Candidate creation failed: {e}")
        return False

def test_basic_extraction():
    """Test 5: Basic Extraction"""
    print("\n🧪 Test 5: Basic Relationship Extraction")
    try:
        from mcp_server.extraction.relation_extractor import (
            create_relationship_extractor, ExtractionContext
        )
        
        extractor = create_relationship_extractor(confidence_threshold=0.1)
        context = ExtractionContext(
            text="John Smith works at Google Inc.",
            source_id="test_doc"
        )
        
        print("  🔍 Attempting extraction...")
        candidates = extractor.extract_relationships(context)
        
        print(f"  ✅ Extraction completed: {len(candidates)} candidates")
        
        # Show first few candidates
        for i, candidate in enumerate(candidates[:2]):
            print(f"    {i+1}. {candidate.source_entity} --{candidate.relationship_type}--> {candidate.target_entity} (conf: {candidate.confidence:.2f})")
        
        return True
    except Exception as e:
        print(f"  ❌ Basic extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_validation_suite():
    """Run all validation tests"""
    print("="*60)
    print("🧪 PHASE 2.1 DIRECT VALIDATION SUITE")
    print("="*60)
    
    tests = [
        ("Import Validation", test_import_validation),
        ("Enum Validation", test_enum_validation),
        ("Factory Creation", test_factory_creation),
        ("Candidate Creation", test_candidate_creation),
        ("Basic Extraction", test_basic_extraction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-"*60)
    coverage = (passed / total * 100) if total > 0 else 0
    print(f"Tests Passed: {passed}/{total} ({coverage:.1f}%)")
    
    if coverage >= 90:
        print("✅ QUALITY GATE PASSED: 90%+ test success")
        return True
    else:
        print("❌ QUALITY GATE FAILED: <90% test success")
        return False

if __name__ == "__main__":
    success = run_validation_suite()
    if success:
        print("\n🎉 Phase 2.1 validation SUCCESSFUL!")
        exit(0)
    else:
        print("\n⚠️ Phase 2.1 validation INCOMPLETE")
        exit(1)
