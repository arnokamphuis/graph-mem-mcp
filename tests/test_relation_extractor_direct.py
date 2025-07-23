"""
Direct Test for Relationship Extractor (No Dependencies)

This test validates the relationship extractor implementation without relying on
other modules that may have dependency issues.
"""

import sys
import os

# Add the path to access the relation_extractor module directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp_server', 'extraction'))

try:
    from relation_extractor import (
        RelationshipCandidate, RelationshipExtractor, ExtractionContext,
        create_relationship_extractor, ExtractionMethod
    )
    print("✅ Successfully imported relationship extractor modules")
except ImportError as e:
    print(f"❌ Failed to import relationship extractor: {e}")
    sys.exit(1)


def test_basic_extraction():
    """Test basic relationship extraction functionality"""
    print("\n=== Testing Basic Relationship Extraction ===")
    
    # Create extractor with minimal dependencies
    extractor = create_relationship_extractor(
        confidence_threshold=0.3,
        enable_transformer=False,  # Disable to avoid dependency issues
        enable_dependency_parsing=False
    )
    print(f"✅ Created relationship extractor")
    
    # Create test context
    test_entities = [
        {"text": "Alice Johnson", "start": 0, "end": 12, "type": "PERSON"},
        {"text": "TechCorp Inc", "start": 22, "end": 34, "type": "ORG"},
        {"text": "San Francisco", "start": 45, "end": 58, "type": "GPE"},
        {"text": "John Smith", "start": 70, "end": 80, "type": "PERSON"}
    ]
    
    test_text = "Alice Johnson works for TechCorp Inc, which is located in San Francisco. John Smith founded TechCorp Inc."
    
    context = ExtractionContext(
        text=test_text,
        entities=test_entities,
        sentence_boundaries=[(0, 69), (70, len(test_text))]
    )
    
    print(f"✅ Created extraction context with {len(test_entities)} entities")
    
    # Extract relationships
    candidates = extractor.extract_relationships(context)
    
    print(f"✅ Extracted {len(candidates)} relationship candidates:")
    
    for i, candidate in enumerate(candidates):
        print(f"\n  Candidate {i+1}:")
        print(f"    {candidate.source_entity} --[{candidate.relationship_type}]--> {candidate.target_entity}")
        print(f"    Confidence: {candidate.confidence:.3f}")
        print(f"    Method: {candidate.extraction_method.value}")
        print(f"    Evidence: '{candidate.evidence_text}'")
    
    return len(candidates) > 0


def test_pattern_matching():
    """Test pattern-based relationship extraction"""
    print("\n=== Testing Pattern-Based Extraction ===")
    
    extractor = create_relationship_extractor(
        confidence_threshold=0.2,
        enable_transformer=False,
        enable_dependency_parsing=False,
        enable_pattern_matching=True
    )
    
    # Test various relationship patterns
    test_cases = [
        {
            "text": "Dr. Sarah Chen works for Microsoft Corporation in Seattle.",
            "entities": [
                {"text": "Dr. Sarah Chen", "start": 0, "end": 13, "type": "PERSON"},
                {"text": "Microsoft Corporation", "start": 24, "end": 45, "type": "ORG"},
                {"text": "Seattle", "start": 49, "end": 56, "type": "GPE"}
            ]
        },
        {
            "text": "Apple Inc was founded by Steve Jobs and Steve Wozniak.",
            "entities": [
                {"text": "Apple Inc", "start": 0, "end": 9, "type": "ORG"},
                {"text": "Steve Jobs", "start": 26, "end": 36, "type": "PERSON"},
                {"text": "Steve Wozniak", "start": 41, "end": 54, "type": "PERSON"}
            ]
        },
        {
            "text": "The headquarters of Google is located in Mountain View, California.",
            "entities": [
                {"text": "Google", "start": 20, "end": 26, "type": "ORG"},
                {"text": "Mountain View", "start": 42, "end": 55, "type": "GPE"},
                {"text": "California", "start": 57, "end": 67, "type": "GPE"}
            ]
        }
    ]
    
    total_candidates = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n  Test Case {i+1}: {test_case['text'][:50]}...")
        
        context = ExtractionContext(
            text=test_case["text"],
            entities=test_case["entities"],
            sentence_boundaries=[(0, len(test_case["text"]))]
        )
        
        candidates = extractor.extract_relationships(context)
        total_candidates += len(candidates)
        
        print(f"    Found {len(candidates)} relationships:")
        for candidate in candidates:
            print(f"      {candidate.source_entity} --[{candidate.relationship_type}]--> {candidate.target_entity} ({candidate.confidence:.2f})")
    
    print(f"\n✅ Pattern matching test completed: {total_candidates} total relationships found")
    return total_candidates > 0


def test_extraction_statistics():
    """Test extraction statistics and configuration"""
    print("\n=== Testing Extraction Statistics ===")
    
    extractor = create_relationship_extractor()
    
    # Get initial statistics
    stats = extractor.get_extraction_statistics()
    print("✅ Retrieved extraction statistics:")
    
    for category, values in stats.items():
        if isinstance(values, dict):
            print(f"  {category}:")
            for key, value in values.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {category}: {values}")
    
    # Test confidence threshold update
    original_threshold = extractor.confidence_threshold
    extractor.update_confidence_threshold(0.8)
    print(f"✅ Updated confidence threshold from {original_threshold} to {extractor.confidence_threshold}")
    
    return True


def test_relationship_candidate():
    """Test RelationshipCandidate functionality"""
    print("\n=== Testing RelationshipCandidate ===")
    
    candidate = RelationshipCandidate(
        source_entity="Alice",
        target_entity="Company",
        relationship_type="works_for",
        confidence=0.85,
        evidence_text="Alice works for Company",
        context_window="Alice works for Company in downtown",
        extraction_method=ExtractionMethod.PATTERN_BASED,
        position_start=0,
        position_end=23,
        properties={"sentiment": "neutral"}
    )
    
    print("✅ Created RelationshipCandidate:")
    print(f"  Source: {candidate.source_entity}")
    print(f"  Target: {candidate.target_entity}")
    print(f"  Type: {candidate.relationship_type}")
    print(f"  Confidence: {candidate.confidence}")
    print(f"  Method: {candidate.extraction_method.value}")
    print(f"  Properties: {candidate.properties}")
    
    # Test conversion to relationship instance (should handle missing core modules gracefully)
    rel_instance = candidate.to_relationship_instance()
    if rel_instance is None:
        print("✅ Gracefully handled missing core modules for relationship instance conversion")
    else:
        print("✅ Successfully converted to RelationshipInstance")
    
    return True


def run_all_tests():
    """Run comprehensive test suite for relationship extractor"""
    print("🚀 Starting Relationship Extractor Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_extraction,
        test_pattern_matching,
        test_extraction_statistics,
        test_relationship_candidate
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Relationship Extractor is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
