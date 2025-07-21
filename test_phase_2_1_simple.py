#!/usr/bin/env python3
"""
Simple validation script for Phase 2.1 components
"""
import sys
import os

def main():
    print("🧪 Phase 2.1 Validation Test")
    print("=" * 40)
    
    # Test 1: Import validation
    try:
        sys.path.insert(0, '.')
        from mcp_server.extraction.relation_extractor import ExtractionMethod, RelationshipCandidate
        print("✅ Basic imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Enum validation
    try:
        methods = list(ExtractionMethod)
        print(f"✅ ExtractionMethod has {len(methods)} strategies")
        for method in methods:
            print(f"   - {method.value}")
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
        return False
    
    # Test 3: Candidate creation
    try:
        candidate = RelationshipCandidate(
            source_entity="John",
            target_entity="Google",
            relationship_type="works_at", 
            confidence=0.8,
            evidence_text="John works at Google",
            context_window="John works at Google Inc.",
            extraction_method=ExtractionMethod.PATTERN_BASED,
            position_start=0,
            position_end=10
        )
        print("✅ RelationshipCandidate created successfully")
        print(f"   Source: {candidate.source_entity}")
        print(f"   Target: {candidate.target_entity}")
        print(f"   Type: {candidate.relationship_type}")
        print(f"   Confidence: {candidate.confidence}")
    except Exception as e:
        print(f"❌ Candidate creation failed: {e}")
        return False
    
    print("=" * 40)
    print("🎉 Phase 2.1 Basic Validation PASSED")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
