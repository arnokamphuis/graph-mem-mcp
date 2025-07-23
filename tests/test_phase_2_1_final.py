#!/usr/bin/env python3
"""
Direct import test for Phase 2.1 - bypasses relative import issues
Tests core functionality by importing from absolute paths
"""

import sys
import os
import unittest
from typing import List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestPhase21RelationshipExtraction(unittest.TestCase):
    """Test suite for Phase 2.1 Sophisticated Relationship Extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_text = "John Smith works at Google Inc. He graduated from Stanford University in 2018."
        
    def test_basic_imports(self):
        """Test that all required imports work"""
        try:
            from mcp_server.extraction.relation_extractor import (
                ExtractionMethod, RelationshipCandidate, ExtractionContext,
                SophisticatedRelationshipExtractor, create_relationship_extractor
            )
            from mcp_server.core.graph_schema import RelationshipInstance
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_extraction_method_enum(self):
        """Test ExtractionMethod enum has required values"""
        from mcp_server.extraction.relation_extractor import ExtractionMethod
        
        expected_methods = ['TRANSFORMER', 'PATTERN_BASED', 'DEPENDENCY_PARSING', 'RULE_BASED']
        actual_methods = [method.value for method in ExtractionMethod]
        
        for method in expected_methods:
            self.assertIn(method, actual_methods, f"Missing extraction method: {method}")
        
        self.assertGreaterEqual(len(actual_methods), 3, "Should have at least 3 extraction methods")
    
    def test_relationship_candidate_creation(self):
        """Test RelationshipCandidate data structure"""
        from mcp_server.extraction.relation_extractor import RelationshipCandidate, ExtractionMethod
        
        candidate = RelationshipCandidate(
            source_entity="John Smith",
            target_entity="Google Inc",
            relationship_type="works_at",
            confidence=0.85,
            evidence_text="John Smith works at Google Inc",
            context_window="John Smith works at Google Inc. He graduated",
            extraction_method=ExtractionMethod.PATTERN_BASED,
            position_start=0,
            position_end=30
        )
        
        # Validate required fields
        self.assertEqual(candidate.source_entity, "John Smith")
        self.assertEqual(candidate.target_entity, "Google Inc")
        self.assertEqual(candidate.relationship_type, "works_at")
        self.assertEqual(candidate.confidence, 0.85)
        self.assertEqual(candidate.extraction_method, ExtractionMethod.PATTERN_BASED)
        
        # Test RelationshipInstance conversion
        instance = candidate.to_relationship_instance()
        self.assertIsNotNone(instance, "Should convert to RelationshipInstance")
    
    def test_extraction_context_creation(self):
        """Test ExtractionContext data structure"""
        from mcp_server.extraction.relation_extractor import ExtractionContext
        
        context = ExtractionContext(
            text=self.test_text,
            source_id="test_document",
            domain_context="professional"
        )
        
        self.assertEqual(context.text, self.test_text)
        self.assertEqual(context.source_id, "test_document")
        self.assertEqual(context.domain_context, "professional")
        self.assertIsNotNone(context.entities)  # Should have default empty list
    
    def test_extractor_factory_creation(self):
        """Test factory function creates extractor properly"""
        from mcp_server.extraction.relation_extractor import create_relationship_extractor
        
        extractor = create_relationship_extractor(confidence_threshold=0.6)
        self.assertIsNotNone(extractor, "Factory should create extractor")
        self.assertEqual(extractor.confidence_threshold, 0.6)
        
        # Test statistics method exists
        stats = extractor.get_extraction_statistics()
        self.assertIsInstance(stats, dict, "Should return statistics dictionary")
        self.assertIn('confidence_threshold', stats)
    
    def test_basic_relationship_extraction(self):
        """Test basic relationship extraction functionality"""
        from mcp_server.extraction.relation_extractor import (
            create_relationship_extractor, ExtractionContext
        )
        
        # Create extractor with low threshold for testing
        extractor = create_relationship_extractor(confidence_threshold=0.1)
        
        # Create context
        context = ExtractionContext(
            text=self.test_text,
            source_id="test_extraction"
        )
        
        # Extract relationships
        candidates = extractor.extract_relationships(context)
        
        # Validate results
        self.assertIsInstance(candidates, list, "Should return list of candidates")
        
        # Verify candidate structure if any found
        if candidates:
            candidate = candidates[0]
            self.assertTrue(hasattr(candidate, 'source_entity'))
            self.assertTrue(hasattr(candidate, 'target_entity'))
            self.assertTrue(hasattr(candidate, 'relationship_type'))
            self.assertTrue(hasattr(candidate, 'confidence'))
            self.assertGreaterEqual(candidate.confidence, 0.0)
            self.assertLessEqual(candidate.confidence, 1.0)
    
    def test_phase_1_integration(self):
        """Test Phase 1 RelationshipInstance integration"""
        from mcp_server.extraction.relation_extractor import (
            create_relationship_extractor, ExtractionContext
        )
        from mcp_server.core.graph_schema import RelationshipInstance
        
        # Create extractor
        extractor = create_relationship_extractor(confidence_threshold=0.1)
        
        # Create context
        context = ExtractionContext(
            text="Alice manages Bob at TechCorp.",
            source_id="integration_test"
        )
        
        # Extract as Phase 1 instances
        instances = extractor.extract_relationships_as_instances(context)
        
        # Validate instances
        self.assertIsInstance(instances, list, "Should return list of instances")
        
        for instance in instances:
            self.assertIsInstance(instance, RelationshipInstance, 
                                "All items should be RelationshipInstance objects")
            self.assertTrue(hasattr(instance, 'source_entity_id'))
            self.assertTrue(hasattr(instance, 'target_entity_id'))
            self.assertTrue(hasattr(instance, 'relationship_type'))
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        from mcp_server.extraction.relation_extractor import (
            create_relationship_extractor, ExtractionContext
        )
        
        extractor = create_relationship_extractor()
        
        # Test empty text
        context = ExtractionContext(text="", source_id="empty_test")
        candidates = extractor.extract_relationships(context)
        self.assertIsInstance(candidates, list, "Should handle empty text gracefully")
        
        # Test whitespace-only text
        context = ExtractionContext(text="   \n\t  ", source_id="whitespace_test")
        candidates = extractor.extract_relationships(context)
        self.assertIsInstance(candidates, list, "Should handle whitespace-only text")
    
    def test_confidence_calibration(self):
        """Test confidence calibration functionality"""
        from mcp_server.extraction.relation_extractor import (
            RelationshipCandidate, ExtractionMethod, SophisticatedRelationshipExtractor
        )
        
        extractor = SophisticatedRelationshipExtractor(confidence_threshold=0.5)
        
        # Create test candidates for ensemble
        candidates = [
            RelationshipCandidate(
                source_entity="John", target_entity="Google", relationship_type="works_at",
                confidence=0.7, evidence_text="test", context_window="test",
                extraction_method=ExtractionMethod.PATTERN_BASED, position_start=0, position_end=10
            ),
            RelationshipCandidate(
                source_entity="John", target_entity="Google", relationship_type="works_at",
                confidence=0.8, evidence_text="test", context_window="test",
                extraction_method=ExtractionMethod.DEPENDENCY_PARSING, position_start=0, position_end=10
            )
        ]
        
        # Test calibration
        calibrated = extractor._calibrate_confidence(candidates)
        self.assertIsInstance(calibrated, list, "Should return calibrated candidates")
        self.assertGreater(len(calibrated), 0, "Should have calibrated candidates")

def run_comprehensive_tests():
    """Run all tests and provide detailed results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase21RelationshipExtraction)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Calculate coverage metrics
    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    passed_tests = total_tests - failed_tests
    coverage_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 PHASE 2.1 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {coverage_percentage:.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚨 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Quality gate check
    if coverage_percentage >= 90:
        print(f"\n✅ QUALITY GATE PASSED: {coverage_percentage:.1f}% >= 90% required")
        return True
    else:
        print(f"\n❌ QUALITY GATE FAILED: {coverage_percentage:.1f}% < 90% required")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
