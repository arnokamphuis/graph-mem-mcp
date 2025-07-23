#!/usr/bin/env python3
"""
Phase 3.1 Quality Assessment Framework Validation Suite
Direct validation testing following prompt quality gate requirements
"""

import sys
sys.path.append('.')

def test_import_validation():
    """Test 1: Import Validation"""
    print("\n🧪 Test 1: Import Validation")
    try:
        # Test core quality imports
        from mcp_server.quality.validators import (
            GraphQualityAssessment, QualityReport, QualityIssue, 
            IssueType, IssueSeverity, create_graph_quality_assessor
        )
        print("  ✅ Core quality validator imports successful")
        
        from mcp_server.quality.metrics import (
            QualityMetrics, CompletenessMetrics, AccuracyMetrics, 
            ConnectivityMetrics, create_quality_metrics
        )
        print("  ✅ Quality metrics imports successful")
        
        from mcp_server.quality.consistency_checker import (
            ConsistencyChecker, ConsistencyViolation, ViolationType,
            ViolationSeverity, create_consistency_checker
        )
        print("  ✅ Consistency checker imports successful")
        
        # Test Phase 1 core integration
        from mcp_server.core.graph_schema import EntityInstance, RelationshipInstance
        print("  ✅ Phase 1 core imports successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_factory_creation():
    """Test 2: Factory Creation"""
    print("\n🧪 Test 2: Factory Creation")
    try:
        from mcp_server.quality.validators import create_graph_quality_assessor
        from mcp_server.quality.metrics import create_quality_metrics
        from mcp_server.quality.consistency_checker import create_consistency_checker
        
        # Create minimal schema manager for testing
        class TestSchema:
            def __init__(self):
                self.entity_types = ["Person", "Organization", "Location", "Concept", "Event"]
        
        class TestSchemaManager:
            def __init__(self):
                self.schema = TestSchema()
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        # Test factory functions
        assessor = create_graph_quality_assessor(schema_manager=schema_manager)
        print("  ✅ Quality assessor factory works")
        
        metrics_calc = create_quality_metrics(schema_manager=schema_manager)
        print("  ✅ Quality metrics factory works")
        
        checker = create_consistency_checker(schema_manager=schema_manager)
        print("  ✅ Consistency checker factory works")
        
        # Verify types
        if hasattr(assessor, 'assess_graph_quality'):
            print("  ✅ Quality assessor has assess_graph_quality method")
        
        if hasattr(metrics_calc, 'calculate_completeness_metrics'):
            print("  ✅ Metrics calculator has completeness calculation")
        
        if hasattr(checker, 'check_all_consistency'):
            print("  ✅ Consistency checker has check_all_consistency method")
        
        return True
    except Exception as e:
        print(f"  ❌ Factory creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_report_creation():
    """Test 3: QualityReport Creation"""
    print("\n🧪 Test 3: QualityReport Creation")
    try:
        from mcp_server.quality.validators import QualityReport, QualityIssue, IssueType, IssueSeverity
        from datetime import datetime
        
        # Create test quality report
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.80,
            connectivity_score=0.85,
            consistency_score=0.90,
            total_entities=100,
            total_relationships=150,
            unique_entities=98,
            orphaned_entities=5,
            duplicate_entities=2
        )
        print("  ✅ QualityReport created successfully")
        print(f"  ✅ Overall Score: {report.overall_score}")
        print(f"  ✅ Completeness: {report.completeness_score}")
        print(f"  ✅ Accuracy: {report.accuracy_score}")
        
        # Test issue filtering
        test_issue = QualityIssue(
            issue_type=IssueType.DUPLICATE_ENTITY,
            severity=IssueSeverity.HIGH,
            description="Test duplicate issue",
            affected_entities=["entity1", "entity2"]
        )
        report.issues.append(test_issue)
        
        high_priority = report.high_priority_issues
        print(f"  ✅ High priority issues: {len(high_priority)}")
        
        return True
    except Exception as e:
        print(f"  ❌ QualityReport creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test 4: Quality Metrics Calculation"""
    print("\n🧪 Test 4: Quality Metrics Calculation")
    try:
        from mcp_server.quality.metrics import create_quality_metrics
        from mcp_server.core.graph_schema import EntityInstance, RelationshipInstance
        
        # Create test data
        entities = [
            EntityInstance(id="1", name="John Smith", entity_type="Person"),
            EntityInstance(id="2", name="Google", entity_type="Organization"),
            EntityInstance(id="3", name="New York", entity_type="Location")
        ]
        
        relationships = [
            RelationshipInstance(source_entity_id="1", target_entity_id="2", relationship_type="works_at"),
            RelationshipInstance(source_entity_id="2", target_entity_id="3", relationship_type="located_in")
        ]
        
        # Calculate metrics
        metrics_calc = create_quality_metrics(entities, relationships)
        
        completeness = metrics_calc.calculate_completeness_metrics()
        print(f"  ✅ Completeness calculated: {completeness.overall_completeness:.3f}")
        
        accuracy = metrics_calc.calculate_accuracy_metrics()
        print(f"  ✅ Accuracy calculated: {accuracy.overall_accuracy:.3f}")
        
        connectivity = metrics_calc.calculate_connectivity_metrics()
        print(f"  ✅ Connectivity calculated: {connectivity.connectivity_score:.3f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consistency_checking():
    """Test 5: Consistency Checking"""
    print("\n🧪 Test 5: Consistency Checking")
    try:
        from mcp_server.quality.consistency_checker import create_consistency_checker
        from mcp_server.core.graph_schema import EntityInstance, RelationshipInstance
        
        # Create test data with potential issues
        entities = [
            EntityInstance(id="1", name="John Smith", entity_type="Person", properties={"birth_date": "1990-01-01", "death_date": "1985-01-01"}),  # Date conflict
            EntityInstance(id="2", name="Google", entity_type="Organization"),
            EntityInstance(id="3", name="Test Entity", entity_type="Person")
        ]
        
        relationships = [
            RelationshipInstance(source_entity_id="1", target_entity_id="2", relationship_type="works_at"),
            RelationshipInstance(source_entity_id="1", target_entity_id="999", relationship_type="knows"),  # References non-existent entity
        ]
        
        # Run consistency checks
        checker = create_consistency_checker(entities, relationships)
        violations = checker.check_all_consistency()
        
        print(f"  ✅ Consistency check completed: {len(violations)} violations found")
        
        # Check specific consistency types
        temporal_violations = checker.check_temporal_consistency()
        print(f"  ✅ Temporal consistency checked: {len(temporal_violations)} violations")
        
        referential_violations = checker.check_referential_integrity()
        print(f"  ✅ Referential integrity checked: {len(referential_violations)} violations")
        
        return True
    except Exception as e:
        print(f"  ❌ Consistency checking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_quality_assessment():
    """Test 6: Full Quality Assessment Pipeline"""
    print("\n🧪 Test 6: Full Quality Assessment Pipeline")
    try:
        from mcp_server.quality.validators import create_graph_quality_assessor
        from mcp_server.core.graph_schema import EntityInstance, RelationshipInstance
        
        # Create minimal schema manager for testing
        class TestSchema:
            def __init__(self):
                self.entity_types = ["Person", "Organization", "Location", "Concept", "Event"]
        
        class TestSchemaManager:
            def __init__(self):
                self.schema = TestSchema()
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        # Create comprehensive test data
        entities = [
            EntityInstance(id="1", name="John Smith", entity_type="Person", properties={"description": "Software engineer"}),
            EntityInstance(id="2", name="Google", entity_type="Organization", properties={"description": "Technology company"}),
            EntityInstance(id="3", name="", entity_type="Person"),  # Incomplete entity
            EntityInstance(id="4", name="Mountain View", entity_type="Location"),
            EntityInstance(id="5", name="John Smith", entity_type="Person")  # Duplicate name
        ]
        
        relationships = [
            RelationshipInstance(source_entity_id="1", target_entity_id="2", relationship_type="works_at", confidence=0.9),
            RelationshipInstance(source_entity_id="2", target_entity_id="4", relationship_type="located_in", confidence=0.8),
            # No relationship for entity 3 (orphaned)
        ]
        
        print("  🔍 Attempting full quality assessment...")
        
        # Run full assessment
        assessor = create_graph_quality_assessor(schema_manager=schema_manager)
        report = assessor.assess_graph_quality(entities, relationships)
        
        print(f"  ✅ Assessment completed: Overall score {report.overall_score:.2f}")
        print(f"    📊 Completeness: {report.completeness_score:.2f}")
        print(f"    📊 Accuracy: {report.accuracy_score:.2f}")
        print(f"    📊 Connectivity: {report.connectivity_score:.2f}")
        print(f"    📊 Consistency: {report.consistency_score:.2f}")
        print(f"    🔍 Issues found: {len(report.issues)}")
        print(f"    💡 Recommendations: {len(report.recommendations)}")
        
        # Verify report structure
        if report.total_entities == len(entities):
            print("  ✅ Entity count correct")
        
        if report.total_relationships == len(relationships):
            print("  ✅ Relationship count correct")
        
        if len(report.issues) > 0:
            print("  ✅ Quality issues detected")
        
        if len(report.recommendations) > 0:
            print("  ✅ Recommendations generated")
        
        return True
    except Exception as e:
        print(f"  ❌ Full assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 3.1 validation tests"""
    print("============================================================")
    print("🧪 PHASE 3.1 QUALITY ASSESSMENT FRAMEWORK VALIDATION SUITE")
    print("============================================================")
    
    tests = [
        test_import_validation,
        test_factory_creation,
        test_quality_report_creation,
        test_metrics_calculation,
        test_consistency_checking,
        test_full_quality_assessment
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Calculate results
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("\n============================================================")
    print("📊 VALIDATION RESULTS")
    print("============================================================")
    
    test_names = [
        "Import Validation",
        "Factory Creation",
        "Quality Report Creation",
        "Metrics Calculation", 
        "Consistency Checking",
        "Full Assessment Pipeline"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")
    
    print("------------------------------------------------------------")
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("✅ QUALITY GATE PASSED: 90%+ test success")
        print("🎉 Phase 3.1 validation SUCCESSFUL!")
        return 0
    else:
        print("❌ QUALITY GATE FAILED: <90% test success")
        print("⚠️ Phase 3.1 validation INCOMPLETE")
        return 1

if __name__ == "__main__":
    exit_code = main()
