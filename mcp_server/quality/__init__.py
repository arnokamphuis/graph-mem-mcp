"""
Quality Assessment Framework for Knowledge Graph Validation

This module provides comprehensive quality assessment tools for knowledge graphs,
including entity completeness scoring, relationship accuracy metrics, graph
connectivity analysis, duplicate detection, and consistency validation.

Quality Modules:
- validators: Core quality assessment framework
- metrics: Graph quality metrics and scoring
- consistency_checker: Cross-relationship consistency validation

Usage:
    from mcp_server.quality import GraphQualityAssessment, QualityMetrics
    
    # Initialize quality assessment
    quality_assessor = GraphQualityAssessment(graph_store, schema_manager)
    
    # Run comprehensive quality check
    quality_report = quality_assessor.assess_graph_quality()
    
    # Get specific metrics
    completeness_score = quality_assessor.calculate_entity_completeness()
    accuracy_score = quality_assessor.calculate_relationship_accuracy()
"""

from .validators import (
    GraphQualityAssessment,
    QualityReport,
    QualityIssue,
    IssueType,
    IssueSeverity
)

from .metrics import (
    QualityMetrics,
    CompletenessMetrics,
    AccuracyMetrics,
    ConnectivityMetrics
)

from .consistency_checker import (
    ConsistencyChecker,
    ConsistencyViolation,
    ViolationType
)

__all__ = [
    # Core Quality Assessment
    'GraphQualityAssessment',
    'QualityReport',
    'QualityIssue', 
    'IssueType',
    'IssueSeverity',
    
    # Quality Metrics
    'QualityMetrics',
    'CompletenessMetrics',
    'AccuracyMetrics',
    'ConnectivityMetrics',
    
    # Consistency Checking
    'ConsistencyChecker',
    'ConsistencyViolation',
    'ViolationType'
]
