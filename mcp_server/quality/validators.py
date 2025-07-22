#!/usr/bin/env python3
"""
Graph Quality Assessment Framework - Core Validators

This module provides comprehensive quality assessment tools for knowledge graphs,
including entity completeness scoring, relationship accuracy metrics, graph
connectivity analysis, duplicate detection, and consistency validation.

Features:
- Entity completeness scoring
- Relationship accuracy metrics
- Graph connectivity analysis
- Duplicate detection and cleanup
- Consistency validation across relationships
- Quality report generation with actionable insights

Usage:
    from mcp_server.quality.validators import GraphQualityAssessment
    from mcp_server.core.graph_schema import SchemaManager
    
    # Initialize with graph store and schema
    quality_assessor = GraphQualityAssessment(graph_store, schema_manager)
    
    # Run comprehensive quality assessment
    quality_report = quality_assessor.assess_graph_quality()
    
    # Access specific quality metrics
    print(f"Overall Quality Score: {quality_report.overall_score:.2f}")
    print(f"Entity Completeness: {quality_report.completeness_score:.2f}")
    print(f"Relationship Accuracy: {quality_report.accuracy_score:.2f}")
"""

import logging
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import datetime

# Core imports with graceful fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    accuracy_score = precision_score = recall_score = f1_score = None

# Phase 1 core imports
try:
    from ..core.graph_schema import EntityInstance, RelationshipInstance, SchemaManager, GraphSchema
    from ..core.entity_resolution import EntityResolver, MatchResult
    from ..core.graph_analytics import GraphAnalytics
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    # Fallback classes for testing
    class EntityInstance:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class RelationshipInstance:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class SchemaManager:
        def __init__(self):
            self.entity_types = set()
            self.relationship_types = set()

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of quality issues that can be detected"""
    DUPLICATE_ENTITY = "duplicate_entity"
    INCOMPLETE_ENTITY = "incomplete_entity"
    INVALID_RELATIONSHIP = "invalid_relationship"
    ORPHANED_ENTITY = "orphaned_entity"
    INCONSISTENT_DATA = "inconsistent_data"
    MISSING_REQUIRED_PROPERTY = "missing_required_property"
    INVALID_PROPERTY_VALUE = "invalid_property_value"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    WEAK_CONNECTIVITY = "weak_connectivity"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


class IssueSeverity(Enum):
    """Severity levels for quality issues"""
    CRITICAL = "critical"  # Data corruption, breaks functionality
    HIGH = "high"         # Significant quality impact
    MEDIUM = "medium"     # Moderate quality impact
    LOW = "low"          # Minor quality impact
    INFO = "info"        # Informational only


@dataclass
class QualityIssue:
    """Represents a quality issue found in the knowledge graph"""
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    affected_entities: List[str] = field(default_factory=list)
    affected_relationships: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate issue data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    timestamp: datetime.datetime
    overall_score: float
    completeness_score: float
    accuracy_score: float
    connectivity_score: float
    consistency_score: float
    
    # Detailed metrics
    total_entities: int
    total_relationships: int
    unique_entities: int
    orphaned_entities: int
    duplicate_entities: int
    
    # Issues found
    issues: List[QualityIssue] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Processing stats
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Validate report data"""
        for score in [self.overall_score, self.completeness_score, 
                     self.accuracy_score, self.connectivity_score, self.consistency_score]:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All scores must be between 0.0 and 1.0")
    
    @property
    def critical_issues(self) -> List[QualityIssue]:
        """Get critical issues only"""
        return [issue for issue in self.issues if issue.severity == IssueSeverity.CRITICAL]
    
    @property
    def high_priority_issues(self) -> List[QualityIssue]:
        """Get high priority issues"""
        return [issue for issue in self.issues 
                if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]]
    
    def get_issues_by_type(self, issue_type: IssueType) -> List[QualityIssue]:
        """Get issues of a specific type"""
        return [issue for issue in self.issues if issue.issue_type == issue_type]


class GraphQualityAssessment:
    """
    Comprehensive Quality Assessment Framework for Knowledge Graphs
    
    Provides tools for evaluating and improving knowledge graph quality across
    multiple dimensions including completeness, accuracy, connectivity, and consistency.
    """
    
    def __init__(self, graph_store=None, schema_manager: Optional[SchemaManager] = None):
        """
        Initialize quality assessment framework
        
        Args:
            graph_store: Graph storage backend (optional for testing)
            schema_manager: Schema manager for validation (optional)
        """
        self.graph_store = graph_store
        self.schema_manager = schema_manager or SchemaManager()
        
        # Initialize analytics if available
        if CORE_MODULES_AVAILABLE and NETWORKX_AVAILABLE:
            self.graph_analytics = GraphAnalytics()
        else:
            self.graph_analytics = None
            logger.warning("Graph analytics not available - some quality metrics will be limited")
        
        # Quality thresholds (configurable)
        self.quality_thresholds = {
            'completeness_min': 0.8,
            'accuracy_min': 0.9,
            'connectivity_min': 0.7,
            'consistency_min': 0.95,
            'overall_min': 0.8
        }
        
        logger.info("🧪 Graph Quality Assessment Framework initialized")
    
    def assess_graph_quality(self, entities: List[EntityInstance] = None, 
                           relationships: List[RelationshipInstance] = None) -> QualityReport:
        """
        Run comprehensive quality assessment on the knowledge graph
        
        Args:
            entities: List of entities to assess (optional, will use graph_store if available)
            relationships: List of relationships to assess (optional)
            
        Returns:
            QualityReport: Comprehensive quality assessment report
        """
        start_time = datetime.datetime.now()
        logger.info("🔍 Starting comprehensive graph quality assessment...")
        
        # Get data from graph store if not provided
        if entities is None and self.graph_store:
            entities = self._get_entities_from_store()
        elif entities is None:
            entities = []
            
        if relationships is None and self.graph_store:
            relationships = self._get_relationships_from_store()
        elif relationships is None:
            relationships = []
        
        # Calculate quality scores
        completeness_score = self.calculate_entity_completeness(entities)
        accuracy_score = self.calculate_relationship_accuracy(relationships)
        connectivity_score = self.calculate_connectivity_score(entities, relationships)
        consistency_score = self.calculate_consistency_score(entities, relationships)
        
        # Calculate overall score (weighted average)
        overall_score = (
            completeness_score * 0.3 +
            accuracy_score * 0.3 +
            connectivity_score * 0.2 +
            consistency_score * 0.2
        )
        
        # Detect quality issues
        issues = self._detect_quality_issues(entities, relationships)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, {
            'completeness': completeness_score,
            'accuracy': accuracy_score,
            'connectivity': connectivity_score,
            'consistency': consistency_score
        })
        
        # Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Create quality report
        report = QualityReport(
            timestamp=datetime.datetime.now(),
            overall_score=overall_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            connectivity_score=connectivity_score,
            consistency_score=consistency_score,
            total_entities=len(entities),
            total_relationships=len(relationships),
            unique_entities=len(set(e.id if hasattr(e, 'id') else str(e) for e in entities)),
            orphaned_entities=self._count_orphaned_entities(entities, relationships),
            duplicate_entities=self._count_duplicate_entities(entities),
            issues=issues,
            recommendations=recommendations,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"✅ Quality assessment completed in {processing_time:.2f}s")
        logger.info(f"📊 Overall quality score: {overall_score:.2f}")
        logger.info(f"🔍 Found {len(issues)} quality issues")
        
        return report
    
    def calculate_entity_completeness(self, entities: List[EntityInstance]) -> float:
        """
        Calculate entity completeness score based on required properties
        
        Args:
            entities: List of entities to assess
            
        Returns:
            float: Completeness score between 0.0 and 1.0
        """
        if not entities:
            return 0.0
        
        total_completeness = 0.0
        
        for entity in entities:
            entity_completeness = self._calculate_single_entity_completeness(entity)
            total_completeness += entity_completeness
        
        average_completeness = total_completeness / len(entities)
        logger.debug(f"Entity completeness score: {average_completeness:.3f}")
        
        return average_completeness
    
    def calculate_relationship_accuracy(self, relationships: List[RelationshipInstance]) -> float:
        """
        Calculate relationship accuracy score based on schema validation
        
        Args:
            relationships: List of relationships to assess
            
        Returns:
            float: Accuracy score between 0.0 and 1.0
        """
        if not relationships:
            return 1.0  # No relationships means no inaccuracies
        
        valid_relationships = 0
        
        for relationship in relationships:
            if self._validate_relationship_accuracy(relationship):
                valid_relationships += 1
        
        accuracy_score = valid_relationships / len(relationships)
        logger.debug(f"Relationship accuracy score: {accuracy_score:.3f}")
        
        return accuracy_score
    
    def calculate_connectivity_score(self, entities: List[EntityInstance], 
                                   relationships: List[RelationshipInstance]) -> float:
        """
        Calculate graph connectivity score
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            float: Connectivity score between 0.0 and 1.0
        """
        if not entities:
            return 0.0
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available - using simplified connectivity calculation")
            return self._calculate_simple_connectivity(entities, relationships)
        
        # Build NetworkX graph for analysis
        graph = nx.Graph()
        
        # Add entities as nodes
        for entity in entities:
            entity_id = entity.id if hasattr(entity, 'id') else str(entity)
            graph.add_node(entity_id)
        
        # Add relationships as edges
        for relationship in relationships:
            source_id = relationship.source_entity_id if hasattr(relationship, 'source_entity_id') else str(relationship.source_entity_id)
            target_id = relationship.target_entity_id if hasattr(relationship, 'target_entity_id') else str(relationship.target_entity_id)
            graph.add_edge(source_id, target_id)
        
        # Calculate connectivity metrics
        if len(graph.nodes) == 0:
            return 0.0
        
        # Number of connected components (lower is better)
        num_components = nx.number_connected_components(graph)
        component_score = 1.0 - (num_components - 1) / max(1, len(graph.nodes) - 1)
        
        # Average clustering coefficient
        clustering_score = nx.average_clustering(graph) if len(graph.nodes) > 2 else 0.0
        
        # Combine metrics
        connectivity_score = (component_score * 0.7 + clustering_score * 0.3)
        
        logger.debug(f"Connectivity score: {connectivity_score:.3f}")
        return connectivity_score
    
    def calculate_consistency_score(self, entities: List[EntityInstance], 
                                  relationships: List[RelationshipInstance]) -> float:
        """
        Calculate consistency score across entities and relationships
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            float: Consistency score between 0.0 and 1.0
        """
        if not entities and not relationships:
            return 1.0
        
        total_checks = 0
        passed_checks = 0
        
        # Check entity consistency
        entity_consistency_checks = self._check_entity_consistency(entities)
        total_checks += entity_consistency_checks['total']
        passed_checks += entity_consistency_checks['passed']
        
        # Check relationship consistency
        relationship_consistency_checks = self._check_relationship_consistency(relationships)
        total_checks += relationship_consistency_checks['total']
        passed_checks += relationship_consistency_checks['passed']
        
        # Check cross-consistency
        cross_consistency_checks = self._check_cross_consistency(entities, relationships)
        total_checks += cross_consistency_checks['total']
        passed_checks += cross_consistency_checks['passed']
        
        consistency_score = passed_checks / max(1, total_checks)
        logger.debug(f"Consistency score: {consistency_score:.3f}")
        
        return consistency_score
    
    def _get_entities_from_store(self) -> List[EntityInstance]:
        """Get entities from graph store (placeholder for implementation)"""
        # This would connect to the actual graph store
        # For now, return empty list
        return []
    
    def _get_relationships_from_store(self) -> List[RelationshipInstance]:
        """Get relationships from graph store (placeholder for implementation)"""
        # This would connect to the actual graph store
        # For now, return empty list
        return []
    
    def _calculate_single_entity_completeness(self, entity: EntityInstance) -> float:
        """Calculate completeness for a single entity"""
        # Check required properties based on entity type
        required_properties = self._get_required_properties(entity)
        if not required_properties:
            return 1.0  # No requirements means complete
        
        present_properties = 0
        for prop in required_properties:
            if hasattr(entity, prop) and getattr(entity, prop) is not None:
                present_properties += 1
        
        return present_properties / len(required_properties)
    
    def _get_required_properties(self, entity: EntityInstance) -> List[str]:
        """Get required properties for an entity type"""
        # This would use schema manager to get requirements
        # For now, return basic properties
        return ['name', 'entity_type']
    
    def _validate_relationship_accuracy(self, relationship: RelationshipInstance) -> bool:
        """Validate if a relationship is accurate according to schema"""
        # Check if relationship type is valid
        rel_type = relationship.relationship_type if hasattr(relationship, 'relationship_type') else 'unknown'
        
        # Check if source and target exist and are valid
        if not (hasattr(relationship, 'source_entity_id') and hasattr(relationship, 'target_entity_id')):
            return False
        
        # Additional validation would go here
        return True
    
    def _calculate_simple_connectivity(self, entities: List[EntityInstance], 
                                     relationships: List[RelationshipInstance]) -> float:
        """Simple connectivity calculation without NetworkX"""
        if not entities:
            return 0.0
        
        # Count entities with at least one relationship
        entity_ids = set()
        for rel in relationships:
            if hasattr(rel, 'source_entity_id'):
                entity_ids.add(rel.source_entity_id)
            if hasattr(rel, 'target_entity_id'):
                entity_ids.add(rel.target_entity_id)
        
        connected_entities = len(entity_ids)
        total_entities = len(entities)
        
        return connected_entities / total_entities if total_entities > 0 else 0.0
    
    def _check_entity_consistency(self, entities: List[EntityInstance]) -> Dict[str, int]:
        """Check consistency within entities"""
        total = 0
        passed = 0
        
        for entity in entities:
            total += 1
            # Check basic consistency (has required attributes)
            if hasattr(entity, 'type') and hasattr(entity, 'id'):
                passed += 1
        
        return {'total': total, 'passed': passed}
    
    def _check_relationship_consistency(self, relationships: List[RelationshipInstance]) -> Dict[str, int]:
        """Check consistency within relationships"""
        total = 0
        passed = 0
        
        for relationship in relationships:
            total += 1
            # Check basic consistency
            if (hasattr(relationship, 'source_entity_id') and 
                hasattr(relationship, 'target_entity_id') and 
                hasattr(relationship, 'relation_type')):
                passed += 1
        
        return {'total': total, 'passed': passed}
    
    def _check_cross_consistency(self, entities: List[EntityInstance], 
                               relationships: List[RelationshipInstance]) -> Dict[str, int]:
        """Check consistency between entities and relationships"""
        total = 0
        passed = 0
        
        # Build entity ID set
        entity_ids = set()
        for entity in entities:
            if hasattr(entity, 'id'):
                entity_ids.add(entity.id)
        
        # Check that all relationships reference existing entities
        for relationship in relationships:
            if hasattr(relationship, 'source_entity_id') and hasattr(relationship, 'target_entity_id'):
                total += 2  # Check both source and target
                if relationship.source_entity_id in entity_ids:
                    passed += 1
                if relationship.target_entity_id in entity_ids:
                    passed += 1
        
        return {'total': total, 'passed': passed}
    
    def _detect_quality_issues(self, entities: List[EntityInstance], 
                             relationships: List[RelationshipInstance]) -> List[QualityIssue]:
        """Detect various quality issues in the graph"""
        issues = []
        
        # Detect duplicate entities
        issues.extend(self._detect_duplicate_entities(entities))
        
        # Detect incomplete entities
        issues.extend(self._detect_incomplete_entities(entities))
        
        # Detect invalid relationships
        issues.extend(self._detect_invalid_relationships(relationships))
        
        # Detect orphaned entities
        issues.extend(self._detect_orphaned_entities(entities, relationships))
        
        return issues
    
    def _detect_duplicate_entities(self, entities: List[EntityInstance]) -> List[QualityIssue]:
        """Detect potential duplicate entities"""
        issues = []
        
        # Group entities by name for duplicate detection
        entity_groups = defaultdict(list)
        for entity in entities:
            name = getattr(entity, 'name', str(entity))
            entity_groups[name].append(entity)
        
        # Find duplicates
        for name, group in entity_groups.items():
            if len(group) > 1:
                entity_ids = [getattr(e, 'id', str(e)) for e in group]
                issue = QualityIssue(
                    issue_type=IssueType.DUPLICATE_ENTITY,
                    severity=IssueSeverity.MEDIUM,
                    description=f"Potential duplicate entities found with name '{name}'",
                    affected_entities=entity_ids,
                    suggested_action="Review and merge duplicate entities",
                    confidence=0.8
                )
                issues.append(issue)
        
        return issues
    
    def _detect_incomplete_entities(self, entities: List[EntityInstance]) -> List[QualityIssue]:
        """Detect entities with missing required properties"""
        issues = []
        
        for entity in entities:
            completeness = self._calculate_single_entity_completeness(entity)
            if completeness < 0.7:  # Threshold for incomplete entities
                entity_id = getattr(entity, 'id', str(entity))
                issue = QualityIssue(
                    issue_type=IssueType.INCOMPLETE_ENTITY,
                    severity=IssueSeverity.MEDIUM if completeness < 0.5 else IssueSeverity.LOW,
                    description=f"Entity '{entity_id}' is incomplete ({completeness:.1%} complete)",
                    affected_entities=[entity_id],
                    suggested_action="Add missing required properties",
                    confidence=1.0 - completeness
                )
                issues.append(issue)
        
        return issues
    
    def _detect_invalid_relationships(self, relationships: List[RelationshipInstance]) -> List[QualityIssue]:
        """Detect invalid relationships"""
        issues = []
        
        for relationship in relationships:
            if not self._validate_relationship_accuracy(relationship):
                rel_id = getattr(relationship, 'id', str(relationship))
                issue = QualityIssue(
                    issue_type=IssueType.INVALID_RELATIONSHIP,
                    severity=IssueSeverity.HIGH,
                    description=f"Invalid relationship '{rel_id}'",
                    affected_relationships=[rel_id],
                    suggested_action="Review and fix relationship properties",
                    confidence=0.9
                )
                issues.append(issue)
        
        return issues
    
    def _detect_orphaned_entities(self, entities: List[EntityInstance], 
                                relationships: List[RelationshipInstance]) -> List[QualityIssue]:
        """Detect entities with no relationships"""
        issues = []
        
        # Get entities referenced in relationships
        referenced_entities = set()
        for rel in relationships:
            if hasattr(rel, 'source_entity_id'):
                referenced_entities.add(rel.source_entity_id)
            if hasattr(rel, 'target_entity_id'):
                referenced_entities.add(rel.target_entity_id)
        
        # Find orphaned entities
        orphaned = []
        for entity in entities:
            entity_id = getattr(entity, 'id', str(entity))
            if entity_id not in referenced_entities:
                orphaned.append(entity_id)
        
        if orphaned:
            issue = QualityIssue(
                issue_type=IssueType.ORPHANED_ENTITY,
                severity=IssueSeverity.LOW,
                description=f"Found {len(orphaned)} orphaned entities with no relationships",
                affected_entities=orphaned,
                suggested_action="Review orphaned entities and add relationships or remove if unnecessary",
                confidence=1.0
            )
            issues.append(issue)
        
        return issues
    
    def _count_orphaned_entities(self, entities: List[EntityInstance], 
                               relationships: List[RelationshipInstance]) -> int:
        """Count orphaned entities"""
        referenced_entities = set()
        for rel in relationships:
            if hasattr(rel, 'source_entity_id'):
                referenced_entities.add(rel.source_entity_id)
            if hasattr(rel, 'target_entity_id'):
                referenced_entities.add(rel.target_entity_id)
        
        orphaned_count = 0
        for entity in entities:
            entity_id = getattr(entity, 'id', str(entity))
            if entity_id not in referenced_entities:
                orphaned_count += 1
        
        return orphaned_count
    
    def _count_duplicate_entities(self, entities: List[EntityInstance]) -> int:
        """Count potential duplicate entities"""
        entity_names = [getattr(e, 'name', str(e)) for e in entities]
        name_counts = Counter(entity_names)
        duplicates = sum(count - 1 for count in name_counts.values() if count > 1)
        return duplicates
    
    def _generate_recommendations(self, issues: List[QualityIssue], 
                                scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on quality assessment"""
        recommendations = []
        
        # Score-based recommendations
        if scores['completeness'] < self.quality_thresholds['completeness_min']:
            recommendations.append(
                "📝 Improve entity completeness by adding missing required properties"
            )
        
        if scores['accuracy'] < self.quality_thresholds['accuracy_min']:
            recommendations.append(
                "🎯 Review and validate relationship accuracy against schema definitions"
            )
        
        if scores['connectivity'] < self.quality_thresholds['connectivity_min']:
            recommendations.append(
                "🔗 Enhance graph connectivity by adding more relationships between entities"
            )
        
        if scores['consistency'] < self.quality_thresholds['consistency_min']:
            recommendations.append(
                "⚖️ Address consistency issues across entities and relationships"
            )
        
        # Issue-based recommendations
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            recommendations.append(
                f"🚨 Address {len(critical_issues)} critical issues immediately"
            )
        
        high_priority = [i for i in issues if i.severity == IssueSeverity.HIGH]
        if high_priority:
            recommendations.append(
                f"⚠️ Review {len(high_priority)} high-priority quality issues"
            )
        
        duplicate_issues = [i for i in issues if i.issue_type == IssueType.DUPLICATE_ENTITY]
        if duplicate_issues:
            recommendations.append(
                "🔍 Run entity deduplication process to merge similar entities"
            )
        
        orphaned_issues = [i for i in issues if i.issue_type == IssueType.ORPHANED_ENTITY]
        if orphaned_issues:
            recommendations.append(
                "🏝️ Review orphaned entities and establish relevant relationships"
            )
        
        if not recommendations:
            recommendations.append("✅ Knowledge graph quality is good - continue monitoring")
        
        return recommendations


def create_graph_quality_assessor(graph_store=None, schema_manager: Optional[SchemaManager] = None) -> GraphQualityAssessment:
    """
    Factory function to create a GraphQualityAssessment instance
    
    Args:
        graph_store: Graph storage backend (optional)
        schema_manager: Schema manager for validation (optional)
        
    Returns:
        GraphQualityAssessment: Configured quality assessment instance
    """
    return GraphQualityAssessment(graph_store=graph_store, schema_manager=schema_manager)


# Example usage
if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample entities and relationships for testing
    sample_entities = [
        EntityInstance(id="1", name="John Smith", type="Person"),
        EntityInstance(id="2", name="Google", type="Organization"),
        EntityInstance(id="3", name="", type="Person"),  # Incomplete entity
    ]
    
    sample_relationships = [
        RelationshipInstance(source_entity_id="1", target_entity_id="2", relation_type="works_at"),
    ]
    
    # Run quality assessment
    assessor = create_graph_quality_assessor()
    report = assessor.assess_graph_quality(sample_entities, sample_relationships)
    
    print(f"📊 Quality Assessment Results:")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Completeness: {report.completeness_score:.2f}")
    print(f"Accuracy: {report.accuracy_score:.2f}")
    print(f"Connectivity: {report.connectivity_score:.2f}")
    print(f"Consistency: {report.consistency_score:.2f}")
    print(f"Issues Found: {len(report.issues)}")
    print(f"Recommendations: {len(report.recommendations)}")
