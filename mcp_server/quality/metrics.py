#!/usr/bin/env python3
"""
Quality Metrics Module for Knowledge Graph Assessment

This module provides specialized quality metrics for different aspects of 
knowledge graph quality including completeness, accuracy, and connectivity.

Features:
- Completeness metrics for entities and relationships
- Accuracy metrics with confidence scoring
- Connectivity metrics using graph theory
- Performance and scalability metrics
- Temporal quality metrics for tracking changes

Usage:
    from mcp_server.quality.metrics import QualityMetrics, CompletenessMetrics
    
    # Initialize metrics calculator
    metrics = QualityMetrics(entities, relationships)
    
    # Calculate specific metrics
    completeness = metrics.calculate_completeness_metrics()
    accuracy = metrics.calculate_accuracy_metrics()
    connectivity = metrics.calculate_connectivity_metrics()
"""

import logging
import math
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
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
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    silhouette_score = KMeans = None

# Phase 1 core imports with fallbacks
try:
    from ..core.graph_schema import EntityInstance, RelationshipInstance, SchemaManager
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    # Fallback classes
    class EntityInstance:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class RelationshipInstance:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

logger = logging.getLogger(__name__)


@dataclass
class CompletenessMetrics:
    """Metrics for entity and relationship completeness"""
    entity_completeness: float
    relationship_completeness: float
    property_coverage: float
    required_property_coverage: float
    optional_property_coverage: float
    type_distribution: Dict[str, int] = field(default_factory=dict)
    missing_properties: Dict[str, List[str]] = field(default_factory=dict)
    
    @property
    def overall_completeness(self) -> float:
        """Calculate overall completeness score"""
        return (self.entity_completeness * 0.4 + 
                self.relationship_completeness * 0.3 + 
                self.required_property_coverage * 0.3)


@dataclass 
class AccuracyMetrics:
    """Metrics for relationship and entity accuracy"""
    relationship_accuracy: float
    entity_validation_accuracy: float
    schema_compliance: float
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    
    @property
    def overall_accuracy(self) -> float:
        """Calculate overall accuracy score"""
        return (self.relationship_accuracy * 0.4 + 
                self.entity_validation_accuracy * 0.3 + 
                self.schema_compliance * 0.3)


@dataclass
class ConnectivityMetrics:
    """Metrics for graph connectivity and structure"""
    connected_components: int
    largest_component_size: int
    average_degree: float
    clustering_coefficient: float
    diameter: int
    density: float
    centrality_distribution: Dict[str, float] = field(default_factory=dict)
    hub_entities: List[str] = field(default_factory=list)
    isolated_entities: List[str] = field(default_factory=list)
    
    @property
    def connectivity_score(self) -> float:
        """Calculate overall connectivity score"""
        # Normalize metrics to 0-1 scale
        component_score = 1.0 / max(1, self.connected_components)
        clustering_score = self.clustering_coefficient
        density_score = min(1.0, self.density * 10)  # Scale density
        
        return (component_score * 0.4 + clustering_score * 0.3 + density_score * 0.3)


@dataclass
class PerformanceMetrics:
    """Metrics for graph processing performance"""
    processing_time_seconds: float
    memory_usage_mb: float
    entities_per_second: float
    relationships_per_second: float
    index_efficiency: float = 0.0
    query_response_time: float = 0.0


class QualityMetrics:
    """
    Comprehensive Quality Metrics Calculator for Knowledge Graphs
    
    Provides detailed metrics across multiple quality dimensions including
    completeness, accuracy, connectivity, and performance.
    """
    
    def __init__(self, entities: List[EntityInstance] = None, 
                 relationships: List[RelationshipInstance] = None,
                 schema_manager=None):
        """
        Initialize quality metrics calculator
        
        Args:
            entities: List of entities to analyze
            relationships: List of relationships to analyze
            schema_manager: Schema manager for validation
        """
        self.entities = entities or []
        self.relationships = relationships or []
        self.schema_manager = schema_manager
        
        # Build indexes for efficient access
        self._build_indexes()
        
        logger.info(f"🧮 Quality Metrics initialized with {len(self.entities)} entities, {len(self.relationships)} relationships")
    
    def _build_indexes(self):
        """Build indexes for efficient metric calculation"""
        # Entity indexes
        self.entity_by_id = {}
        self.entities_by_type = defaultdict(list)
        
        for entity in self.entities:
            entity_id = getattr(entity, 'id', str(entity))
            self.entity_by_id[entity_id] = entity
            
            entity_type = getattr(entity, 'type', 'unknown')
            self.entities_by_type[entity_type].append(entity)
        
        # Relationship indexes
        self.relationships_by_type = defaultdict(list)
        self.relationships_by_source = defaultdict(list)
        self.relationships_by_target = defaultdict(list)
        
        for relationship in self.relationships:
            rel_type = getattr(relationship, 'type', 'unknown')
            self.relationships_by_type[rel_type].append(relationship)
            
            source_id = getattr(relationship, 'source_entity_id', None)
            if source_id:
                self.relationships_by_source[source_id].append(relationship)
            
            target_id = getattr(relationship, 'target_entity_id', None)
            if target_id:
                self.relationships_by_target[target_id].append(relationship)
    
    def calculate_completeness_metrics(self) -> CompletenessMetrics:
        """
        Calculate comprehensive completeness metrics
        
        Returns:
            CompletenessMetrics: Detailed completeness analysis
        """
        logger.debug("📊 Calculating completeness metrics...")
        
        # Entity completeness
        entity_completeness = self._calculate_entity_completeness()
        
        # Relationship completeness
        relationship_completeness = self._calculate_relationship_completeness()
        
        # Property coverage analysis
        property_coverage_results = self._analyze_property_coverage()
        
        # Type distribution
        type_distribution = {
            entity_type: len(entities) 
            for entity_type, entities in self.entities_by_type.items()
        }
        
        metrics = CompletenessMetrics(
            entity_completeness=entity_completeness,
            relationship_completeness=relationship_completeness,
            property_coverage=property_coverage_results['overall'],
            required_property_coverage=property_coverage_results['required'],
            optional_property_coverage=property_coverage_results['optional'],
            type_distribution=type_distribution,
            missing_properties=property_coverage_results['missing']
        )
        
        logger.debug(f"✅ Completeness metrics calculated - Overall: {metrics.overall_completeness:.3f}")
        return metrics
    
    def calculate_accuracy_metrics(self) -> AccuracyMetrics:
        """
        Calculate comprehensive accuracy metrics
        
        Returns:
            AccuracyMetrics: Detailed accuracy analysis
        """
        logger.debug("📊 Calculating accuracy metrics...")
        
        # Relationship accuracy
        relationship_accuracy = self._calculate_relationship_accuracy()
        
        # Entity validation accuracy
        entity_accuracy = self._calculate_entity_accuracy()
        
        # Schema compliance
        schema_compliance = self._calculate_schema_compliance()
        
        # Confidence distribution
        confidence_dist = self._analyze_confidence_distribution()
        
        # Accuracy by type
        accuracy_by_type = self._calculate_accuracy_by_type()
        
        # Validation errors
        validation_errors = self._collect_validation_errors()
        
        metrics = AccuracyMetrics(
            relationship_accuracy=relationship_accuracy,
            entity_validation_accuracy=entity_accuracy,
            schema_compliance=schema_compliance,
            confidence_distribution=confidence_dist,
            validation_errors=validation_errors,
            accuracy_by_type=accuracy_by_type
        )
        
        logger.debug(f"✅ Accuracy metrics calculated - Overall: {metrics.overall_accuracy:.3f}")
        return metrics
    
    def calculate_connectivity_metrics(self) -> ConnectivityMetrics:
        """
        Calculate comprehensive connectivity metrics
        
        Returns:
            ConnectivityMetrics: Detailed connectivity analysis
        """
        logger.debug("📊 Calculating connectivity metrics...")
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available - using simplified connectivity metrics")
            return self._calculate_simple_connectivity_metrics()
        
        # Build NetworkX graph
        graph = self._build_networkx_graph()
        
        # Basic connectivity metrics
        connected_components = nx.number_connected_components(graph)
        largest_component = max(nx.connected_components(graph), key=len) if graph.nodes else set()
        largest_component_size = len(largest_component)
        
        # Degree metrics
        degrees = dict(graph.degree())
        average_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0
        
        # Clustering
        clustering_coefficient = nx.average_clustering(graph) if len(graph.nodes) > 2 else 0.0
        
        # Diameter (for largest component only)
        diameter = 0
        if largest_component_size > 1:
            largest_subgraph = graph.subgraph(largest_component)
            if nx.is_connected(largest_subgraph):
                diameter = nx.diameter(largest_subgraph)
        
        # Density
        density = nx.density(graph)
        
        # Centrality analysis
        centrality_dist = self._calculate_centrality_distribution(graph)
        
        # Hub entities (high degree)
        hub_threshold = average_degree * 2 if average_degree > 0 else 1
        hub_entities = [node for node, degree in degrees.items() if degree >= hub_threshold]
        
        # Isolated entities (degree 0)
        isolated_entities = [node for node, degree in degrees.items() if degree == 0]
        
        metrics = ConnectivityMetrics(
            connected_components=connected_components,
            largest_component_size=largest_component_size,
            average_degree=average_degree,
            clustering_coefficient=clustering_coefficient,
            diameter=diameter,
            density=density,
            centrality_distribution=centrality_dist,
            hub_entities=hub_entities,
            isolated_entities=isolated_entities
        )
        
        logger.debug(f"✅ Connectivity metrics calculated - Score: {metrics.connectivity_score:.3f}")
        return metrics
    
    def calculate_performance_metrics(self, processing_start_time: datetime.datetime = None) -> PerformanceMetrics:
        """
        Calculate performance metrics
        
        Args:
            processing_start_time: Start time for processing time calculation
            
        Returns:
            PerformanceMetrics: Performance analysis
        """
        logger.debug("📊 Calculating performance metrics...")
        
        # Processing time
        if processing_start_time:
            processing_time = (datetime.datetime.now() - processing_start_time).total_seconds()
        else:
            processing_time = 0.0
        
        # Memory usage (simplified calculation)
        memory_usage = self._estimate_memory_usage()
        
        # Processing rates
        entities_per_second = len(self.entities) / max(0.001, processing_time)
        relationships_per_second = len(self.relationships) / max(0.001, processing_time)
        
        metrics = PerformanceMetrics(
            processing_time_seconds=processing_time,
            memory_usage_mb=memory_usage,
            entities_per_second=entities_per_second,
            relationships_per_second=relationships_per_second
        )
        
        logger.debug(f"✅ Performance metrics calculated - {entities_per_second:.1f} entities/sec")
        return metrics
    
    def _calculate_entity_completeness(self) -> float:
        """Calculate overall entity completeness score"""
        if not self.entities:
            return 1.0
        
        total_completeness = 0.0
        for entity in self.entities:
            entity_completeness = self._calculate_single_entity_completeness(entity)
            total_completeness += entity_completeness
        
        return total_completeness / len(self.entities)
    
    def _calculate_single_entity_completeness(self, entity: EntityInstance) -> float:
        """Calculate completeness for a single entity"""
        required_properties = self._get_required_properties_for_entity(entity)
        if not required_properties:
            return 1.0
        
        present_properties = 0
        for prop in required_properties:
            if hasattr(entity, prop) and getattr(entity, prop) is not None:
                value = getattr(entity, prop)
                # Check if value is meaningful (not empty string, etc.)
                if value != "" and value != []:
                    present_properties += 1
        
        return present_properties / len(required_properties)
    
    def _get_required_properties_for_entity(self, entity: EntityInstance) -> List[str]:
        """Get required properties for an entity based on its type"""
        entity_type = getattr(entity, 'type', 'unknown')
        
        # Basic required properties for all entities
        base_properties = ['id', 'name', 'type']
        
        # Type-specific properties (would come from schema manager)
        type_specific = {
            'Person': ['name', 'type'],
            'Organization': ['name', 'type'],
            'Location': ['name', 'type'],
            'Event': ['name', 'type', 'date'],
            'Concept': ['name', 'type', 'description']
        }
        
        return type_specific.get(entity_type, base_properties)
    
    def _calculate_relationship_completeness(self) -> float:
        """Calculate relationship completeness score"""
        if not self.relationships:
            return 1.0
        
        complete_relationships = 0
        for relationship in self.relationships:
            if self._is_relationship_complete(relationship):
                complete_relationships += 1
        
        return complete_relationships / len(self.relationships)
    
    def _is_relationship_complete(self, relationship: RelationshipInstance) -> bool:
        """Check if a relationship is complete"""
        required_attrs = ['source_entity_id', 'target_entity_id', 'relation_type']
        
        for attr in required_attrs:
            if not hasattr(relationship, attr):
                return False
            value = getattr(relationship, attr)
            if value is None or value == "":
                return False
        
        return True
    
    def _analyze_property_coverage(self) -> Dict[str, Any]:
        """Analyze property coverage across entities"""
        total_possible_properties = 0
        present_properties = 0
        required_present = 0
        total_required = 0
        optional_present = 0
        total_optional = 0
        missing_by_entity = {}
        
        for entity in self.entities:
            entity_id = getattr(entity, 'id', str(entity))
            required_props = self._get_required_properties_for_entity(entity)
            optional_props = self._get_optional_properties_for_entity(entity)
            
            all_props = required_props + optional_props
            total_possible_properties += len(all_props)
            total_required += len(required_props)
            total_optional += len(optional_props)
            
            missing_props = []
            
            for prop in all_props:
                if hasattr(entity, prop) and getattr(entity, prop) is not None:
                    present_properties += 1
                    if prop in required_props:
                        required_present += 1
                    else:
                        optional_present += 1
                else:
                    missing_props.append(prop)
            
            if missing_props:
                missing_by_entity[entity_id] = missing_props
        
        overall_coverage = present_properties / max(1, total_possible_properties)
        required_coverage = required_present / max(1, total_required)
        optional_coverage = optional_present / max(1, total_optional)
        
        return {
            'overall': overall_coverage,
            'required': required_coverage,
            'optional': optional_coverage,
            'missing': missing_by_entity
        }
    
    def _get_optional_properties_for_entity(self, entity: EntityInstance) -> List[str]:
        """Get optional properties for an entity based on its type"""
        entity_type = getattr(entity, 'type', 'unknown')
        
        type_specific = {
            'Person': ['description', 'birth_date', 'location'],
            'Organization': ['description', 'founded_date', 'headquarters'],
            'Location': ['description', 'coordinates', 'country'],
            'Event': ['description', 'location', 'participants'],
            'Concept': ['category', 'related_concepts']
        }
        
        return type_specific.get(entity_type, ['description'])
    
    def _calculate_relationship_accuracy(self) -> float:
        """Calculate relationship accuracy based on validation"""
        if not self.relationships:
            return 1.0
        
        accurate_relationships = 0
        for relationship in self.relationships:
            if self._validate_relationship(relationship):
                accurate_relationships += 1
        
        return accurate_relationships / len(self.relationships)
    
    def _validate_relationship(self, relationship: RelationshipInstance) -> bool:
        """Validate a relationship for accuracy"""
        # Check if source and target entities exist
        source_id = getattr(relationship, 'source_entity_id', None)
        target_id = getattr(relationship, 'target_entity_id', None)
        
        if not source_id or not target_id:
            return False
        
        # Check if entities exist in our entity set
        if source_id not in self.entity_by_id or target_id not in self.entity_by_id:
            return False
        
        # Check relationship type validity
        rel_type = getattr(relationship, 'type', None)
        if not rel_type:
            return False
        
        # Additional validation could include schema compliance
        return True
    
    def _calculate_entity_accuracy(self) -> float:
        """Calculate entity validation accuracy"""
        if not self.entities:
            return 1.0
        
        valid_entities = 0
        for entity in self.entities:
            if self._validate_entity(entity):
                valid_entities += 1
        
        return valid_entities / len(self.entities)
    
    def _validate_entity(self, entity: EntityInstance) -> bool:
        """Validate an entity for accuracy"""
        # Check required attributes
        if not hasattr(entity, 'id') or not getattr(entity, 'id'):
            return False
        
        if not hasattr(entity, 'type') or not getattr(entity, 'type'):
            return False
        
        # Additional validation would include schema compliance
        return True
    
    def _calculate_schema_compliance(self) -> float:
        """Calculate compliance with schema definitions"""
        if not self.schema_manager:
            return 1.0  # No schema to validate against
        
        total_items = len(self.entities) + len(self.relationships)
        if total_items == 0:
            return 1.0
        
        compliant_items = 0
        
        # Check entity compliance
        for entity in self.entities:
            if self._check_entity_schema_compliance(entity):
                compliant_items += 1
        
        # Check relationship compliance
        for relationship in self.relationships:
            if self._check_relationship_schema_compliance(relationship):
                compliant_items += 1
        
        return compliant_items / total_items
    
    def _check_entity_schema_compliance(self, entity: EntityInstance) -> bool:
        """Check if entity complies with schema"""
        # This would use schema_manager for validation
        # For now, basic validation
        return hasattr(entity, 'type') and hasattr(entity, 'id')
    
    def _check_relationship_schema_compliance(self, relationship: RelationshipInstance) -> bool:
        """Check if relationship complies with schema"""
        # This would use schema_manager for validation
        # For now, basic validation
        return (hasattr(relationship, 'type') and 
                hasattr(relationship, 'source_entity_id') and 
                hasattr(relationship, 'target_entity_id'))
    
    def _analyze_confidence_distribution(self) -> Dict[str, int]:
        """Analyze confidence score distribution"""
        confidence_buckets = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        
        for relationship in self.relationships:
            confidence = getattr(relationship, 'confidence', None)
            if confidence is None:
                confidence_buckets['unknown'] += 1
            elif confidence >= 0.8:
                confidence_buckets['high'] += 1
            elif confidence >= 0.6:
                confidence_buckets['medium'] += 1
            else:
                confidence_buckets['low'] += 1
        
        return confidence_buckets
    
    def _calculate_accuracy_by_type(self) -> Dict[str, float]:
        """Calculate accuracy metrics by type"""
        accuracy_by_type = {}
        
        # Entity accuracy by type
        for entity_type, entities in self.entities_by_type.items():
            valid_count = sum(1 for entity in entities if self._validate_entity(entity))
            accuracy_by_type[f"entity_{entity_type}"] = valid_count / len(entities)
        
        # Relationship accuracy by type
        for rel_type, relationships in self.relationships_by_type.items():
            valid_count = sum(1 for rel in relationships if self._validate_relationship(rel))
            accuracy_by_type[f"relationship_{rel_type}"] = valid_count / len(relationships)
        
        return accuracy_by_type
    
    def _collect_validation_errors(self) -> List[str]:
        """Collect validation errors found during analysis"""
        errors = []
        
        # Entity validation errors
        for entity in self.entities:
            entity_id = getattr(entity, 'id', 'unknown')
            if not self._validate_entity(entity):
                errors.append(f"Invalid entity: {entity_id}")
        
        # Relationship validation errors
        for relationship in self.relationships:
            if not self._validate_relationship(relationship):
                source = getattr(relationship, 'source_entity_id', 'unknown')
                target = getattr(relationship, 'target_id', 'unknown')
                rel_type = getattr(relationship, 'type', 'unknown')
                errors.append(f"Invalid relationship: {source} -> {target} ({rel_type})")
        
        return errors[:10]  # Limit to first 10 errors
    
    def _build_networkx_graph(self) -> 'nx.Graph':
        """Build NetworkX graph from entities and relationships"""
        graph = nx.Graph()
        
        # Add nodes
        for entity in self.entities:
            entity_id = getattr(entity, 'id', str(entity))
            entity_type = getattr(entity, 'type', 'unknown')
            graph.add_node(entity_id, type=entity_type)
        
        # Add edges
        for relationship in self.relationships:
            source_id = getattr(relationship, 'source_id', None)
            target_id = getattr(relationship, 'target_id', None)
            rel_type = getattr(relationship, 'type', 'unknown')
            
            if source_id and target_id:
                graph.add_edge(source_id, target_id, type=rel_type)
        
        return graph
    
    def _calculate_centrality_distribution(self, graph: 'nx.Graph') -> Dict[str, float]:
        """Calculate centrality measures for the graph"""
        if len(graph.nodes) == 0:
            return {}
        
        centrality_measures = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            centrality_measures['avg_degree_centrality'] = np.mean(list(degree_centrality.values())) if NUMPY_AVAILABLE else sum(degree_centrality.values()) / len(degree_centrality)
            
            # Betweenness centrality (for smaller graphs)
            if len(graph.nodes) <= 1000:  # Limit for performance
                betweenness_centrality = nx.betweenness_centrality(graph)
                centrality_measures['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values())) if NUMPY_AVAILABLE else sum(betweenness_centrality.values()) / len(betweenness_centrality)
            
            # Closeness centrality (for connected components)
            if nx.is_connected(graph) and len(graph.nodes) <= 1000:
                closeness_centrality = nx.closeness_centrality(graph)
                centrality_measures['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values())) if NUMPY_AVAILABLE else sum(closeness_centrality.values()) / len(closeness_centrality)
        
        except Exception as e:
            logger.warning(f"Error calculating centrality measures: {e}")
        
        return centrality_measures
    
    def _calculate_simple_connectivity_metrics(self) -> ConnectivityMetrics:
        """Calculate simplified connectivity metrics without NetworkX"""
        logger.info("📊 Using simplified connectivity metrics (NetworkX not available)")
        
        # Count entities with relationships
        connected_entities = set()
        for rel in self.relationships:
            if hasattr(rel, 'source_id'):
                connected_entities.add(rel.source_id)
            if hasattr(rel, 'target_id'):
                connected_entities.add(rel.target_id)
        
        # Basic metrics
        total_entities = len(self.entities)
        connected_count = len(connected_entities)
        isolated_count = total_entities - connected_count
        
        # Simple connectivity score
        connectivity_score = connected_count / max(1, total_entities)
        
        # Estimate components (simplified)
        components = 1 if connected_count > 0 else 0
        if isolated_count > 0:
            components += isolated_count
        
        # Calculate average degree
        degree_sum = len(self.relationships) * 2  # Each relationship adds 2 to total degree
        avg_degree = degree_sum / max(1, total_entities)
        
        return ConnectivityMetrics(
            connected_components=components,
            largest_component_size=connected_count,
            average_degree=avg_degree,
            clustering_coefficient=0.0,  # Can't calculate without NetworkX
            diameter=0,  # Can't calculate without NetworkX
            density=len(self.relationships) / max(1, total_entities * (total_entities - 1) / 2),
            centrality_distribution={},
            hub_entities=[],
            isolated_entities=[getattr(e, 'id', str(e)) for e in self.entities 
                             if getattr(e, 'id', str(e)) not in connected_entities]
        )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimation
        entity_size = len(self.entities) * 0.001  # ~1KB per entity
        relationship_size = len(self.relationships) * 0.0005  # ~0.5KB per relationship
        
        return entity_size + relationship_size


def create_quality_metrics(entities: List[EntityInstance] = None,
                         relationships: List[RelationshipInstance] = None,
                         schema_manager=None) -> QualityMetrics:
    """
    Factory function to create a QualityMetrics instance
    
    Args:
        entities: List of entities to analyze
        relationships: List of relationships to analyze
        schema_manager: Schema manager for validation
        
    Returns:
        QualityMetrics: Configured metrics calculator
    """
    return QualityMetrics(entities=entities, relationships=relationships, schema_manager=schema_manager)


# Example usage
if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_entities = [
        EntityInstance(id="1", name="John Smith", type="Person"),
        EntityInstance(id="2", name="Google", type="Organization"),
        EntityInstance(id="3", name="New York", type="Location"),
    ]
    
    sample_relationships = [
        RelationshipInstance(source_id="1", target_id="2", type="works_at", confidence=0.9),
        RelationshipInstance(source_id="2", target_id="3", type="located_in", confidence=0.8),
    ]
    
    # Calculate metrics
    metrics_calc = create_quality_metrics(sample_entities, sample_relationships)
    
    completeness = metrics_calc.calculate_completeness_metrics()
    accuracy = metrics_calc.calculate_accuracy_metrics()
    connectivity = metrics_calc.calculate_connectivity_metrics()
    
    print(f"📊 Quality Metrics Results:")
    print(f"Completeness: {completeness.overall_completeness:.2f}")
    print(f"Accuracy: {accuracy.overall_accuracy:.2f}")
    print(f"Connectivity: {connectivity.connectivity_score:.2f}")
