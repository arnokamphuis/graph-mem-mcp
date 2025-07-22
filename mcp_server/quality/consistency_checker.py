#!/usr/bin/env python3
"""
Consistency Checker Module for Knowledge Graph Validation

This module provides comprehensive consistency checking across entities and
relationships in knowledge graphs, detecting logical inconsistencies,
temporal conflicts, and constraint violations.

Features:
- Cross-entity consistency validation
- Relationship consistency checking
- Temporal consistency analysis
- Schema constraint validation
- Logical consistency detection
- Circular dependency detection

Usage:
    from mcp_server.quality.consistency_checker import ConsistencyChecker
    
    # Initialize consistency checker
    checker = ConsistencyChecker(entities, relationships, schema_manager)
    
    # Run comprehensive consistency check
    violations = checker.check_all_consistency()
    
    # Check specific consistency types
    temporal_violations = checker.check_temporal_consistency()
    logical_violations = checker.check_logical_consistency()
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import datetime
import re

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
    
    class SchemaManager:
        def __init__(self):
            self.entity_types = set()
            self.relationship_types = set()

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of consistency violations"""
    TEMPORAL_CONFLICT = "temporal_conflict"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    SCHEMA_VIOLATION = "schema_violation"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    CARDINALITY_VIOLATION = "cardinality_violation"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    REQUIRED_RELATIONSHIP_MISSING = "required_relationship_missing"
    INVALID_RELATIONSHIP_COMBINATION = "invalid_relationship_combination"
    DATA_TYPE_INCONSISTENCY = "data_type_inconsistency"
    REFERENTIAL_INTEGRITY = "referential_integrity"


class ViolationSeverity(Enum):
    """Severity levels for consistency violations"""
    CRITICAL = "critical"  # Breaks fundamental graph integrity
    HIGH = "high"         # Significant logical problems
    MEDIUM = "medium"     # Moderate consistency issues
    LOW = "low"          # Minor inconsistencies
    WARNING = "warning"   # Potential issues to review


@dataclass
class ConsistencyViolation:
    """Represents a consistency violation found in the knowledge graph"""
    violation_type: ViolationType
    severity: ViolationSeverity
    description: str
    affected_entities: List[str] = field(default_factory=list)
    affected_relationships: List[str] = field(default_factory=list)
    rule_name: str = ""
    suggested_fix: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate violation data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ConsistencyRule:
    """Defines a consistency rule to be checked"""
    name: str
    description: str
    violation_type: ViolationType
    severity: ViolationSeverity
    check_function: str  # Name of the method to call
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class ConsistencyChecker:
    """
    Comprehensive Consistency Checker for Knowledge Graphs
    
    Validates consistency across multiple dimensions including temporal,
    logical, schema-based, and referential integrity constraints.
    """
    
    def __init__(self, entities: List[EntityInstance] = None,
                 relationships: List[RelationshipInstance] = None,
                 schema_manager: Optional[SchemaManager] = None):
        """
        Initialize consistency checker
        
        Args:
            entities: List of entities to check
            relationships: List of relationships to check
            schema_manager: Schema manager for validation rules
        """
        self.entities = entities or []
        self.relationships = relationships or []
        self.schema_manager = schema_manager
        
        # Build indexes for efficient checking
        self._build_indexes()
        
        # Initialize consistency rules
        self._initialize_consistency_rules()
        
        logger.info(f"⚖️ Consistency Checker initialized with {len(self.entities)} entities, {len(self.relationships)} relationships")
    
    def _build_indexes(self):
        """Build indexes for efficient consistency checking"""
        # Entity indexes
        self.entity_by_id = {}
        self.entities_by_type = defaultdict(list)
        self.entities_by_name = defaultdict(list)
        
        for entity in self.entities:
            entity_id = getattr(entity, 'id', str(entity))
            self.entity_by_id[entity_id] = entity
            
            entity_type = getattr(entity, 'type', 'unknown')
            self.entities_by_type[entity_type].append(entity)
            
            entity_name = getattr(entity, 'name', '')
            if entity_name:
                self.entities_by_name[entity_name.lower()].append(entity)
        
        # Relationship indexes
        self.relationships_by_type = defaultdict(list)
        self.relationships_by_source = defaultdict(list)
        self.relationships_by_target = defaultdict(list)
        self.relationship_pairs = defaultdict(list)
        
        for relationship in self.relationships:
            rel_id = getattr(relationship, 'id', str(relationship))
            rel_type = getattr(relationship, 'relation_type', 'unknown')
            source_id = getattr(relationship, 'source_entity_id', None)
            target_id = getattr(relationship, 'target_entity_id', None)
            
            self.relationships_by_type[rel_type].append(relationship)
            
            if source_id:
                self.relationships_by_source[source_id].append(relationship)
            if target_id:
                self.relationships_by_target[target_id].append(relationship)
            
            if source_id and target_id:
                pair_key = (source_id, target_id)
                self.relationship_pairs[pair_key].append(relationship)
    
    def _initialize_consistency_rules(self):
        """Initialize predefined consistency rules"""
        self.consistency_rules = [
            ConsistencyRule(
                name="temporal_order_check",
                description="Verify temporal relationships maintain proper chronological order",
                violation_type=ViolationType.TEMPORAL_CONFLICT,
                severity=ViolationSeverity.HIGH,
                check_function="check_temporal_consistency"
            ),
            ConsistencyRule(
                name="circular_dependency_check",
                description="Detect circular dependencies in hierarchical relationships",
                violation_type=ViolationType.CIRCULAR_DEPENDENCY,
                severity=ViolationSeverity.CRITICAL,
                check_function="check_circular_dependencies"
            ),
            ConsistencyRule(
                name="cardinality_constraint_check",
                description="Validate relationship cardinality constraints",
                violation_type=ViolationType.CARDINALITY_VIOLATION,
                severity=ViolationSeverity.MEDIUM,
                check_function="check_cardinality_constraints"
            ),
            ConsistencyRule(
                name="mutual_exclusion_check",
                description="Check for mutually exclusive relationships",
                violation_type=ViolationType.MUTUAL_EXCLUSION,
                severity=ViolationSeverity.HIGH,
                check_function="check_mutual_exclusion"
            ),
            ConsistencyRule(
                name="referential_integrity_check",
                description="Validate referential integrity of relationships",
                violation_type=ViolationType.REFERENTIAL_INTEGRITY,
                severity=ViolationSeverity.CRITICAL,
                check_function="check_referential_integrity"
            ),
            ConsistencyRule(
                name="data_type_consistency_check",
                description="Check consistency of data types across similar entities",
                violation_type=ViolationType.DATA_TYPE_INCONSISTENCY,
                severity=ViolationSeverity.MEDIUM,
                check_function="check_data_type_consistency"
            )
        ]
    
    def check_all_consistency(self) -> List[ConsistencyViolation]:
        """
        Run all consistency checks and return violations found
        
        Returns:
            List[ConsistencyViolation]: All consistency violations found
        """
        logger.info("🔍 Running comprehensive consistency check...")
        
        all_violations = []
        
        for rule in self.consistency_rules:
            if not rule.enabled:
                continue
            
            try:
                logger.debug(f"Running consistency rule: {rule.name}")
                check_method = getattr(self, rule.check_function)
                violations = check_method()
                all_violations.extend(violations)
                
            except Exception as e:
                logger.error(f"Error running consistency rule {rule.name}: {e}")
                # Create violation for the check failure
                violation = ConsistencyViolation(
                    violation_type=ViolationType.SCHEMA_VIOLATION,
                    severity=ViolationSeverity.WARNING,
                    description=f"Consistency check '{rule.name}' failed: {e}",
                    rule_name=rule.name
                )
                all_violations.append(violation)
        
        logger.info(f"✅ Consistency check completed - Found {len(all_violations)} violations")
        return all_violations
    
    def check_temporal_consistency(self) -> List[ConsistencyViolation]:
        """
        Check temporal consistency across entities and relationships
        
        Returns:
            List[ConsistencyViolation]: Temporal consistency violations
        """
        violations = []
        
        # Check temporal order in events
        violations.extend(self._check_event_temporal_order())
        
        # Check birth/death dates for persons
        violations.extend(self._check_person_temporal_consistency())
        
        # Check organization founding/dissolution dates
        violations.extend(self._check_organization_temporal_consistency())
        
        # Check temporal relationships (before/after)
        violations.extend(self._check_temporal_relationships())
        
        return violations
    
    def check_circular_dependencies(self) -> List[ConsistencyViolation]:
        """
        Check for circular dependencies in hierarchical relationships
        
        Returns:
            List[ConsistencyViolation]: Circular dependency violations
        """
        violations = []
        
        # Define hierarchical relationship types
        hierarchical_types = [
            'part_of', 'belongs_to', 'contains', 'manages', 'supervises',
            'parent_of', 'child_of', 'owns', 'member_of'
        ]
        
        for rel_type in hierarchical_types:
            cycles = self._detect_cycles_in_relationship_type(rel_type)
            for cycle in cycles:
                violation = ConsistencyViolation(
                    violation_type=ViolationType.CIRCULAR_DEPENDENCY,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Circular dependency detected in '{rel_type}' relationships: {' -> '.join(cycle)}",
                    affected_entities=cycle,
                    rule_name="circular_dependency_check",
                    suggested_fix="Break the circular dependency by removing or redirecting one of the relationships",
                    confidence=1.0
                )
                violations.append(violation)
        
        return violations
    
    def check_cardinality_constraints(self) -> List[ConsistencyViolation]:
        """
        Check relationship cardinality constraints
        
        Returns:
            List[ConsistencyViolation]: Cardinality constraint violations
        """
        violations = []
        
        # Define cardinality constraints (relationship_type: (min, max))
        cardinality_constraints = {
            'married_to': (0, 1),  # Person can be married to at most 1 person
            'born_in': (1, 1),     # Person must be born in exactly 1 location
            'ceo_of': (0, 1),      # Person can be CEO of at most 1 organization
            'headquarters_in': (0, 1),  # Organization can have headquarters in at most 1 location
        }
        
        for rel_type, (min_count, max_count) in cardinality_constraints.items():
            violations.extend(self._check_cardinality_for_type(rel_type, min_count, max_count))
        
        return violations
    
    def check_mutual_exclusion(self) -> List[ConsistencyViolation]:
        """
        Check for mutually exclusive relationships
        
        Returns:
            List[ConsistencyViolation]: Mutual exclusion violations
        """
        violations = []
        
        # Define mutually exclusive relationship pairs
        mutual_exclusions = [
            ('employed_by', 'unemployed'),
            ('alive', 'deceased'),
            ('active', 'inactive'),
            ('public_company', 'private_company'),
        ]
        
        for rel_type1, rel_type2 in mutual_exclusions:
            violations.extend(self._check_mutual_exclusion_pair(rel_type1, rel_type2))
        
        return violations
    
    def check_referential_integrity(self) -> List[ConsistencyViolation]:
        """
        Check referential integrity of relationships
        
        Returns:
            List[ConsistencyViolation]: Referential integrity violations
        """
        violations = []
        
        for relationship in self.relationships:
            source_id = getattr(relationship, 'source_entity_id', None)
            target_id = getattr(relationship, 'target_entity_id', None)
            rel_id = getattr(relationship, 'id', str(relationship))
            rel_type = getattr(relationship, 'relation_type', 'unknown')
            
            # Check if source entity exists
            if source_id and source_id not in self.entity_by_id:
                violation = ConsistencyViolation(
                    violation_type=ViolationType.REFERENTIAL_INTEGRITY,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Relationship '{rel_id}' references non-existent source entity '{source_id}'",
                    affected_relationships=[rel_id],
                    rule_name="referential_integrity_check",
                    suggested_fix=f"Remove relationship or create missing entity '{source_id}'",
                    confidence=1.0
                )
                violations.append(violation)
            
            # Check if target entity exists
            if target_id and target_id not in self.entity_by_id:
                violation = ConsistencyViolation(
                    violation_type=ViolationType.REFERENTIAL_INTEGRITY,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Relationship '{rel_id}' references non-existent target entity '{target_id}'",
                    affected_relationships=[rel_id],
                    rule_name="referential_integrity_check",
                    suggested_fix=f"Remove relationship or create missing entity '{target_id}'",
                    confidence=1.0
                )
                violations.append(violation)
        
        return violations
    
    def check_data_type_consistency(self) -> List[ConsistencyViolation]:
        """
        Check consistency of data types across similar entities
        
        Returns:
            List[ConsistencyViolation]: Data type consistency violations
        """
        violations = []
        
        # Check date format consistency
        violations.extend(self._check_date_format_consistency())
        
        # Check numeric value consistency
        violations.extend(self._check_numeric_consistency())
        
        # Check string format consistency
        violations.extend(self._check_string_format_consistency())
        
        return violations
    
    def check_logical_consistency(self) -> List[ConsistencyViolation]:
        """
        Check logical consistency across relationships
        
        Returns:
            List[ConsistencyViolation]: Logical consistency violations
        """
        violations = []
        
        # Check symmetric relationship consistency
        violations.extend(self._check_symmetric_relationships())
        
        # Check transitive relationship consistency
        violations.extend(self._check_transitive_relationships())
        
        # Check inverse relationship consistency
        violations.extend(self._check_inverse_relationships())
        
        return violations
    
    def _check_event_temporal_order(self) -> List[ConsistencyViolation]:
        """Check temporal order consistency for events"""
        violations = []
        
        events = self.entities_by_type.get('Event', [])
        
        for event in events:
            start_date = getattr(event, 'start_date', None)
            end_date = getattr(event, 'end_date', None)
            
            if start_date and end_date:
                try:
                    start_dt = self._parse_date(start_date)
                    end_dt = self._parse_date(end_date)
                    
                    if start_dt and end_dt and start_dt > end_dt:
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.TEMPORAL_CONFLICT,
                            severity=ViolationSeverity.HIGH,
                            description=f"Event '{getattr(event, 'name', event.id)}' has start date after end date",
                            affected_entities=[getattr(event, 'id', str(event))],
                            rule_name="temporal_order_check",
                            suggested_fix="Correct the start or end date",
                            confidence=1.0
                        )
                        violations.append(violation)
                
                except Exception as e:
                    logger.debug(f"Error parsing dates for event {event}: {e}")
        
        return violations
    
    def _check_person_temporal_consistency(self) -> List[ConsistencyViolation]:
        """Check temporal consistency for persons (birth/death dates)"""
        violations = []
        
        persons = self.entities_by_type.get('Person', [])
        
        for person in persons:
            birth_date = getattr(person, 'birth_date', None)
            death_date = getattr(person, 'death_date', None)
            
            if birth_date and death_date:
                try:
                    birth_dt = self._parse_date(birth_date)
                    death_dt = self._parse_date(death_date)
                    
                    if birth_dt and death_dt and birth_dt > death_dt:
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.TEMPORAL_CONFLICT,
                            severity=ViolationSeverity.HIGH,
                            description=f"Person '{getattr(person, 'name', person.id)}' has birth date after death date",
                            affected_entities=[getattr(person, 'id', str(person))],
                            rule_name="temporal_order_check",
                            suggested_fix="Correct the birth or death date",
                            confidence=1.0
                        )
                        violations.append(violation)
                
                except Exception as e:
                    logger.debug(f"Error parsing dates for person {person}: {e}")
        
        return violations
    
    def _check_organization_temporal_consistency(self) -> List[ConsistencyViolation]:
        """Check temporal consistency for organizations"""
        violations = []
        
        organizations = self.entities_by_type.get('Organization', [])
        
        for org in organizations:
            founded_date = getattr(org, 'founded_date', None)
            dissolved_date = getattr(org, 'dissolved_date', None)
            
            if founded_date and dissolved_date:
                try:
                    founded_dt = self._parse_date(founded_date)
                    dissolved_dt = self._parse_date(dissolved_date)
                    
                    if founded_dt and dissolved_dt and founded_dt > dissolved_dt:
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.TEMPORAL_CONFLICT,
                            severity=ViolationSeverity.HIGH,
                            description=f"Organization '{getattr(org, 'name', org.id)}' has founding date after dissolution date",
                            affected_entities=[getattr(org, 'id', str(org))],
                            rule_name="temporal_order_check",
                            suggested_fix="Correct the founding or dissolution date",
                            confidence=1.0
                        )
                        violations.append(violation)
                
                except Exception as e:
                    logger.debug(f"Error parsing dates for organization {org}: {e}")
        
        return violations
    
    def _check_temporal_relationships(self) -> List[ConsistencyViolation]:
        """Check consistency of temporal relationships (before/after)"""
        violations = []
        
        # Check 'before' relationships
        before_rels = self.relationships_by_type.get('before', [])
        
        for rel in before_rels:
            source_id = getattr(rel, 'source_id', None)
            target_id = getattr(rel, 'target_id', None)
            
            if source_id and target_id:
                source_entity = self.entity_by_id.get(source_id)
                target_entity = self.entity_by_id.get(target_id)
                
                if source_entity and target_entity:
                    source_date = self._get_entity_date(source_entity)
                    target_date = self._get_entity_date(target_entity)
                    
                    if source_date and target_date and source_date >= target_date:
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.TEMPORAL_CONFLICT,
                            severity=ViolationSeverity.MEDIUM,
                            description=f"'Before' relationship violated: {source_id} is not before {target_id}",
                            affected_entities=[source_id, target_id],
                            affected_relationships=[getattr(rel, 'id', str(rel))],
                            rule_name="temporal_order_check",
                            suggested_fix="Review and correct temporal relationship or entity dates",
                            confidence=0.8
                        )
                        violations.append(violation)
        
        return violations
    
    def _detect_cycles_in_relationship_type(self, rel_type: str) -> List[List[str]]:
        """Detect cycles in a specific relationship type"""
        relationships = self.relationships_by_type.get(rel_type, [])
        
        # Build directed graph for this relationship type
        graph = defaultdict(list)
        for rel in relationships:
            source_id = getattr(rel, 'source_id', None)
            target_id = getattr(rel, 'target_id', None)
            if source_id and target_id:
                graph[source_id].append(target_id)
        
        # Use DFS to detect cycles
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            if node in rec_stack:
                # Found cycle - extract it
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        # Check all nodes
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def _check_cardinality_for_type(self, rel_type: str, min_count: int, max_count: int) -> List[ConsistencyViolation]:
        """Check cardinality constraints for a specific relationship type"""
        violations = []
        
        # Count relationships by source entity
        source_counts = defaultdict(int)
        for rel in self.relationships_by_type.get(rel_type, []):
            source_id = getattr(rel, 'source_id', None)
            if source_id:
                source_counts[source_id] += 1
        
        # Check violations
        for source_id, count in source_counts.items():
            if count < min_count:
                violation = ConsistencyViolation(
                    violation_type=ViolationType.CARDINALITY_VIOLATION,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Entity '{source_id}' has too few '{rel_type}' relationships ({count} < {min_count})",
                    affected_entities=[source_id],
                    rule_name="cardinality_constraint_check",
                    suggested_fix=f"Add {min_count - count} more '{rel_type}' relationships",
                    confidence=1.0
                )
                violations.append(violation)
            
            elif count > max_count:
                violation = ConsistencyViolation(
                    violation_type=ViolationType.CARDINALITY_VIOLATION,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Entity '{source_id}' has too many '{rel_type}' relationships ({count} > {max_count})",
                    affected_entities=[source_id],
                    rule_name="cardinality_constraint_check",
                    suggested_fix=f"Remove {count - max_count} '{rel_type}' relationships",
                    confidence=1.0
                )
                violations.append(violation)
        
        return violations
    
    def _check_mutual_exclusion_pair(self, rel_type1: str, rel_type2: str) -> List[ConsistencyViolation]:
        """Check mutual exclusion between two relationship types"""
        violations = []
        
        # Get entities that have both relationship types
        entities_with_rel1 = set()
        entities_with_rel2 = set()
        
        for rel in self.relationships_by_type.get(rel_type1, []):
            source_id = getattr(rel, 'source_id', None)
            if source_id:
                entities_with_rel1.add(source_id)
        
        for rel in self.relationships_by_type.get(rel_type2, []):
            source_id = getattr(rel, 'source_id', None)
            if source_id:
                entities_with_rel2.add(source_id)
        
        # Find entities with both (violation)
        conflicting_entities = entities_with_rel1.intersection(entities_with_rel2)
        
        for entity_id in conflicting_entities:
            violation = ConsistencyViolation(
                violation_type=ViolationType.MUTUAL_EXCLUSION,
                severity=ViolationSeverity.HIGH,
                description=f"Entity '{entity_id}' has mutually exclusive relationships: '{rel_type1}' and '{rel_type2}'",
                affected_entities=[entity_id],
                rule_name="mutual_exclusion_check",
                suggested_fix=f"Remove either '{rel_type1}' or '{rel_type2}' relationship",
                confidence=1.0
            )
            violations.append(violation)
        
        return violations
    
    def _check_date_format_consistency(self) -> List[ConsistencyViolation]:
        """Check consistency of date formats across entities"""
        violations = []
        
        date_properties = ['birth_date', 'death_date', 'founded_date', 'dissolved_date', 'start_date', 'end_date', 'date']
        
        for prop in date_properties:
            date_formats = defaultdict(list)
            
            for entity in self.entities:
                date_value = getattr(entity, prop, None)
                if date_value:
                    date_format = self._detect_date_format(str(date_value))
                    date_formats[date_format].append(getattr(entity, 'id', str(entity)))
            
            # Check for multiple formats (inconsistency)
            if len(date_formats) > 1:
                format_counts = {fmt: len(entities) for fmt, entities in date_formats.items()}
                dominant_format = max(format_counts, key=format_counts.get)
                
                for fmt, entities in date_formats.items():
                    if fmt != dominant_format and fmt != 'unknown':
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.DATA_TYPE_INCONSISTENCY,
                            severity=ViolationSeverity.LOW,
                            description=f"Inconsistent date format for property '{prop}': {fmt} (expected {dominant_format})",
                            affected_entities=entities,
                            rule_name="data_type_consistency_check",
                            suggested_fix=f"Convert dates to {dominant_format} format",
                            confidence=0.7
                        )
                        violations.append(violation)
        
        return violations
    
    def _check_numeric_consistency(self) -> List[ConsistencyViolation]:
        """Check consistency of numeric values"""
        violations = []
        
        numeric_properties = ['age', 'year', 'count', 'quantity', 'amount']
        
        for prop in numeric_properties:
            for entity in self.entities:
                value = getattr(entity, prop, None)
                if value is not None:
                    try:
                        numeric_value = float(value)
                        # Check for unreasonable values
                        if prop == 'age' and (numeric_value < 0 or numeric_value > 150):
                            violation = ConsistencyViolation(
                                violation_type=ViolationType.DATA_TYPE_INCONSISTENCY,
                                severity=ViolationSeverity.MEDIUM,
                                description=f"Unreasonable age value: {numeric_value} for entity '{getattr(entity, 'id', str(entity))}'",
                                affected_entities=[getattr(entity, 'id', str(entity))],
                                rule_name="data_type_consistency_check",
                                suggested_fix="Review and correct age value",
                                confidence=0.9
                            )
                            violations.append(violation)
                    
                    except (ValueError, TypeError):
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.DATA_TYPE_INCONSISTENCY,
                            severity=ViolationSeverity.MEDIUM,
                            description=f"Non-numeric value '{value}' for numeric property '{prop}' in entity '{getattr(entity, 'id', str(entity))}'",
                            affected_entities=[getattr(entity, 'id', str(entity))],
                            rule_name="data_type_consistency_check",
                            suggested_fix="Convert to numeric value or change property type",
                            confidence=1.0
                        )
                        violations.append(violation)
        
        return violations
    
    def _check_string_format_consistency(self) -> List[ConsistencyViolation]:
        """Check consistency of string formats (emails, URLs, etc.)"""
        violations = []
        
        format_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s]+$',
            'phone': r'^[\+]?[1-9][\d]{0,15}$'
        }
        
        for entity in self.entities:
            for prop_name in ['email', 'website', 'url', 'phone']:
                value = getattr(entity, prop_name, None)
                if value:
                    expected_pattern = format_patterns.get(prop_name)
                    if expected_pattern and not re.match(expected_pattern, str(value)):
                        violation = ConsistencyViolation(
                            violation_type=ViolationType.DATA_TYPE_INCONSISTENCY,
                            severity=ViolationSeverity.LOW,
                            description=f"Invalid format for {prop_name}: '{value}' in entity '{getattr(entity, 'id', str(entity))}'",
                            affected_entities=[getattr(entity, 'id', str(entity))],
                            rule_name="data_type_consistency_check",
                            suggested_fix=f"Correct {prop_name} format",
                            confidence=0.8
                        )
                        violations.append(violation)
        
        return violations
    
    def _check_symmetric_relationships(self) -> List[ConsistencyViolation]:
        """Check symmetry of symmetric relationships"""
        violations = []
        
        # Define symmetric relationship types
        symmetric_types = ['married_to', 'partners_with', 'allied_with', 'connected_to']
        
        for rel_type in symmetric_types:
            relationships = self.relationships_by_type.get(rel_type, [])
            
            # Build bidirectional map
            forward_rels = {}
            for rel in relationships:
                source_id = getattr(rel, 'source_id', None)
                target_id = getattr(rel, 'target_id', None)
                if source_id and target_id:
                    forward_rels[(source_id, target_id)] = rel
            
            # Check for missing reverse relationships
            for (source_id, target_id), rel in forward_rels.items():
                if (target_id, source_id) not in forward_rels:
                    violation = ConsistencyViolation(
                        violation_type=ViolationType.LOGICAL_INCONSISTENCY,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Symmetric relationship '{rel_type}' missing reverse: {target_id} -> {source_id}",
                        affected_entities=[source_id, target_id],
                        affected_relationships=[getattr(rel, 'id', str(rel))],
                        rule_name="logical_consistency_check",
                        suggested_fix=f"Add reverse '{rel_type}' relationship",
                        confidence=0.9
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_transitive_relationships(self) -> List[ConsistencyViolation]:
        """Check transitivity of transitive relationships"""
        violations = []
        
        # Define transitive relationship types
        transitive_types = ['part_of', 'located_in', 'subclass_of']
        
        for rel_type in transitive_types:
            # This is a simplified check - full transitivity checking is complex
            # We check for obvious missing transitivity
            relationships = self.relationships_by_type.get(rel_type, [])
            
            # Build relationship map
            rel_map = defaultdict(set)
            for rel in relationships:
                source_id = getattr(rel, 'source_id', None)
                target_id = getattr(rel, 'target_id', None)
                if source_id and target_id:
                    rel_map[source_id].add(target_id)
            
            # Check for obvious transitivity violations (simplified)
            # A -> B and B -> C should imply A -> C
            for a in rel_map:
                for b in rel_map[a]:
                    for c in rel_map.get(b, []):
                        if c not in rel_map[a] and a != c:
                            # Potential missing transitive relationship
                            violation = ConsistencyViolation(
                                violation_type=ViolationType.LOGICAL_INCONSISTENCY,
                                severity=ViolationSeverity.LOW,
                                description=f"Potential missing transitive '{rel_type}' relationship: {a} -> {c}",
                                affected_entities=[a, b, c],
                                rule_name="logical_consistency_check",
                                suggested_fix=f"Consider adding transitive '{rel_type}' relationship",
                                confidence=0.6
                            )
                            violations.append(violation)
        
        return violations
    
    def _check_inverse_relationships(self) -> List[ConsistencyViolation]:
        """Check consistency of inverse relationships"""
        violations = []
        
        # Define inverse relationship pairs
        inverse_pairs = [
            ('parent_of', 'child_of'),
            ('owns', 'owned_by'),
            ('manages', 'managed_by'),
            ('teaches', 'taught_by'),
            ('employs', 'employed_by')
        ]
        
        for rel_type1, rel_type2 in inverse_pairs:
            relationships1 = self.relationships_by_type.get(rel_type1, [])
            relationships2 = self.relationships_by_type.get(rel_type2, [])
            
            # Build relationship maps
            forward_map = set()
            reverse_map = set()
            
            for rel in relationships1:
                source_id = getattr(rel, 'source_id', None)
                target_id = getattr(rel, 'target_id', None)
                if source_id and target_id:
                    forward_map.add((source_id, target_id))
            
            for rel in relationships2:
                source_id = getattr(rel, 'source_id', None)
                target_id = getattr(rel, 'target_id', None)
                if source_id and target_id:
                    reverse_map.add((target_id, source_id))  # Note the reversal
            
            # Check for missing inverse relationships
            for (source_id, target_id) in forward_map:
                if (source_id, target_id) not in reverse_map:
                    violation = ConsistencyViolation(
                        violation_type=ViolationType.LOGICAL_INCONSISTENCY,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Missing inverse relationship: '{rel_type1}' {source_id} -> {target_id} should have inverse '{rel_type2}' {target_id} -> {source_id}",
                        affected_entities=[source_id, target_id],
                        rule_name="logical_consistency_check",
                        suggested_fix=f"Add inverse '{rel_type2}' relationship",
                        confidence=0.8
                    )
                    violations.append(violation)
        
        return violations
    
    def _parse_date(self, date_str: str) -> Optional[datetime.datetime]:
        """Parse date string into datetime object"""
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y',
            '%B %d, %Y',
            '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        
        return None
    
    def _get_entity_date(self, entity: EntityInstance) -> Optional[datetime.datetime]:
        """Get the primary date for an entity"""
        date_props = ['date', 'start_date', 'birth_date', 'founded_date', 'created_date']
        
        for prop in date_props:
            date_value = getattr(entity, prop, None)
            if date_value:
                return self._parse_date(str(date_value))
        
        return None
    
    def _detect_date_format(self, date_str: str) -> str:
        """Detect the format of a date string"""
        formats = {
            r'^\d{4}-\d{2}-\d{2}$': 'YYYY-MM-DD',
            r'^\d{4}/\d{2}/\d{2}$': 'YYYY/MM/DD',
            r'^\d{2}/\d{2}/\d{4}$': 'MM/DD/YYYY',
            r'^\d{2}-\d{2}-\d{4}$': 'MM-DD-YYYY',
            r'^\d{4}$': 'YYYY',
            r'^[A-Za-z]+ \d{1,2}, \d{4}$': 'Month DD, YYYY'
        }
        
        for pattern, format_name in formats.items():
            if re.match(pattern, date_str):
                return format_name
        
        return 'unknown'


def create_consistency_checker(entities: List[EntityInstance] = None,
                             relationships: List[RelationshipInstance] = None,
                             schema_manager: Optional[SchemaManager] = None) -> ConsistencyChecker:
    """
    Factory function to create a ConsistencyChecker instance
    
    Args:
        entities: List of entities to check
        relationships: List of relationships to check
        schema_manager: Schema manager for validation rules
        
    Returns:
        ConsistencyChecker: Configured consistency checker
    """
    return ConsistencyChecker(entities=entities, relationships=relationships, schema_manager=schema_manager)


# Example usage
if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with potential consistency issues
    sample_entities = [
        EntityInstance(id="1", name="John Smith", type="Person", birth_date="1990-01-01", death_date="1985-01-01"),  # Inconsistent dates
        EntityInstance(id="2", name="Google", type="Organization", founded_date="1998-09-04"),
        EntityInstance(id="3", name="Alice Johnson", type="Person", age="not_a_number"),  # Invalid data type
    ]
    
    sample_relationships = [
        RelationshipInstance(source_id="1", target_id="2", type="works_at"),
        RelationshipInstance(source_id="1", target_id="4", type="married_to"),  # References non-existent entity
        RelationshipInstance(source_id="2", target_id="3", type="part_of"),
        RelationshipInstance(source_id="3", target_id="2", type="part_of"),  # Could create circular dependency
    ]
    
    # Run consistency checks
    checker = create_consistency_checker(sample_entities, sample_relationships)
    violations = checker.check_all_consistency()
    
    print(f"⚖️ Consistency Check Results:")
    print(f"Found {len(violations)} violations:")
    
    for violation in violations:
        print(f"- {violation.severity.value.upper()}: {violation.description}")
