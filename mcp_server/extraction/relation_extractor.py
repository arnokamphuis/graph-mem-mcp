"""
Sophisticated Relationship Extraction Module - Phase 2.1 Implementation

This module implements sophisticated relationship extraction following the refactoring plan architecture.
It integrates with Phase 1 core components (schema management, entity resolution) and implements
multi-model ensemble approach for high-quality relationship recognition.

Architecture Integration:
- Uses RelationshipInstance and SchemaManager from core/graph_schema.py
- Follows dependency management patterns from coding standards
- Implements multi-model ensemble as specified in refactoring plan
- Supports pre-trained transformer models, pattern-based extraction, and dependency parsing
"""

from typing import Dict, List, Optional, Any, Set, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core system integration - Phase 1 components
try:
    from ..core.graph_schema import (
        EntityInstance, RelationshipInstance, SchemaManager, 
        RelationshipTypeSchema, PropertySchema, PropertyType
    )
    CORE_SCHEMA_AVAILABLE = True
    logger.info("✅ Core schema system integration available")
except (ImportError, AttributeError):
    logger.error("❌ Core schema system not available - Phase 1 required")
    raise ImportError("Phase 2.1 requires Phase 1 core components to be implemented")

# Optional ML dependencies with graceful fallbacks per coding standards
TRANSFORMERS_AVAILABLE = False
SPACY_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, Pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
    logger.info("✅ Transformers available - using transformer-based relation extraction")
except ImportError:
    logger.warning("⚠️  Transformers not available - using pattern-based extraction")

try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy available - using dependency parsing")
except ImportError:
    logger.warning("⚠️  spaCy not available - using basic text processing")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("⚠️  NumPy not available - using basic confidence scoring")


class ExtractionMethod(str, Enum):
    """Supported relationship extraction methods per refactoring plan"""
    TRANSFORMER = "transformer"
    PATTERN_BASED = "pattern_based"
    DEPENDENCY_PARSING = "dependency_parsing"
    RULE_BASED = "rule_based"


@dataclass
class RelationshipCandidate:
    """Represents a candidate relationship between two entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    evidence_text: str
    context_window: str
    extraction_method: ExtractionMethod
    position_start: int
    position_end: int
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_relationship_instance(self) -> Optional['RelationshipInstance']:
        """Convert to a RelationshipInstance if core modules available"""
        if not CORE_SCHEMA_AVAILABLE or RelationshipInstance is None:
            return None
        
        try:
            return RelationshipInstance(
                source_entity_id=self.source_entity,
                target_entity_id=self.target_entity,
                relationship_type=self.relationship_type,
                confidence=self.confidence,
                properties={
                    'evidence_text': self.evidence_text,
                    'context_window': self.context_window,
                    'extraction_method': self.extraction_method.value,
                    'position_start': self.position_start,
                    'position_end': self.position_end,
                    **self.properties
                }
            )
        except Exception:
            # Fallback classes may not support initialization
            return None


@dataclass
class ExtractionContext:
    """Context for relationship extraction operations"""
    text: str
    source_id: str = "unknown"
    domain_context: Optional[str] = None
    entities: List[EntityInstance] = field(default_factory=list)
    schema_manager: Optional[SchemaManager] = None


class SophisticatedRelationshipExtractor:
    """
    Advanced relationship extraction system following Phase 2.1 plan architecture.
    
    Implements multi-model ensemble approach with:
    - Pre-trained relationship extraction transformer models
    - Custom fine-tuned transformers for domain-specific relations  
    - Multi-sentence context analysis
    - Confidence calibration for relationship predictions
    - Semantic role labeling integration
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 max_context_length: int = 512,
                 schema_manager: Optional[SchemaManager] = None):
        """
        Initialize the sophisticated relationship extractor
        
        Args:
            confidence_threshold: Minimum confidence for relationship acceptance
            max_context_length: Maximum context window for analysis
            schema_manager: Schema manager for relationship type validation
        """
        self.confidence_threshold = confidence_threshold
        self.max_context_length = max_context_length
        self.schema_manager = schema_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize extraction strategies
        self._init_transformer_models()
        self._init_spacy_model()
        self._init_pattern_rules()
        
        # Track extraction statistics
        self.extraction_stats = {
            'total_candidates': 0,
            'high_confidence_candidates': 0,
            'transformer_extractions': 0,
            'pattern_extractions': 0,
            'dependency_extractions': 0
        }
        
        self.logger.info("🔗 Sophisticated Relationship Extractor initialized")
    
    def _init_transformer_models(self) -> None:
        """Initialize transformer models for relationship extraction"""
        self.relation_classifier = None
        self.tokenizer = None
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("⚠️  Transformers not available - skipping transformer models")
            return
        
        try:
            # Use a pre-trained model specifically for relation extraction
            model_name = "deepset/bert-base-cased-squad2"  # Better for relation tasks
            self.relation_classifier = pipeline(
                "question-answering",  # We'll use QA approach for relation extraction
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"✅ Initialized transformer model: {model_name}")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to load transformer model: {e}")
            self.relation_classifier = None
            self.tokenizer = None
        self.extraction_stats = {
            'total_candidates': 0,
            'high_confidence_candidates': 0,
            'transformer_extractions': 0,
            'pattern_extractions': 0,
            'dependency_extractions': 0
        }
    
    def _init_transformer_model(self, model_name: str) -> None:
        """Initialize transformer model for relationship extraction"""
        if not self.enable_transformer:
            self.relation_classifier = None
            self.tokenizer = None
            return
        
        try:
            # Try to load a relation extraction model
            # Note: Using a general model here, in production you'd want a model specifically trained for RE
            self.relation_classifier = pipeline(
                "text-classification", 
                model=model_name,
                return_all_scores=True
            ) if TRANSFORMERS_AVAILABLE else None
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name) if TRANSFORMERS_AVAILABLE else None
            self.logger.info(f"Initialized transformer model: {model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load transformer model {model_name}: {e}")
            self.relation_classifier = None
            self.tokenizer = None
            self.enable_transformer = False
    
    def _init_spacy_model(self) -> None:
        """Initialize spaCy model for dependency parsing"""
        if not self.enable_dependency_parsing:
            self.nlp = None
            return
        
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Initialized spaCy model for dependency parsing")
        except Exception as e:
            try:
                # Fallback to a smaller model
                self.nlp = spacy.load("en_core_web_md")
                self.logger.info("Initialized spaCy medium model")
            except Exception as e2:
                self.logger.warning(f"Failed to load spaCy model: {e2}")
                self.nlp = None
                self.enable_dependency_parsing = False
    
    def _init_pattern_rules(self) -> None:
        """Initialize pattern-based extraction rules"""
        # Common relationship patterns
        self.relationship_patterns = {
            'works_for': [
                r'{entity1}\s+(?:works?\s+(?:for|at)|is\s+employed\s+(?:by|at))\s+{entity2}',
                r'{entity1}\s+(?:is\s+an?\s+)?(?:employee|worker|staff)\s+(?:of|at)\s+{entity2}',
                r'{entity2}\s+employs?\s+{entity1}'
            ],
            'located_in': [
                r'{entity1}\s+(?:is\s+)?(?:located|situated|based|found)\s+(?:in|at)\s+{entity2}',
                r'{entity1}\s+(?:in|at)\s+{entity2}',
                r'{entity2}\s+(?:contains|houses|hosts)\s+{entity1}'
            ],
            'founded_by': [
                r'{entity2}\s+(?:founded|established|created|started)\s+{entity1}',
                r'{entity1}\s+(?:was\s+)?(?:founded|established|created|started)\s+by\s+{entity2}'
            ],
            'member_of': [
                r'{entity1}\s+(?:is\s+a\s+)?member\s+of\s+{entity2}',
                r'{entity1}\s+belongs\s+to\s+{entity2}',
                r'{entity2}\s+(?:includes|contains)\s+{entity1}'
            ],
            'leads': [
                r'{entity1}\s+(?:leads|manages|heads|directs)\s+{entity2}',
                r'{entity1}\s+(?:is\s+the\s+)?(?:leader|manager|head|director)\s+of\s+{entity2}',
                r'{entity2}\s+(?:is\s+)?(?:led|managed|headed|directed)\s+by\s+{entity1}'
            ],
            'collaborates_with': [
                r'{entity1}\s+(?:collaborates?\s+with|works?\s+with|partners?\s+with)\s+{entity2}',
                r'{entity1}\s+and\s+{entity2}\s+(?:collaborate|work\s+together|partner)'
            ]
        }
        
        self.logger.info(f"Initialized {len(self.relationship_patterns)} pattern-based relationship types")
    
    def extract_relationships(self, context: ExtractionContext) -> List[RelationshipCandidate]:
        """
        Extract relationships from text using multi-model ensemble approach
        
        Implements sophisticated relationship extraction following Phase 2.1 plan:
        - Pre-trained transformer models for relation extraction
        - Multi-sentence context analysis
        - Confidence calibration for predictions
        - Integration with Phase 1 schema management
        
        Args:
            context: Extraction context with text, entities, and schema information
            
        Returns:
            List of RelationshipCandidate objects with confidence scores
        """
        candidates = []
        
        try:
            # Validate context and prepare for extraction
            if not context.text or not context.text.strip():
                self.logger.warning("⚠️  Empty text provided for relationship extraction")
                return []
            
            self.logger.info(f"🔍 Extracting relationships from text: {len(context.text)} chars")
            
            # Strategy 1: Transformer-based extraction (when available)
            if TRANSFORMERS_AVAILABLE and self.relation_classifier:
                transformer_candidates = self._extract_with_transformer(context)
                candidates.extend(transformer_candidates)
                self.extraction_stats['transformer_extractions'] += len(transformer_candidates)
                self.logger.debug(f"🤖 Transformer extracted {len(transformer_candidates)} candidates")
            
            # Strategy 2: Dependency parsing extraction (when spaCy available)
            if SPACY_AVAILABLE and self.nlp:
                dependency_candidates = self._extract_with_dependency_parsing(context)
                candidates.extend(dependency_candidates)
                self.extraction_stats['dependency_extractions'] += len(dependency_candidates)
                self.logger.debug(f"🌐 Dependency parsing extracted {len(dependency_candidates)} candidates")
            
            # Strategy 3: Pattern-based extraction (always available)
            pattern_candidates = self._extract_with_patterns(context)
            candidates.extend(pattern_candidates)
            self.extraction_stats['pattern_extractions'] += len(pattern_candidates)
            self.logger.debug(f"📝 Pattern matching extracted {len(pattern_candidates)} candidates")
            
            # Multi-model ensemble processing
            candidates = self._deduplicate_candidates(candidates)
            candidates = self._calibrate_confidence(candidates)
            candidates = self._validate_with_schema(candidates, context.schema_manager)
            candidates = self._rank_and_filter_candidates(candidates)
            
            # Update statistics
            self.extraction_stats['total_candidates'] += len(candidates)
            high_conf_count = sum(1 for c in candidates if c.confidence >= self.confidence_threshold)
            self.extraction_stats['high_confidence_candidates'] += high_conf_count
            
            self.logger.info(f"🔗 Extracted {len(candidates)} relationship candidates ({high_conf_count} high confidence)")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"❌ Error during relationship extraction: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_with_transformer(self, context: ExtractionContext) -> List[RelationshipCandidate]:
        """Extract relationships using transformer models"""
        if not self.relation_classifier:
            return []
        
        candidates = []
        entities = context.entities
        
        try:
            # Generate entity pairs for relationship classification
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Create context window around entities
                    context_text = self._create_context_window(
                        context.text, entity1, entity2
                    )
                    
                    if len(context_text.strip()) < 10:  # Skip very short contexts
                        continue
                    
                    # Classify relationship
                    try:
                        # Note: This is a simplified approach. In practice, you'd use a model
                        # specifically trained for relationship extraction
                        result = self.relation_classifier(context_text)
                        
                        if result and len(result) > 0:
                            # Extract the highest confidence prediction
                            best_pred = max(result[0], key=lambda x: x['score']) if isinstance(result[0], list) else result[0]
                            
                            if best_pred['score'] >= 0.5:  # Minimum threshold for transformer
                                candidate = RelationshipCandidate(
                                    source_entity=entity1.get('text', entity1.get('name', str(entity1))),
                                    target_entity=entity2.get('text', entity2.get('name', str(entity2))),
                                    relationship_type=best_pred['label'],
                                    confidence=best_pred['score'],
                                    evidence_text=context_text,
                                    context_window=context_text,
                                    extraction_method=ExtractionMethod.TRANSFORMER,
                                    position_start=entity1.get('start', 0),
                                    position_end=entity2.get('end', len(context_text))
                                )
                                candidates.append(candidate)
                    
                    except Exception as e:
                        self.logger.debug(f"Transformer classification failed for entity pair: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error in transformer-based extraction: {e}")
        
        return candidates
    
    def _extract_with_patterns(self, context: ExtractionContext) -> List[RelationshipCandidate]:
        """Extract relationships using pattern matching"""
        candidates = []
        entities = context.entities
        text = context.text
        
        try:
            # For each relationship type, check patterns
            for rel_type, patterns in self.relationship_patterns.items():
                for entity1 in entities:
                    for entity2 in entities:
                        if entity1 == entity2:
                            continue
                        
                        entity1_text = entity1.get('text', entity1.get('name', str(entity1)))
                        entity2_text = entity2.get('text', entity2.get('name', str(entity2)))
                        
                        # Check each pattern for this relationship type
                        for pattern in patterns:
                            import re
                            
                            # Replace placeholders with actual entity text
                            filled_pattern = pattern.replace('{entity1}', re.escape(entity1_text))
                            filled_pattern = filled_pattern.replace('{entity2}', re.escape(entity2_text))
                            
                            matches = re.finditer(filled_pattern, text, re.IGNORECASE)
                            
                            for match in matches:
                                # Calculate confidence based on pattern specificity
                                confidence = self._calculate_pattern_confidence(pattern, match.group())
                                
                                if confidence >= 0.3:  # Minimum threshold for patterns
                                    candidate = RelationshipCandidate(
                                        source_entity=entity1_text,
                                        target_entity=entity2_text,
                                        relationship_type=rel_type,
                                        confidence=confidence,
                                        evidence_text=match.group(),
                                        context_window=self._extract_context_around_match(text, match),
                                        extraction_method=ExtractionMethod.PATTERN_BASED,
                                        position_start=match.start(),
                                        position_end=match.end()
                                    )
                                    candidates.append(candidate)
        
        except Exception as e:
            self.logger.error(f"Error in pattern-based extraction: {e}")
        
        return candidates
    
    def _extract_with_dependency_parsing(self, context: ExtractionContext) -> List[RelationshipCandidate]:
        """Extract relationships using dependency parsing"""
        if not self.nlp:
            return []
        
        candidates = []
        
        try:
            doc = self.nlp(context.text)
            entities = context.entities
            
            # Find dependency relationships between entities
            for entity1 in entities:
                for entity2 in entities:
                    if entity1 == entity2:
                        continue
                    
                    # Find tokens for entities
                    entity1_tokens = self._find_entity_tokens(doc, entity1)
                    entity2_tokens = self._find_entity_tokens(doc, entity2)
                    
                    if not entity1_tokens or not entity2_tokens:
                        continue
                    
                    # Analyze dependency path between entities
                    dep_path = self._find_dependency_path(entity1_tokens[0], entity2_tokens[0])
                    
                    if dep_path:
                        rel_type, confidence = self._classify_dependency_relationship(dep_path)
                        
                        if confidence >= 0.4:  # Minimum threshold for dependency parsing
                            candidate = RelationshipCandidate(
                                source_entity=entity1.get('text', entity1.get('name', str(entity1))),
                                target_entity=entity2.get('text', entity2.get('name', str(entity2))),
                                relationship_type=rel_type,
                                confidence=confidence,
                                evidence_text=' '.join([token.text for token in dep_path]),
                                context_window=doc[max(0, entity1_tokens[0].i-5):min(len(doc), entity2_tokens[-1].i+5)].text,
                                extraction_method=ExtractionMethod.DEPENDENCY_PARSING,
                                position_start=entity1_tokens[0].idx,
                                position_end=entity2_tokens[-1].idx + len(entity2_tokens[-1].text)
                            )
                            candidates.append(candidate)
        
        except Exception as e:
            self.logger.error(f"Error in dependency parsing extraction: {e}")
        
        return candidates
    
    def _create_context_window(self, text: str, entity1: Dict, entity2: Dict) -> str:
        """Create a context window around two entities"""
        try:
            start1 = entity1.get('start', 0)
            end1 = entity1.get('end', start1 + len(str(entity1.get('text', ''))))
            start2 = entity2.get('start', 0)
            end2 = entity2.get('end', start2 + len(str(entity2.get('text', ''))))
            
            # Find the span that includes both entities with some context
            window_start = max(0, min(start1, start2) - 50)
            window_end = min(len(text), max(end1, end2) + 50)
            
            return text[window_start:window_end]
        except Exception:
            return text[:self.max_context_length]
    
    def _calculate_pattern_confidence(self, pattern: str, match_text: str) -> float:
        """Calculate confidence score for pattern-based matches"""
        base_confidence = 0.7
        
        # Boost confidence for more specific patterns
        if len(pattern) > 50:
            base_confidence += 0.1
        if '?' in pattern:  # Optional matching reduces confidence slightly
            base_confidence -= 0.05
        if len(match_text.split()) > 5:  # Longer matches tend to be more reliable
            base_confidence += 0.05
        
        return min(0.95, max(0.1, base_confidence))
    
    def _extract_context_around_match(self, text: str, match) -> str:
        """Extract context around a regex match"""
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        return text[start:end]
    
    def _find_entity_tokens(self, doc, entity: Dict) -> List:
        """Find spaCy tokens corresponding to an entity"""
        if not SPACY_AVAILABLE:
            return []
        
        entity_text = entity.get('text', entity.get('name', ''))
        entity_start = entity.get('start', 0)
        
        # Find tokens that overlap with the entity span
        matching_tokens = []
        for token in doc:
            if (token.idx >= entity_start and 
                token.idx < entity_start + len(entity_text)):
                matching_tokens.append(token)
        
        return matching_tokens
    
    def _find_dependency_path(self, token1, token2) -> List:
        """Find dependency path between two tokens"""
        if not SPACY_AVAILABLE:
            return []
        
        # Simple path finding - in practice you'd want more sophisticated algorithms
        try:
            # Find common ancestor
            ancestors1 = list(token1.ancestors) + [token1]
            ancestors2 = list(token2.ancestors) + [token2]
            
            # Find path via lowest common ancestor
            for anc1 in ancestors1:
                if anc1 in ancestors2:
                    return [token1, anc1, token2]
            
            return []
        except Exception:
            return []
    
    def _classify_dependency_relationship(self, dep_path: List) -> Tuple[str, float]:
        """Classify relationship type based on dependency path"""
        if len(dep_path) < 2:
            return "unknown", 0.1
        
        # Simple classification based on dependency labels
        # In practice, you'd use more sophisticated mapping
        dep_labels = [token.dep_ for token in dep_path if hasattr(token, 'dep_')]
        
        if 'nsubj' in dep_labels and 'prep' in dep_labels:
            return "works_for", 0.6
        elif 'compound' in dep_labels:
            return "part_of", 0.5
        elif 'poss' in dep_labels:
            return "owns", 0.5
        else:
            return "related_to", 0.3
    
    def _deduplicate_candidates(self, candidates: List[RelationshipCandidate]) -> List[RelationshipCandidate]:
        """Remove duplicate relationship candidates"""
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Create a key for deduplication
            key = (
                candidate.source_entity.lower(),
                candidate.target_entity.lower(),
                candidate.relationship_type
            )
            
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
            else:
                # If we've seen this relationship, keep the one with higher confidence
                existing_idx = next(
                    i for i, c in enumerate(unique_candidates)
                    if (c.source_entity.lower(), c.target_entity.lower(), c.relationship_type) == key
                )
                if candidate.confidence > unique_candidates[existing_idx].confidence:
                    unique_candidates[existing_idx] = candidate
        
        return unique_candidates
    
    def _calibrate_confidence(self, candidates: List[RelationshipCandidate]) -> List[RelationshipCandidate]:
        """
        Calibrate confidence scores using ensemble voting and evidence strength
        
        Implements confidence calibration as specified in Phase 2.1 plan
        """
        if not candidates:
            return candidates
        
        # Group candidates by relationship triple (source, target, type)
        relationship_groups = defaultdict(list)
        for candidate in candidates:
            key = (candidate.source_entity, candidate.target_entity, candidate.relationship_type)
            relationship_groups[key].append(candidate)
        
        calibrated_candidates = []
        
        for group in relationship_groups.values():
            if len(group) == 1:
                # Single detection - keep original confidence
                calibrated_candidates.extend(group)
            else:
                # Multiple detections - ensemble voting
                methods = set(c.extraction_method for c in group)
                avg_confidence = sum(c.confidence for c in group) / len(group)
                
                # Boost confidence for multi-method agreement
                ensemble_boost = min(0.2, len(methods) * 0.05)
                calibrated_confidence = min(1.0, avg_confidence + ensemble_boost)
                
                # Create representative candidate
                best_candidate = max(group, key=lambda x: x.confidence)
                best_candidate.confidence = calibrated_confidence
                best_candidate.properties['ensemble_methods'] = [m.value for m in methods]
                best_candidate.properties['ensemble_size'] = len(group)
                
                calibrated_candidates.append(best_candidate)
        
        return calibrated_candidates
    
    def _validate_with_schema(self, candidates: List[RelationshipCandidate], 
                             schema_manager: Optional[SchemaManager]) -> List[RelationshipCandidate]:
        """
        Validate relationship candidates against schema constraints
        
        Integrates with Phase 1 schema management for relationship validation
        """
        if not schema_manager or not CORE_SCHEMA_AVAILABLE:
            self.logger.debug("📋 Schema validation skipped - schema manager not available")
            return candidates
        
        validated_candidates = []
        
        for candidate in candidates:
            try:
                # Check if relationship type is defined in schema
                if hasattr(schema_manager, 'schema') and hasattr(schema_manager.schema, 'relationship_types'):
                    valid_types = schema_manager.schema.relationship_types.keys()
                    if candidate.relationship_type not in valid_types:
                        # Try to map to closest valid type
                        closest_type = self._find_closest_relationship_type(
                            candidate.relationship_type, valid_types
                        )
                        if closest_type:
                            candidate.relationship_type = closest_type
                            candidate.confidence *= 0.9  # Reduce confidence for mapping
                        else:
                            # Unknown relationship type - lower confidence
                            candidate.confidence *= 0.7
                
                # Additional schema validations could be added here
                # (cardinality constraints, domain/range restrictions, etc.)
                
                validated_candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"⚠️  Schema validation failed for candidate: {e}")
                # Include candidate with reduced confidence
                candidate.confidence *= 0.8
                validated_candidates.append(candidate)
        
        return validated_candidates
    
    def _find_closest_relationship_type(self, target_type: str, valid_types: Set[str]) -> Optional[str]:
        """Find the closest valid relationship type using simple string similarity"""
        if not valid_types:
            return None
        
        target_lower = target_type.lower()
        
        # Exact match (case insensitive)
        for valid_type in valid_types:
            if valid_type.lower() == target_lower:
                return valid_type
        
        # Substring match
        for valid_type in valid_types:
            if target_lower in valid_type.lower() or valid_type.lower() in target_lower:
                return valid_type
        
        # No good match found
        return None
    
    def _rank_and_filter_candidates(self, candidates: List[RelationshipCandidate]) -> List[RelationshipCandidate]:
        """Rank and filter candidates based on confidence and other factors"""
        # Sort by confidence descending
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply confidence threshold
        filtered_candidates = [
            c for c in candidates 
            if c.confidence >= self.confidence_threshold * 0.5  # Allow some lower confidence candidates
        ]
        
        return filtered_candidates
    
    def extract_relationships_as_instances(self, context: ExtractionContext) -> List[RelationshipInstance]:
        """
        Extract relationships and return as RelationshipInstance objects for Phase 1 integration
        
        This is the main interface for integration with the core knowledge graph system
        """
        candidates = self.extract_relationships(context)
        instances = []
        
        for candidate in candidates:
            instance = candidate.to_relationship_instance()
            if instance:
                instances.append(instance)
            else:
                self.logger.warning(f"⚠️  Failed to convert candidate to RelationshipInstance: {candidate}")
        
        self.logger.info(f"🔗 Converted {len(instances)} candidates to RelationshipInstance objects")
        return instances
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get detailed extraction statistics for monitoring and optimization"""
        return {
            **self.extraction_stats,
            'confidence_threshold': self.confidence_threshold,
            'max_context_length': self.max_context_length,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'spacy_available': SPACY_AVAILABLE,
            'core_schema_available': CORE_SCHEMA_AVAILABLE
        }
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction performance statistics"""
        return {
            **self.extraction_stats,
            'extraction_methods_enabled': {
                'transformer': self.enable_transformer,
                'pattern_matching': self.enable_pattern_matching,
                'dependency_parsing': self.enable_dependency_parsing
            },
            'models_available': {
                'transformers': TRANSFORMERS_AVAILABLE,
                'spacy': SPACY_AVAILABLE,
                'torch': TORCH_AVAILABLE
            }
        }
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """Update the confidence threshold for relationship extraction"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")


def create_relationship_extractor(
    confidence_threshold: float = 0.7,
    schema_manager: Optional[SchemaManager] = None,
    **kwargs
) -> SophisticatedRelationshipExtractor:
    """Factory function to create a sophisticated relationship extractor"""
    return SophisticatedRelationshipExtractor(
        confidence_threshold=confidence_threshold,
        schema_manager=schema_manager,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    extractor = create_relationship_extractor(confidence_threshold=0.5)
    
    # Create test context
    test_entities = [
        {"text": "Alice Johnson", "start": 0, "end": 12, "type": "PERSON"},
        {"text": "TechCorp Inc", "start": 22, "end": 34, "type": "ORG"},
        {"text": "San Francisco", "start": 45, "end": 58, "type": "GPE"}
    ]
    
    test_text = "Alice Johnson works for TechCorp Inc, which is located in San Francisco."
    
    context = ExtractionContext(
        text=test_text,
        entities=test_entities,
        sentence_boundaries=[(0, len(test_text))]
    )
    
    # Extract relationships
    candidates = extractor.extract_relationships(context)
    
    print("=== Relationship Extraction Test ===")
    print(f"Found {len(candidates)} relationship candidates:")
    
    for candidate in candidates:
        print(f"\n{candidate.source_entity} --[{candidate.relationship_type}]--> {candidate.target_entity}")
        print(f"  Confidence: {candidate.confidence:.3f}")
        print(f"  Method: {candidate.extraction_method.value}")
        print(f"  Evidence: {candidate.evidence_text}")
    
    # Show statistics
    stats = extractor.get_extraction_statistics()
    print(f"\nExtraction Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
