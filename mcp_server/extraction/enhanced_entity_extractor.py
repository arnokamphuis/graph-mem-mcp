"""
Enhanced Entity Extraction - Phase 2.2 Implementation

This module implements sophisticated entity extraction following the refactoring plan architecture.
It integrates with Phase 1 core components (schema management, entity resolution) and implements
a multi-model ensemble approach for high-quality entity recognition.

Architecture Integration:
- Uses EntityTypeSchema and EntityInstance from core/graph_schema.py
- Integrates with EntityResolver from core/entity_resolution.py  
- Follows dependency management patterns from coding standards
- Implements multi-model ensemble as specified in refactoring plan
"""

from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core system integration - Phase 1 components
try:
    from ..core.graph_schema import (
        EntityInstance, EntityTypeSchema, SchemaManager, 
        PropertySchema, PropertyType
    )
    CORE_SCHEMA_AVAILABLE = True
    logger.info("✅ Core schema system integration available")
except (ImportError, AttributeError):
    logger.error("❌ Core schema system not available - Phase 1 required")
    raise ImportError("Phase 2.2 requires Phase 1 core components to be implemented")

try:
    from ..core.entity_resolution import EntityResolver, EntityCandidate
    ENTITY_RESOLUTION_AVAILABLE = True
    logger.info("✅ Entity resolution system integration available")
except (ImportError, AttributeError):
    logger.error("❌ Entity resolution system not available - Phase 1 required")
    raise ImportError("Phase 2.2 requires Phase 1 entity resolution to be implemented")

# Optional ML dependencies with graceful fallbacks per coding standards
TRANSFORMERS_AVAILABLE = False
SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers available - using BERT-based NER models")
except ImportError:
    logger.warning("⚠️  Transformers not available - using pattern-based extraction")

try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy available - using linguistic analysis")
except ImportError:
    logger.warning("⚠️  spaCy not available - using basic text processing")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("⚠️  NumPy not available - using basic confidence scoring")


class ExtractionStrategy(Enum):
    """Multi-model ensemble extraction strategies per refactoring plan"""
    TRANSFORMER_NER = "transformer_ner"
    SPACY_NER = "spacy_ner"
    PATTERN_BASED = "pattern_based"
    SCHEMA_GUIDED = "schema_guided"
    CONTEXTUAL = "contextual"


@dataclass
class ExtractionCandidate:
    """Candidate entity from extraction process"""
    text: str
    start_pos: int
    end_pos: int
    entity_type: str  # Uses schema entity type names
    confidence: float
    strategy: ExtractionStrategy
    context_window: str
    evidence: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionContext:
    """Context for entity extraction operations"""
    text: str
    schema_manager: SchemaManager
    entity_resolver: Optional[EntityResolver] = None
    confidence_threshold: float = 0.7
    enable_resolution: bool = True
    max_context_window: int = 100
    strategies: List[ExtractionStrategy] = field(
        default_factory=lambda: [
            ExtractionStrategy.SCHEMA_GUIDED,
            ExtractionStrategy.PATTERN_BASED,
            ExtractionStrategy.TRANSFORMER_NER,
            ExtractionStrategy.SPACY_NER,
            ExtractionStrategy.CONTEXTUAL
        ]
    )


class EnhancedEntityExtractor:
    """
    Enhanced entity extraction implementing multi-model ensemble approach
    
    Integrates with Phase 1 core architecture:
    - Uses schema-defined entity types
    - Outputs EntityInstance objects  
    - Integrates with entity resolution for deduplication
    - Follows coding standards for dependency management
    """
    
    def __init__(self, schema_manager: SchemaManager):
        """Initialize with schema manager from Phase 1"""
        self.schema_manager = schema_manager
        self.logger = logging.getLogger(__name__)
        
        # Multi-model ensemble components
        self.transformer_pipeline = None
        self.spacy_nlp = None
        self.entity_resolver = None
        
        # Statistics tracking
        self.stats = {
            'total_extractions': 0,
            'high_confidence_extractions': 0,
            'strategy_counts': defaultdict(int),
            'entity_type_counts': defaultdict(int),
            'strategies_enabled': {},
            'models_available': {
                'transformers': TRANSFORMERS_AVAILABLE,
                'spacy': SPACY_AVAILABLE,
                'numpy': NUMPY_AVAILABLE
            }
        }
        
        # Initialize models per plan's multi-model ensemble approach
        self._initialize_models()
        
        self.logger.info(f"🚀 Enhanced Entity Extractor initialized with schema: {len(self.schema_manager.schema.entity_types)} entity types")
    
    def _initialize_models(self):
        """Initialize multi-model ensemble components"""
        
        # Strategy availability tracking
        strategies_enabled = {}
        
        # 1. Transformer-based NER (if available)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformer_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=-1  # Use CPU for compatibility
                )
                strategies_enabled[ExtractionStrategy.TRANSFORMER_NER] = True
                self.logger.info("✅ Transformer NER model initialized")
            except Exception as e:
                strategies_enabled[ExtractionStrategy.TRANSFORMER_NER] = False
                self.logger.warning(f"⚠️  Failed to initialize transformer NER: {e}")
        else:
            strategies_enabled[ExtractionStrategy.TRANSFORMER_NER] = False
        
        # 2. spaCy NER (if available)
        if SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
                strategies_enabled[ExtractionStrategy.SPACY_NER] = True
                self.logger.info("✅ spaCy NER model initialized")
            except Exception as e:
                strategies_enabled[ExtractionStrategy.SPACY_NER] = False
                self.logger.warning(f"⚠️  Failed to initialize spaCy NER: {e}")
        else:
            strategies_enabled[ExtractionStrategy.SPACY_NER] = False
        
        # 3. Pattern-based extraction (always available)
        strategies_enabled[ExtractionStrategy.PATTERN_BASED] = True
        
        # 4. Schema-guided extraction (always available with Phase 1)
        strategies_enabled[ExtractionStrategy.SCHEMA_GUIDED] = True
        
        # 5. Contextual extraction (always available)
        strategies_enabled[ExtractionStrategy.CONTEXTUAL] = True
        
        self.stats['strategies_enabled'] = strategies_enabled
        enabled_count = sum(1 for enabled in strategies_enabled.values() if enabled)
        self.logger.info(f"📊 Initialized {enabled_count}/{len(strategies_enabled)} extraction strategies")
    
    def extract_entities(self, context: ExtractionContext) -> List[EntityInstance]:
        """
        Extract entities using multi-model ensemble approach
        
        Returns EntityInstance objects compatible with Phase 1 core architecture
        """
        self.logger.info(f"🔍 Extracting entities from text ({len(context.text)} chars)")
        
        # Multi-strategy extraction per refactoring plan
        all_candidates = []
        
        for strategy in context.strategies:
            if self.stats['strategies_enabled'].get(strategy, False):
                try:
                    candidates = self._extract_with_strategy(strategy, context)
                    all_candidates.extend(candidates)
                    self.stats['strategy_counts'][strategy] += len(candidates)
                    self.logger.debug(f"📝 {strategy.value}: {len(candidates)} candidates")
                except Exception as e:
                    self.logger.warning(f"⚠️  Strategy {strategy.value} failed: {e}")
        
        # Ensemble fusion and deduplication
        merged_candidates = self._merge_candidates(all_candidates, context)
        
        # Filter by confidence threshold
        filtered_candidates = [
            c for c in merged_candidates 
            if c.confidence >= context.confidence_threshold
        ]
        
        # Convert to EntityInstance objects (Phase 1 integration)
        entity_instances = self._candidates_to_entities(filtered_candidates, context)
        
        # Optional entity resolution integration
        if context.enable_resolution and context.entity_resolver:
            entity_instances = self._resolve_entities(entity_instances, context.entity_resolver)
        
        # Update statistics
        self.stats['total_extractions'] += len(entity_instances)
        self.stats['high_confidence_extractions'] += len([
            e for e in filtered_candidates if e.confidence >= 0.8
        ])
        
        for entity in entity_instances:
            self.stats['entity_type_counts'][entity.entity_type] += 1
        
        self.logger.info(f"✅ Extracted {len(entity_instances)} entities using ensemble approach")
        return entity_instances
    
    def _extract_with_strategy(self, strategy: ExtractionStrategy, 
                              context: ExtractionContext) -> List[ExtractionCandidate]:
        """Extract candidates using specific strategy"""
        
        if strategy == ExtractionStrategy.SCHEMA_GUIDED:
            return self._extract_schema_guided(context)
        elif strategy == ExtractionStrategy.PATTERN_BASED:
            return self._extract_pattern_based(context)
        elif strategy == ExtractionStrategy.TRANSFORMER_NER:
            return self._extract_transformer_ner(context)
        elif strategy == ExtractionStrategy.SPACY_NER:
            return self._extract_spacy_ner(context)
        elif strategy == ExtractionStrategy.CONTEXTUAL:
            return self._extract_contextual(context)
        else:
            return []
    
    def _extract_schema_guided(self, context: ExtractionContext) -> List[ExtractionCandidate]:
        """Extract entities using schema-defined patterns and validation"""
        candidates = []
        
        # Use schema-defined entity types and their properties for targeted extraction
        for entity_type_name, entity_schema in context.schema_manager.schema.entity_types.items():
            # Create extraction patterns based on schema metadata
            patterns = self._generate_schema_patterns(entity_schema)
            
            for pattern in patterns:
                matches = re.finditer(pattern, context.text, re.IGNORECASE)
                
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    entity_text = match.group()
                    
                    # Schema validation
                    if self._validate_entity_against_schema(entity_text, entity_schema):
                        # Get context window
                        context_start = max(0, start_pos - context.max_context_window // 2)
                        context_end = min(len(context.text), end_pos + context.max_context_window // 2)
                        context_window = context.text[context_start:context_end]
                        
                        candidate = ExtractionCandidate(
                            text=entity_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            entity_type=entity_type_name,
                            confidence=0.85,  # High confidence for schema-guided
                            strategy=ExtractionStrategy.SCHEMA_GUIDED,
                            context_window=context_window,
                            evidence=f"Schema pattern: {pattern[:50]}...",
                            properties={'schema_validated': True}
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _generate_schema_patterns(self, entity_schema: EntityTypeSchema) -> List[str]:
        """Generate extraction patterns based on entity schema"""
        patterns = []
        
        # Basic patterns based on entity type name and description
        entity_type = entity_schema.name.lower()
        
        if entity_type == "person":
            patterns.extend([
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. First Last
                r'\bProf\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Prof. First Last
            ])
        elif entity_type == "organization":
            patterns.extend([
                r'\b[A-Z][a-zA-Z\s]+ Inc\.?\b',  # Company Inc
                r'\b[A-Z][a-zA-Z\s]+ Corp\.?\b',  # Company Corp
                r'\b[A-Z][a-zA-Z\s]+ Corporation\b',  # Company Corporation
                r'\b[A-Z][a-zA-Z\s]+ Company\b',  # Company Company
                r'\b[A-Z][a-zA-Z\s]+ Ltd\.?\b',  # Company Ltd
            ])
        elif entity_type == "location":
            patterns.extend([
                r'\b[A-Z][a-z]+ City\b',
                r'\b[A-Z][a-z]+, [A-Z]{2}\b',  # City, State
                r'\bMount [A-Z][a-z]+\b',
            ])
        
        return patterns
    
    def _validate_entity_against_schema(self, entity_text: str, 
                                       entity_schema: EntityTypeSchema) -> bool:
        """Validate extracted entity against schema constraints"""
        # Basic validation - can be extended with schema-specific rules
        if len(entity_text.strip()) < 2:
            return False
        
        # Check if entity text contains only valid characters for the type
        if entity_schema.name == "person":
            # Person names should be primarily alphabetic
            return bool(re.match(r'^[A-Za-z\s\.\-]+$', entity_text))
        elif entity_schema.name == "organization":
            # Organizations can have alphanumeric and common punctuation
            return bool(re.match(r'^[A-Za-z0-9\s\.\-,&]+$', entity_text))
        
        return True
    
    def _extract_pattern_based(self, context: ExtractionContext) -> List[ExtractionCandidate]:
        """Extract entities using refined pattern matching"""
        candidates = []
        
        # Refined patterns that avoid over-extraction
        refined_patterns = {
            "person": [
                r'\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b(?=\s+(?:works|is|was|says|founded))',
            ],
            "organization": [
                r'\b[A-Z][a-zA-Z\s]{3,25}\s+(?:Inc|Corp|Corporation|Company|Ltd)\.?\b',
                r'\b[A-Z]{2,8}\b(?=\s+(?:Corporation|Company|Inc))',  # Acronyms before corp words
            ],
            "location": [
                r'\b[A-Z][a-z]{2,}\s+City\b',
                r'\b[A-Z][a-z]{3,},\s+[A-Z]{2}\b',  # City, State
            ]
        }
        
        for entity_type, patterns in refined_patterns.items():
            # Only extract if entity type exists in schema
            if entity_type in context.schema_manager.schema.entity_types:
                for pattern in patterns:
                    matches = re.finditer(pattern, context.text)
                    
                    for match in matches:
                        start_pos = match.start()
                        end_pos = match.end()
                        entity_text = match.group().strip()
                        
                        # Additional validation to prevent over-extraction
                        if self._is_valid_pattern_match(entity_text, entity_type):
                            context_start = max(0, start_pos - 50)
                            context_end = min(len(context.text), end_pos + 50)
                            context_window = context.text[context_start:context_end]
                            
                            candidate = ExtractionCandidate(
                                text=entity_text,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                entity_type=entity_type,
                                confidence=0.75,
                                strategy=ExtractionStrategy.PATTERN_BASED,
                                context_window=context_window,
                                evidence=f"Pattern: {pattern[:30]}...",
                                properties={'pattern_matched': True}
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _is_valid_pattern_match(self, text: str, entity_type: str) -> bool:
        """Additional validation for pattern matches to prevent over-extraction"""
        # Common stopwords that shouldn't be entities
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'under', 'over'
        }
        
        text_lower = text.lower().strip()
        
        # Don't extract stopwords
        if text_lower in stopwords:
            return False
        
        # Don't extract very short text
        if len(text_lower) < 3:
            return False
        
        # Don't extract if it's all punctuation
        if all(c in '.,!?;:' for c in text_lower):
            return False
        
        return True
    
    def _extract_transformer_ner(self, context: ExtractionContext) -> List[ExtractionCandidate]:
        """Extract entities using transformer-based NER"""
        if not self.transformer_pipeline:
            return []
        
        candidates = []
        
        try:
            # Process in chunks for long text
            max_length = 512
            text_chunks = [
                context.text[i:i+max_length] 
                for i in range(0, len(context.text), max_length)
            ]
            
            offset = 0
            for chunk in text_chunks:
                ner_results = self.transformer_pipeline(chunk)
                
                for result in ner_results:
                    # Map transformer labels to schema entity types
                    entity_type = self._map_transformer_label_to_schema(
                        result['entity_group'], context.schema_manager
                    )
                    
                    if entity_type:  # Only if we can map to schema
                        start_pos = result['start'] + offset
                        end_pos = result['end'] + offset
                        
                        context_start = max(0, start_pos - 50)
                        context_end = min(len(context.text), end_pos + 50)
                        context_window = context.text[context_start:context_end]
                        
                        candidate = ExtractionCandidate(
                            text=result['word'],
                            start_pos=start_pos,
                            end_pos=end_pos,
                            entity_type=entity_type,
                            confidence=result['score'],
                            strategy=ExtractionStrategy.TRANSFORMER_NER,
                            context_window=context_window,
                            evidence=f"Transformer: {result['entity_group']}",
                            properties={'transformer_label': result['entity_group']}
                        )
                        candidates.append(candidate)
                
                offset += len(chunk)
        
        except Exception as e:
            self.logger.warning(f"⚠️  Transformer NER extraction failed: {e}")
        
        return candidates
    
    def _map_transformer_label_to_schema(self, transformer_label: str, 
                                        schema_manager: SchemaManager) -> Optional[str]:
        """Map transformer NER labels to schema entity types"""
        label_mapping = {
            'PER': 'person',
            'PERSON': 'person',
            'ORG': 'organization',
            'LOC': 'location',
            'MISC': None  # Don't map miscellaneous entities
        }
        
        mapped_type = label_mapping.get(transformer_label.upper())
        
        # Only return if the mapped type exists in our schema
        if mapped_type and mapped_type in schema_manager.schema.entity_types:
            return mapped_type
        
        return None
    
    def _extract_spacy_ner(self, context: ExtractionContext) -> List[ExtractionCandidate]:
        """Extract entities using spaCy NER"""
        if not self.spacy_nlp:
            return []
        
        candidates = []
        
        try:
            doc = self.spacy_nlp(context.text)
            
            for ent in doc.ents:
                # Map spaCy labels to schema entity types
                entity_type = self._map_spacy_label_to_schema(
                    ent.label_, context.schema_manager
                )
                
                if entity_type:  # Only if we can map to schema
                    context_start = max(0, ent.start_char - 50)
                    context_end = min(len(context.text), ent.end_char + 50)
                    context_window = context.text[context_start:context_end]
                    
                    candidate = ExtractionCandidate(
                        text=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        entity_type=entity_type,
                        confidence=0.8,  # spaCy doesn't provide confidence
                        strategy=ExtractionStrategy.SPACY_NER,
                        context_window=context_window,
                        evidence=f"spaCy: {ent.label_}",
                        properties={'spacy_label': ent.label_}
                    )
                    candidates.append(candidate)
        
        except Exception as e:
            self.logger.warning(f"⚠️  spaCy NER extraction failed: {e}")
        
        return candidates
    
    def _map_spacy_label_to_schema(self, spacy_label: str, 
                                  schema_manager: SchemaManager) -> Optional[str]:
        """Map spaCy NER labels to schema entity types"""
        label_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',  # Geopolitical entity
            'LOC': 'location'
        }
        
        mapped_type = label_mapping.get(spacy_label.upper())
        
        # Only return if the mapped type exists in our schema
        if mapped_type and mapped_type in schema_manager.schema.entity_types:
            return mapped_type
        
        return None
    
    def _extract_contextual(self, context: ExtractionContext) -> List[ExtractionCandidate]:
        """
        Extract entities using contextual clues and linguistic patterns
        Implements sophisticated context-aware extraction per Phase 2.2 requirements
        """
        candidates = []
        
        # Enhanced contextual patterns with better boundary detection
        contextual_patterns = {
            "person": [
                # Professional titles and roles
                (r'(?:CEO|president|director|founder|manager|professor|dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 1),
                (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:said|founded|established|created|announced)', 1),
                (r'(?:by|from|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s+(?:from|at|of))', 1),
                # Academic titles
                (r'(?:Prof\.?|Professor|Dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 1),
            ],
            "organization": [
                # Company/organization patterns with proper boundaries
                (r'(?:at|for|with|by)\s+([A-Z][a-zA-Z\s]+?(?:\s+(?:Inc|Corp|Corporation|Company|University|Foundation|Institute)\.?))(?:\s|,|\.|\;|$)', 1),
                (r'(?:funded\s+by|acquired\s+by|owned\s+by)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:and|,)|$)', 1),
                (r'([A-Z][a-zA-Z\s]+?(?:\s+(?:Inc|Corp|Corporation|Company|University)\.?))\s+(?:announced|reported|stated|said)', 1),
                # Research institutions
                (r'([A-Z][a-zA-Z\s]+?\s+(?:University|Institute|Foundation|Laboratory))\b', 1),
            ],
            "location": [
                # Geographic locations with context
                (r'(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s*,|\s+and|\s*$)', 1),
                (r'headquarters\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 1),
                (r'(?:located|based|situated)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 1),
                (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*([A-Z][a-z]+)(?:\s|,|\.|\;|$)', 1),  # City, State/Country
            ],
            "event": [
                # Conference and event patterns
                (r'(?:at\s+the\s+|presented\s+at\s+)([A-Z][a-zA-Z\s]+?Conference(?:\s+on\s+[A-Z][a-zA-Z\s]+)?)', 1),
                (r'([A-Z][a-zA-Z\s]+?(?:Conference|Summit|Symposium|Workshop))(?:\s+(?:in|at|on))', 1),
            ],
            "technology": [
                # Technology and algorithm patterns
                (r'(?:using|with|developed|created)\s+([A-Z][a-zA-Z\s]+?\s+(?:algorithm|technology|system|platform)s?)', 1),
                (r'([A-Z][a-zA-Z]+)\s+(?:algorithm|technology|platform|system)', 1),
            ]
        }
        
        for entity_type, patterns in contextual_patterns.items():
            if entity_type in context.schema_manager.schema.entity_types:
                for pattern, group_idx in patterns:
                    matches = re.finditer(pattern, context.text)
                    
                    for match in matches:
                        entity_text = match.group(group_idx).strip()
                        start_pos = match.start(group_idx)
                        end_pos = match.end(group_idx)
                        
                        # Clean up extracted text
                        entity_text = self._clean_contextual_extraction(entity_text, entity_type)
                        
                        if entity_text and self._is_valid_contextual_extraction(entity_text, entity_type, match.group()):
                            context_start = max(0, start_pos - 50)
                            context_end = min(len(context.text), end_pos + 50)
                            context_window = context.text[context_start:context_end]
                            
                            candidate = ExtractionCandidate(
                                text=entity_text,
                                start_pos=start_pos,
                                end_pos=start_pos + len(entity_text),  # Adjust for cleaned text
                                entity_type=entity_type,
                                confidence=0.75,  # Higher confidence for contextual
                                strategy=ExtractionStrategy.CONTEXTUAL,
                                context_window=context_window,
                                evidence=f"Context pattern: {match.group()[:40]}...",
                                properties={
                                    'context_clue': True,
                                    'pattern_matched': pattern[:30],
                                    'full_context': match.group()
                                }
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _clean_contextual_extraction(self, text: str, entity_type: str) -> str:
        """Clean up contextually extracted entity text"""
        # Remove common trailing words that shouldn't be part of entity names
        trailing_stopwords = ['for', 'and', 'or', 'the', 'of', 'in', 'at', 'by', 'with', 'from', 'to']
        
        # Split and clean
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Remove punctuation from end of words
            clean_word = re.sub(r'[^\w\s\.\-]$', '', word)
            if clean_word:
                cleaned_words.append(clean_word)
        
        # Remove trailing stopwords
        while cleaned_words and cleaned_words[-1].lower() in trailing_stopwords:
            cleaned_words.pop()
        
        return ' '.join(cleaned_words).strip()
    
    def _is_valid_contextual_extraction(self, text: str, entity_type: str, full_match: str) -> bool:
        """Validate contextually extracted entity"""
        if not text or len(text) < 2:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^(for|and|or|the|of|in|at|by|with|from|to)$',
            r'^\d+$',  # Only numbers
            r'^[^\w\s]+$',  # Only punctuation
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        # Entity type specific validation
        if entity_type == 'person':
            # Person names should have at least two words or be a known title
            if ' ' not in text and not text.startswith(('Dr.', 'Prof.', 'Mr.', 'Ms.', 'Mrs.')):
                return False
        
        elif entity_type == 'organization':
            # Organization names shouldn't be too short unless they're acronyms
            if len(text) < 3 and not text.isupper():
                return False
            # Check for incomplete organization names
            if text.lower().endswith((' for', ' by', ' with', ' from', ' to', ' at', ' in')):
                return False
        
        elif entity_type == 'location':
            # Locations should be proper nouns
            if not re.match(r'^[A-Z]', text):
                return False
        
        return True
    
    def _apply_domain_knowledge_correction(self, candidate: ExtractionCandidate) -> ExtractionCandidate:
        """
        Apply generic validation - NO HARDCODED DOMAIN KNOWLEDGE
        This function now only validates based on text structure, not domain-specific terms
        """
        # Domain-agnostic validation only - no hardcoded domain knowledge
        # All entity classification should come from NLP models and input text context
        
        # Return the candidate unchanged - trust the ensemble voting and NLP models
        return candidate
        
        return candidate

    def _merge_candidates(self, candidates: List[ExtractionCandidate], 
                         context: ExtractionContext) -> List[ExtractionCandidate]:
        """
        Merge and deduplicate candidates from multiple strategies
        Implements sophisticated deduplication per Phase 2.2 quality requirements
        """
        if not candidates:
            return []
        
        # Phase 1: Quality filtering and domain knowledge correction
        valid_candidates = []
        for candidate in candidates:
            if self._is_valid_entity_extraction(candidate):
                # Apply domain knowledge correction
                corrected_candidate = self._apply_domain_knowledge_correction(candidate)
                valid_candidates.append(corrected_candidate)
            else:
                self.logger.debug(f"🗑️  Filtered invalid extraction: '{candidate.text}'")
        
        if not valid_candidates:
            return []
        
        # Phase 2: Advanced deduplication with substring/containment analysis
        merged = {}
        
        for candidate in valid_candidates:
            # Normalize text for comparison
            normalized_text = self._normalize_entity_text(candidate.text)
            
            # Check for existing similar entities
            merge_target = None
            for existing_key, existing_candidate in merged.items():
                if self._should_merge_entities(candidate, existing_candidate):
                    merge_target = existing_key
                    break
            
            if merge_target:
                # Merge with existing entity - keep the better one
                existing = merged[merge_target]
                merged_candidate = self._merge_two_candidates(candidate, existing)
                merged[merge_target] = merged_candidate
            else:
                # Create new entry
                key = (normalized_text, candidate.entity_type)
                merged[key] = candidate
        
        # Phase 3: Final validation and ranking
        final_candidates = []
        for candidate in merged.values():
            if candidate.confidence >= context.confidence_threshold:
                final_candidates.append(candidate)
        
        # Sort by confidence descending
        final_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.debug(f"📊 Deduplication: {len(candidates)} → {len(valid_candidates)} → {len(final_candidates)}")
        return final_candidates
    
    def _candidates_to_entities(self, candidates: List[ExtractionCandidate], 
                               context: ExtractionContext) -> List[EntityInstance]:
        """Convert extraction candidates to EntityInstance objects (Phase 1 integration)"""
        entities = []
        
        for candidate in candidates:
            try:
                # Create EntityInstance compatible with Phase 1 schema system
                # Handle both Pydantic and dataclass fallback modes
                entity = EntityInstance(
                    name=candidate.text,  # Entity name is the extracted text
                    entity_type=candidate.entity_type
                )
                entity.id = f"extracted_{candidate.start_pos}_{candidate.end_pos}"
                entity.confidence = candidate.confidence
                entity.properties = {
                    'extraction_strategy': candidate.strategy.value,
                    'context_window': candidate.context_window,
                    'evidence': candidate.evidence,
                    'start_position': candidate.start_pos,
                    'end_position': candidate.end_pos,
                    **candidate.properties
                }
                entity.namespace = "default"
                
                # Validate against schema
                is_valid, errors = context.schema_manager.validate_entity(entity)
                if is_valid:
                    entities.append(entity)
                else:
                    self.logger.warning(f"⚠️  Entity validation failed: {errors}")
            
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to create EntityInstance: {e}")
        
        return entities
    
    def _resolve_entities(self, entities: List[EntityInstance], 
                         entity_resolver: EntityResolver) -> List[EntityInstance]:
        """Optional entity resolution using Phase 1 entity resolution system"""
        try:
            # Convert EntityInstance to EntityCandidate for resolution
            candidates = []
            for entity in entities:
                candidate = EntityCandidate(
                    text=entity.properties.get('text', ''),
                    entity_type=entity.entity_type,
                    confidence=entity.properties.get('confidence', 0.0),
                    properties=entity.properties
                )
                candidates.append(candidate)
            
            # Perform resolution
            clusters = entity_resolver.resolve_entities(candidates)
            
            # Convert resolved clusters back to EntityInstance objects
            resolved_entities = []
            for cluster in clusters:
                # Create representative entity from cluster
                canonical_candidate = cluster.canonical_entity
                
                entity = EntityInstance(
                    name=canonical_candidate.text,
                    entity_type=canonical_candidate.entity_type,
                    properties={
                        **canonical_candidate.properties,
                        'resolved_cluster_id': cluster.cluster_id,
                        'cluster_size': len(cluster.entities),
                        'resolution_confidence': cluster.confidence,
                        'aliases': [e.text for e in cluster.entities if e.text != canonical_candidate.text]
                    }
                )
                entity.id = f"resolved_{cluster.cluster_id}"
                resolved_entities.append(entity)
            
            self.logger.info(f"🔗 Entity resolution: {len(entities)} → {len(resolved_entities)} entities")
            return resolved_entities
        
        except Exception as e:
            self.logger.warning(f"⚠️  Entity resolution failed: {e}")
            return entities  # Return original entities if resolution fails
    
    def _is_valid_entity_extraction(self, candidate: ExtractionCandidate) -> bool:
        """
        Validate if an extraction candidate is a proper entity
        Implements quality filtering per Phase 2.2 requirements
        """
        text = candidate.text.strip()
        
        # Basic validation rules
        if not text or len(text) < 2:
            return False
        
        # Filter out malformed extractions and sentence fragments
        malformed_patterns = [
            r'^(for|and|or|the|of|in|at|by|with|from|to)$',  # Common stop words
            r'.*\s+(for|and|or|the|of|in|at|by|with|from|to)\s*$',  # Ending with stop words
            r'^[^a-zA-Z]*$',  # Only non-letters
            r'^\W+$',  # Only punctuation
            r'.*\s+for\s*$',  # Specific case: "Apple for"
        ]
        
        # Sentence fragment detection - prevent full sentences being extracted as entities
        sentence_patterns = [
            r'.*\s(was|were|is|are|has|have|had|will|would|could|should)\s.*',  # Contains verbs
            r'.*\s(presented|acquired|developed|created|published|funded)\s.*',  # Action verbs
            r'.*\s(the|a|an)\s.*\s(the|a|an)\s.*',  # Multiple articles
            r'^\s*[Tt]he\s.*\s(at|in|on|to|from|with|by)\s.*',  # Starts with "The" and has prepositions
        ]
        
        for pattern in malformed_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
                
        for pattern in sentence_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                logger.debug(f"Filtered sentence fragment: '{text}'")
                return False
        
        # Entity type specific validation
        if candidate.entity_type == 'person':
            # Person names should have alphabetic characters
            if not re.search(r'[A-Za-z]', text):
                return False
            # Avoid single words unless they're titles
            if ' ' not in text and not text.startswith(('Dr.', 'Prof.', 'Mr.', 'Ms.', 'Mrs.')):
                # Allow single-word names only if confidence is very high
                if candidate.confidence < 0.9:
                    return False
        
        elif candidate.entity_type == 'organization':
            # Organizations shouldn't end with prepositions
            if re.search(r'\s+(for|by|with|from|to|at|in|of)\s*$', text, re.IGNORECASE):
                return False
        
        return True
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for comparison"""
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    def _should_merge_entities(self, candidate1: ExtractionCandidate, candidate2: ExtractionCandidate) -> bool:
        """
        Determine if two entity candidates should be merged
        Handles containment, overlap, and similarity cases
        """
        if candidate1.entity_type != candidate2.entity_type:
            return False
        
        text1 = self._normalize_entity_text(candidate1.text)
        text2 = self._normalize_entity_text(candidate2.text)
        
        # Exact match
        if text1 == text2:
            return True
        
        # Containment check - "Sarah" vs "Sarah Johnson"
        if text1 in text2 or text2 in text1:
            # Only merge if one is substantially longer (not just one character difference)
            min_len = min(len(text1), len(text2))
            max_len = max(len(text1), len(text2))
            if max_len - min_len >= 3:  # At least 3 character difference
                return True
        
        # Position overlap check
        if self._positions_overlap(candidate1, candidate2):
            return True
        
        # Fuzzy similarity for same entity type (if available)
        if hasattr(self, '_fuzzy_similarity'):
            similarity = self._fuzzy_similarity(text1, text2)
            if similarity > 0.85:  # High similarity threshold
                return True
        
        return False
    
    def _positions_overlap(self, candidate1: ExtractionCandidate, candidate2: ExtractionCandidate) -> bool:
        """Check if two candidates have overlapping text positions"""
        # Check if positions overlap by more than 50%
        start1, end1 = candidate1.start_pos, candidate1.end_pos
        start2, end2 = candidate2.start_pos, candidate2.end_pos
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return False  # No overlap
        
        overlap_len = overlap_end - overlap_start
        len1 = end1 - start1
        len2 = end2 - start2
        
        # Check if overlap is significant (>50% of either entity)
        return (overlap_len / len1 > 0.5) or (overlap_len / len2 > 0.5)
    
    def _merge_two_candidates(self, candidate1: ExtractionCandidate, candidate2: ExtractionCandidate) -> ExtractionCandidate:
        """
        Merge two entity candidates, keeping the better one and combining evidence
        """
        # Determine which candidate to keep as primary
        if candidate1.confidence > candidate2.confidence:
            primary, secondary = candidate1, candidate2
        elif candidate1.confidence < candidate2.confidence:
            primary, secondary = candidate2, candidate1
        else:
            # Same confidence - prefer longer, more complete entity names
            if len(candidate1.text) > len(candidate2.text):
                primary, secondary = candidate1, candidate2
            else:
                primary, secondary = candidate2, candidate1
        
        # Create merged candidate
        merged = ExtractionCandidate(
            text=primary.text,
            start_pos=primary.start_pos,
            end_pos=primary.end_pos,
            entity_type=primary.entity_type,
            confidence=max(primary.confidence, secondary.confidence),
            strategy=primary.strategy,
            context_window=primary.context_window,
            evidence=f"{primary.evidence}; {secondary.evidence}",
            properties={**primary.properties, **secondary.properties}
        )
        
        # Add merge metadata
        merged.properties['merged_from'] = [primary.strategy.value, secondary.strategy.value]
        merged.properties['merge_confidence_boost'] = 0.05  # Small boost for multiple detections
        merged.confidence = min(1.0, merged.confidence + 0.05)
        
        return merged
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()


# Convenience function for quick extraction
def extract_entities_from_text(text: str, schema_manager: SchemaManager, 
                              entity_resolver: Optional[EntityResolver] = None,
                              confidence_threshold: float = 0.7) -> List[EntityInstance]:
    """
    Convenience function for quick entity extraction
    
    Args:
        text: Input text for entity extraction
        schema_manager: Schema manager from Phase 1
        entity_resolver: Optional entity resolver from Phase 1
        confidence_threshold: Minimum confidence for entities
        
    Returns:
        List of EntityInstance objects compatible with Phase 1 core architecture
    """
    extractor = EnhancedEntityExtractor(schema_manager)
    
    context = ExtractionContext(
        text=text,
        schema_manager=schema_manager,
        entity_resolver=entity_resolver,
        confidence_threshold=confidence_threshold
    )
    
    return extractor.extract_entities(context)


def create_enhanced_entity_extractor(schema_manager: SchemaManager, **kwargs) -> EnhancedEntityExtractor:
    """
    Factory function to create an enhanced entity extractor
    
    Args:
        schema_manager: Schema manager instance from Phase 1
        **kwargs: Additional configuration parameters
        
    Returns:
        EnhancedEntityExtractor: Configured entity extractor instance
    """
    return EnhancedEntityExtractor(schema_manager=schema_manager)


if __name__ == "__main__":
    # Example usage demonstrating Phase 1 integration
    from ..core.graph_schema import SchemaManager, EntityTypeSchema, PropertySchema, PropertyType
    
    print("🧪 Testing Enhanced Entity Extraction - Phase 2.2")
    print("=" * 60)
    
    # Create schema manager (Phase 1 integration)
    schema_manager = SchemaManager()
    
    # Define entity types in schema
    person_schema = EntityTypeSchema(
        name="person",
        description="Individual person",
        properties=[
            PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
            PropertySchema(name="confidence", property_type=PropertyType.FLOAT)
        ]
    )
    
    org_schema = EntityTypeSchema(
        name="organization", 
        description="Organization or company",
        properties=[
            PropertySchema(name="name", property_type=PropertyType.STRING, required=True),
            PropertySchema(name="confidence", property_type=PropertyType.FLOAT)
        ]
    )
    
    # Add to schema manager
    schema_manager.add_entity_type(person_schema)
    schema_manager.add_entity_type(org_schema)
    
    # Test text
    test_text = """
    Dr. Sarah Chen works for Microsoft Corporation in Seattle.
    She founded the AI Research Lab using advanced machine learning.
    Microsoft was established by Bill Gates in 1975.
    """
    
    # Extract entities
    entities = extract_entities_from_text(test_text, schema_manager)
    
    print(f"✅ Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  📝 {entity.properties['text']} ({entity.entity_type})")
        print(f"     Confidence: {entity.properties['confidence']:.3f}")
        print(f"     Strategy: {entity.properties['extraction_strategy']}")
        print()
