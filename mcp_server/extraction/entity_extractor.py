"""
Enhanced Entity Extraction Module for Knowledge Graph Processing

This module provides sophisticated entity extraction capabilities with multiple
extraction strategies, confidence scoring, and graceful dependency fallbacks.

Features:
- Named Entity Recognition using transformers/spaCy when available
- Pattern-based entity extraction with confidence scoring
- Entity type classification and disambiguation
- Coreference resolution and entity linking
- Comprehensive fallback handling for missing dependencies
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency availability flags
TRANSFORMERS_AVAILABLE = False
SPACY_AVAILABLE = False
TORCH_AVAILABLE = False

# Try importing optional dependencies with graceful fallbacks
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
    logger.info("✅ Transformers and PyTorch available - using advanced NER models")
except ImportError:
    logger.warning("⚠️  Transformers/PyTorch not available - using pattern-based extraction")
    # Create dummy classes for type hints
    class AutoTokenizer: pass
    class AutoModelForTokenClassification: pass
    def pipeline(*args, **kwargs): return None

try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy available - using linguistic analysis")
except ImportError:
    logger.warning("⚠️  spaCy not available - using basic text processing")
    spacy = None

# Try importing core modules with fallbacks
CORE_AVAILABLE = False
Entity = None
EntityInstance = None

try:
    from ..core.graph_schema import Entity, EntityInstance
    CORE_AVAILABLE = True
    logger.info("✅ Core schema modules available")
except (ImportError, AttributeError):
    logger.warning("⚠️  Core schema modules not available - using fallback classes")
    # Create fallback classes
    class Entity:
        def __init__(self, **kwargs): pass
    class EntityInstance:
        def __init__(self, **kwargs): pass


class EntityType(Enum):
    """Entity types for classification"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    DATE = "DATE"
    MONEY = "MONEY"
    QUANTITY = "QUANTITY"
    MISC = "MISC"
    UNKNOWN = "UNKNOWN"


class ExtractionMethod(Enum):
    """Methods used for entity extraction"""
    TRANSFORMER_NER = "transformer_ner"
    SPACY_NER = "spacy_ner"
    PATTERN_BASED = "pattern_based"
    CONTEXT_ANALYSIS = "context_analysis"
    COREFERENCE = "coreference"


@dataclass
class EntityCandidate:
    """Represents a candidate entity extracted from text"""
    entity_text: str
    entity_type: EntityType
    confidence: float
    start_position: int
    end_position: int
    context_window: str
    extraction_method: ExtractionMethod
    canonical_form: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    coreference_cluster: Optional[int] = None
    disambiguation_score: float = 0.0
    evidence_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'entity_text': self.entity_text,
            'entity_type': self.entity_type.value,
            'confidence': self.confidence,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'context_window': self.context_window,
            'extraction_method': self.extraction_method.value,
            'canonical_form': self.canonical_form,
            'aliases': self.aliases,
            'properties': self.properties,
            'coreference_cluster': self.coreference_cluster,
            'disambiguation_score': self.disambiguation_score,
            'evidence_text': self.evidence_text
        }
    
    def to_entity_instance(self) -> Optional['EntityInstance']:
        """Convert to an EntityInstance if core modules available"""
        if not CORE_AVAILABLE or EntityInstance is None:
            return None
        
        try:
            return EntityInstance(
                entity_id=self.canonical_form or self.entity_text,
                entity_type=self.entity_type.value,
                properties={
                    'text': self.entity_text,
                    'confidence': self.confidence,
                    'extraction_method': self.extraction_method.value,
                    'start_position': self.start_position,
                    'end_position': self.end_position,
                    'context_window': self.context_window,
                    'aliases': self.aliases,
                    'coreference_cluster': self.coreference_cluster,
                    'disambiguation_score': self.disambiguation_score,
                    **self.properties
                }
            )
        except Exception:
            # Fallback classes may not support initialization
            return None


@dataclass
class ExtractionContext:
    """Context for entity extraction operations"""
    text: str
    existing_entities: List[str] = field(default_factory=list)
    domain_context: Optional[str] = None
    language: str = "en"
    confidence_threshold: float = 0.5
    max_entity_length: int = 100
    enable_coreference: bool = True
    enable_disambiguation: bool = True


class EntityExtractor:
    """Advanced entity extraction with multiple strategies and confidence scoring"""
    
    def __init__(self):
        """Initialize the entity extractor with available models"""
        self.ner_pipeline = None
        self.spacy_nlp = None
        self.stats = {
            'total_extractions': 0,
            'high_confidence_extractions': 0,
            'transformer_extractions': 0,
            'spacy_extractions': 0,
            'pattern_extractions': 0,
            'coreference_resolutions': 0,
            'extraction_methods_enabled': {
                'transformer_ner': TRANSFORMERS_AVAILABLE,
                'spacy_ner': SPACY_AVAILABLE,
                'pattern_based': True,
                'context_analysis': True,
                'coreference': True
            },
            'models_available': {
                'transformers': TRANSFORMERS_AVAILABLE,
                'spacy': SPACY_AVAILABLE,
                'torch': TORCH_AVAILABLE
            }
        }
        
        # Initialize models if available
        self._initialize_models()
        
        # Entity patterns for pattern-based extraction
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. First Last
                r'\bProf\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Prof. First Last
                r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+ Inc\.?\b',  # Company Inc
                r'\b[A-Z][a-z]+ Corp\.?\b',  # Company Corp
                r'\b[A-Z][a-z]+ Ltd\.?\b',  # Company Ltd
                r'\b[A-Z][a-z]+ Company\b',  # Company Company
                r'\b[A-Z][a-z]+ Corporation\b',  # Company Corporation
                r'\b[A-Z]{2,}\b',  # Acronyms like IBM, NASA
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+ City\b',  # City names
                r'\b[A-Z][a-z]+, [A-Z]{2}\b',  # City, State
                r'\bMount [A-Z][a-z]+\b',  # Mountains
                r'\bLake [A-Z][a-z]+\b',  # Lakes
                r'\bRiver [A-Z][a-z]+\b',  # Rivers
            ],
            EntityType.TECHNOLOGY: [
                r'\bPython\b', r'\bJava\b', r'\bJavaScript\b',
                r'\bReact\b', r'\bAngular\b', r'\bVue\b',
                r'\bDocker\b', r'\bKubernetes\b', r'\bAWS\b',
                r'\bAPI\b', r'\bML\b', r'\bAI\b'
            ]
        }
        
        logger.info(f"🚀 EntityExtractor initialized with {len([k for k, v in self.stats['extraction_methods_enabled'].items() if v])} extraction methods")
    
    def _initialize_models(self):
        """Initialize NER models if dependencies are available"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight, fast NER model
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=-1  # Use CPU
                )
                logger.info("✅ Transformer NER pipeline initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize transformer NER: {e}")
                self.ner_pipeline = None
        
        if SPACY_AVAILABLE:
            try:
                # Try to load English model
                self.spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy NER model initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize spaCy NER: {e}")
                self.spacy_nlp = None
    
    def extract_entities(self, context: ExtractionContext) -> List[EntityCandidate]:
        """
        Extract entities using multiple strategies with confidence scoring
        
        Args:
            context: ExtractionContext with text and parameters
            
        Returns:
            List of EntityCandidate objects with confidence scores
        """
        logger.info(f"🔍 Extracting entities from text ({len(context.text)} chars)")
        
        all_candidates = []
        
        # 1. Pattern-based extraction (always available)
        pattern_candidates = self._extract_with_patterns(context)
        all_candidates.extend(pattern_candidates)
        
        # 2. Transformer-based NER (if available)
        if self.ner_pipeline and TRANSFORMERS_AVAILABLE:
            transformer_candidates = self._extract_with_transformer(context)
            all_candidates.extend(transformer_candidates)
        
        # 3. spaCy-based NER (if available)
        if self.spacy_nlp and SPACY_AVAILABLE:
            spacy_candidates = self._extract_with_spacy(context)
            all_candidates.extend(spacy_candidates)
        
        # 4. Context-based analysis
        context_candidates = self._extract_with_context(context)
        all_candidates.extend(context_candidates)
        
        # 5. Remove duplicates and merge similar candidates
        merged_candidates = self._merge_candidates(all_candidates)
        
        # 6. Coreference resolution (if enabled)
        if context.enable_coreference:
            merged_candidates = self._resolve_coreferences(merged_candidates, context)
        
        # 7. Filter by confidence threshold
        filtered_candidates = [
            c for c in merged_candidates 
            if c.confidence >= context.confidence_threshold
        ]
        
        # Update statistics
        self.stats['total_extractions'] += len(filtered_candidates)
        self.stats['high_confidence_extractions'] += len([
            c for c in filtered_candidates if c.confidence >= 0.8
        ])
        
        logger.info(f"✅ Extracted {len(filtered_candidates)} entities (confidence >= {context.confidence_threshold})")
        return filtered_candidates
    
    def _extract_with_patterns(self, context: ExtractionContext) -> List[EntityCandidate]:
        """Extract entities using pattern matching"""
        candidates = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, context.text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get context window (50 chars before and after)
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(context.text), end_pos + 50)
                    context_window = context.text[context_start:context_end]
                    
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_pattern_confidence(
                        entity_text, entity_type, context_window
                    )
                    
                    candidate = EntityCandidate(
                        entity_text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_position=start_pos,
                        end_position=end_pos,
                        context_window=context_window,
                        extraction_method=ExtractionMethod.PATTERN_BASED,
                        evidence_text=f"Pattern match: {pattern}",
                        canonical_form=self._canonicalize_entity(entity_text, entity_type)
                    )
                    
                    candidates.append(candidate)
        
        self.stats['pattern_extractions'] += len(candidates)
        logger.info(f"📝 Pattern-based extraction found {len(candidates)} candidates")
        return candidates
    
    def _extract_with_transformer(self, context: ExtractionContext) -> List[EntityCandidate]:
        """Extract entities using transformer-based NER"""
        if not self.ner_pipeline:
            return []
        
        candidates = []
        
        try:
            # Process text in chunks if too long
            max_length = 512
            text_chunks = [
                context.text[i:i+max_length] 
                for i in range(0, len(context.text), max_length)
            ]
            
            offset = 0
            for chunk in text_chunks:
                ner_results = self.ner_pipeline(chunk)
                
                for result in ner_results:
                    entity_text = result['word']
                    entity_type = self._map_ner_label(result['entity_group'])
                    confidence = result['score']
                    start_pos = result['start'] + offset
                    end_pos = result['end'] + offset
                    
                    # Get context window
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(context.text), end_pos + 50)
                    context_window = context.text[context_start:context_end]
                    
                    candidate = EntityCandidate(
                        entity_text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_position=start_pos,
                        end_position=end_pos,
                        context_window=context_window,
                        extraction_method=ExtractionMethod.TRANSFORMER_NER,
                        evidence_text=f"Transformer NER: {result['entity_group']}",
                        canonical_form=self._canonicalize_entity(entity_text, entity_type)
                    )
                    
                    candidates.append(candidate)
                
                offset += len(chunk)
        
        except Exception as e:
            logger.warning(f"⚠️  Transformer extraction failed: {e}")
        
        self.stats['transformer_extractions'] += len(candidates)
        logger.info(f"🤖 Transformer NER found {len(candidates)} candidates")
        return candidates
    
    def _extract_with_spacy(self, context: ExtractionContext) -> List[EntityCandidate]:
        """Extract entities using spaCy NER"""
        if not self.spacy_nlp:
            return []
        
        candidates = []
        
        try:
            doc = self.spacy_nlp(context.text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                confidence = 0.85  # spaCy doesn't provide confidence scores
                
                # Get context window
                context_start = max(0, ent.start_char - 50)
                context_end = min(len(context.text), ent.end_char + 50)
                context_window = context.text[context_start:context_end]
                
                candidate = EntityCandidate(
                    entity_text=ent.text,
                    entity_type=entity_type,
                    confidence=confidence,
                    start_position=ent.start_char,
                    end_position=ent.end_char,
                    context_window=context_window,
                    extraction_method=ExtractionMethod.SPACY_NER,
                    evidence_text=f"spaCy NER: {ent.label_}",
                    canonical_form=self._canonicalize_entity(ent.text, entity_type)
                )
                
                candidates.append(candidate)
        
        except Exception as e:
            logger.warning(f"⚠️  spaCy extraction failed: {e}")
        
        self.stats['spacy_extractions'] += len(candidates)
        logger.info(f"🔬 spaCy NER found {len(candidates)} candidates")
        return candidates
    
    def _extract_with_context(self, context: ExtractionContext) -> List[EntityCandidate]:
        """Extract entities using contextual analysis"""
        candidates = []
        
        # Context-based patterns (words that often indicate entity types)
        context_indicators = {
            EntityType.PERSON: ['CEO', 'founded by', 'director', 'president', 'manager'],
            EntityType.ORGANIZATION: ['company', 'corporation', 'organization', 'firm'],
            EntityType.LOCATION: ['located in', 'based in', 'headquarters', 'office'],
            EntityType.TECHNOLOGY: ['using', 'built with', 'powered by', 'framework']
        }
        
        for entity_type, indicators in context_indicators.items():
            for indicator in indicators:
                # Find indicator phrases and look for capitalized words nearby
                indicator_matches = re.finditer(
                    rf'\b{re.escape(indicator)}\b', 
                    context.text, 
                    re.IGNORECASE
                )
                
                for indicator_match in indicator_matches:
                    # Look for capitalized words within 20 words after the indicator
                    search_start = indicator_match.end()
                    search_end = min(len(context.text), search_start + 200)
                    search_text = context.text[search_start:search_end]
                    
                    # Find capitalized word sequences
                    cap_matches = re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', search_text)
                    
                    for cap_match in cap_matches:
                        entity_text = cap_match.group()
                        start_pos = search_start + cap_match.start()
                        end_pos = search_start + cap_match.end()
                        
                        # Get context window
                        context_start = max(0, start_pos - 50)
                        context_end = min(len(context.text), end_pos + 50)
                        context_window = context.text[context_start:context_end]
                        
                        # Calculate confidence based on indicator strength
                        confidence = 0.6 + (0.2 if len(entity_text.split()) > 1 else 0)
                        
                        candidate = EntityCandidate(
                            entity_text=entity_text,
                            entity_type=entity_type,
                            confidence=confidence,
                            start_position=start_pos,
                            end_position=end_pos,
                            context_window=context_window,
                            extraction_method=ExtractionMethod.CONTEXT_ANALYSIS,
                            evidence_text=f"Context indicator: {indicator}",
                            canonical_form=self._canonicalize_entity(entity_text, entity_type)
                        )
                        
                        candidates.append(candidate)
        
        logger.info(f"🔍 Context analysis found {len(candidates)} candidates")
        return candidates
    
    def _merge_candidates(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Merge duplicate and overlapping entity candidates"""
        if not candidates:
            return []
        
        # Group candidates by text and type
        groups = defaultdict(list)
        for candidate in candidates:
            key = (candidate.entity_text.lower(), candidate.entity_type)
            groups[key].append(candidate)
        
        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge multiple candidates for the same entity
                best_candidate = max(group, key=lambda c: c.confidence)
                
                # Combine extraction methods and evidence
                methods = set(c.extraction_method for c in group)
                evidence_texts = [c.evidence_text for c in group if c.evidence_text]
                
                best_candidate.confidence = min(1.0, best_candidate.confidence + 0.1 * (len(group) - 1))
                best_candidate.evidence_text = "; ".join(evidence_texts)
                best_candidate.properties['extraction_methods'] = [m.value for m in methods]
                
                merged.append(best_candidate)
        
        return merged
    
    def _resolve_coreferences(self, candidates: List[EntityCandidate], context: ExtractionContext) -> List[EntityCandidate]:
        """Simple coreference resolution for entity linking"""
        # Group entities by type and look for similar text
        type_groups = defaultdict(list)
        for candidate in candidates:
            type_groups[candidate.entity_type].append(candidate)
        
        cluster_id = 0
        for entity_type, group in type_groups.items():
            # Simple similarity-based clustering
            processed = set()
            
            for candidate in group:
                if id(candidate) in processed:
                    continue
                
                # Find similar entities
                similar = [candidate]
                for other in group:
                    if (id(other) not in processed and 
                        id(other) != id(candidate) and
                        self._are_entities_similar(candidate.entity_text, other.entity_text)):
                        similar.append(other)
                        processed.add(id(other))
                
                # Assign cluster ID if multiple similar entities found
                if len(similar) > 1:
                    for entity in similar:
                        entity.coreference_cluster = cluster_id
                    cluster_id += 1
                    self.stats['coreference_resolutions'] += len(similar) - 1
                
                processed.add(id(candidate))
        
        return candidates
    
    def _are_entities_similar(self, text1: str, text2: str) -> bool:
        """Check if two entity texts refer to the same entity"""
        # Simple similarity check
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Exact match
        if text1_lower == text2_lower:
            return True
        
        # One is contained in the other
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return True
        
        # Check for common abbreviations
        words1 = text1.split()
        words2 = text2.split()
        
        # Acronym check
        if len(words1) > 1 and len(words2) == 1:
            acronym = ''.join(w[0].upper() for w in words1)
            if acronym == text2.upper():
                return True
        
        return False
    
    def _calculate_pattern_confidence(self, entity_text: str, entity_type: EntityType, context: str) -> float:
        """Calculate confidence score for pattern-based extractions"""
        base_confidence = 0.7
        
        # Boost confidence for longer entities
        if len(entity_text.split()) > 1:
            base_confidence += 0.1
        
        # Boost confidence for proper capitalization
        if entity_text.istitle():
            base_confidence += 0.05
        
        # Boost confidence based on context indicators
        context_lower = context.lower()
        type_indicators = {
            EntityType.PERSON: ['dr.', 'prof.', 'ceo', 'president'],
            EntityType.ORGANIZATION: ['inc', 'corp', 'ltd', 'company'],
            EntityType.LOCATION: ['city', 'state', 'country', 'located']
        }
        
        if entity_type in type_indicators:
            for indicator in type_indicators[entity_type]:
                if indicator in context_lower:
                    base_confidence += 0.1
                    break
        
        return min(1.0, base_confidence)
    
    def _map_ner_label(self, label: str) -> EntityType:
        """Map NER model labels to our EntityType enum"""
        mapping = {
            'PER': EntityType.PERSON,
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'LOC': EntityType.LOCATION,
            'MISC': EntityType.MISC
        }
        return mapping.get(label.upper(), EntityType.UNKNOWN)
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy NER labels to our EntityType enum"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,  # Geopolitical entity
            'LOC': EntityType.LOCATION,
            'EVENT': EntityType.EVENT,
            'PRODUCT': EntityType.PRODUCT,
            'DATE': EntityType.DATE,
            'MONEY': EntityType.MONEY,
            'QUANTITY': EntityType.QUANTITY
        }
        return mapping.get(label.upper(), EntityType.UNKNOWN)
    
    def _canonicalize_entity(self, entity_text: str, entity_type: EntityType) -> str:
        """Generate canonical form of entity text"""
        # Basic canonicalization - remove extra whitespace and normalize case
        canonical = ' '.join(entity_text.split())
        
        # Entity-type specific canonicalization
        if entity_type == EntityType.ORGANIZATION:
            # Normalize company suffixes
            canonical = re.sub(r'\b(Inc|Corp|Ltd|LLC)\.?$', lambda m: m.group(1).upper() + '.', canonical)
        
        elif entity_type == EntityType.PERSON:
            # Ensure proper title case for names
            canonical = canonical.title()
        
        return canonical
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics and model availability"""
        return self.stats.copy()
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for entity filtering"""
        if 0.0 <= new_threshold <= 1.0:
            self.stats['confidence_threshold'] = new_threshold
            logger.info(f"✅ Updated confidence threshold to {new_threshold}")
        else:
            logger.warning(f"⚠️  Invalid confidence threshold: {new_threshold}")


# Convenience function for quick entity extraction
def extract_entities_quick(text: str, confidence_threshold: float = 0.5) -> List[EntityCandidate]:
    """
    Quick entity extraction with default settings
    
    Args:
        text: Input text for entity extraction
        confidence_threshold: Minimum confidence for entity candidates
        
    Returns:
        List of EntityCandidate objects
    """
    extractor = EntityExtractor()
    context = ExtractionContext(
        text=text,
        confidence_threshold=confidence_threshold
    )
    return extractor.extract_entities(context)


if __name__ == "__main__":
    # Example usage and basic testing
    sample_text = """
    Dr. Sarah Chen works for Microsoft Corporation in Seattle, Washington. 
    She founded the AI Research Lab using Python and TensorFlow frameworks.
    The company was established by Bill Gates in 1975 and is headquartered in Redmond.
    """
    
    print("🧪 Testing Enhanced Entity Extraction")
    print("=" * 50)
    
    # Test quick extraction
    entities = extract_entities_quick(sample_text, confidence_threshold=0.6)
    
    print(f"✅ Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  📝 {entity.entity_text} ({entity.entity_type.value}) - {entity.confidence:.3f}")
        print(f"     Method: {entity.extraction_method.value}")
        print(f"     Context: ...{entity.context_window[:100]}...")
        print()
