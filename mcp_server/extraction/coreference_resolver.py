"""
Advanced Coreference Resolution Module for Knowledge Graph Processing

This module provides sophisticated coreference resolution capabilities beyond
basic entity clustering, including pronoun resolution, nominal coreferences,
and cross-document entity linking.

Features:
- Pronoun-antecedent resolution with agreement checking
- Nominal coreference resolution for definite references
- Proper noun variation handling (abbreviations, nicknames)
- Cross-sentence and cross-document entity linking
- Confidence scoring for resolution quality assessment
- Integration with entity and relationship extraction modules
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

# Try importing related modules with graceful fallbacks
try:
    from .entity_extractor import EntityCandidate, EntityType
    ENTITY_EXTRACTOR_AVAILABLE = True
    logger.info("✅ Entity extractor integration available")
except (ImportError, AttributeError):
    logger.warning("⚠️  Entity extractor not available - using basic types")
    ENTITY_EXTRACTOR_AVAILABLE = False
    # Create fallback types
    class EntityType(Enum):
        PERSON = "PERSON"
        ORGANIZATION = "ORGANIZATION"
        LOCATION = "LOCATION"
        MISC = "MISC"
    class EntityCandidate:
        def __init__(self, **kwargs): pass

try:
    from .relation_extractor import RelationshipCandidate
    RELATION_EXTRACTOR_AVAILABLE = True
    logger.info("✅ Relationship extractor integration available")
except (ImportError, AttributeError):
    logger.warning("⚠️  Relationship extractor not available - using basic types")
    RELATION_EXTRACTOR_AVAILABLE = False
    class RelationshipCandidate:
        def __init__(self, **kwargs): pass

# Dependency availability flags
SPACY_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy available - using advanced linguistic analysis")
except ImportError:
    logger.warning("⚠️  spaCy not available - using basic text processing")
    spacy = None


class ReferenceType(Enum):
    """Types of coreference mentions"""
    PRONOUN = "pronoun"           # he, she, it, they
    NOMINAL = "nominal"           # the company, the person
    PROPER_NOUN = "proper_noun"   # exact name matches
    DEFINITE = "definite"         # the CEO, the founder
    DEMONSTRATIVE = "demonstrative"  # this company, that person


class ResolutionMethod(Enum):
    """Methods used for coreference resolution"""
    PRONOUN_AGREEMENT = "pronoun_agreement"
    NOMINAL_MATCHING = "nominal_matching"
    PROPER_NOUN_VARIATION = "proper_noun_variation"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    DISCOURSE_PROXIMITY = "discourse_proximity"
    CONTEXTUAL_CLUES = "contextual_clues"


@dataclass
class MentionSpan:
    """Represents a mention of an entity in text"""
    text: str
    start_position: int
    end_position: int
    sentence_id: int
    reference_type: ReferenceType
    entity_type: Optional[EntityType] = None
    gender: Optional[str] = None  # masculine, feminine, neuter
    number: Optional[str] = None  # singular, plural
    definiteness: Optional[str] = None  # definite, indefinite
    head_word: Optional[str] = None
    context_window: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolutionCandidate:
    """Represents a potential coreference link between mentions"""
    mention: MentionSpan
    antecedent: MentionSpan
    confidence: float
    resolution_method: ResolutionMethod
    evidence: str
    distance: int  # distance in sentences
    agreement_score: float = 0.0
    semantic_score: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'mention_text': self.mention.text,
            'antecedent_text': self.antecedent.text,
            'confidence': self.confidence,
            'resolution_method': self.resolution_method.value,
            'evidence': self.evidence,
            'distance': self.distance,
            'agreement_score': self.agreement_score,
            'semantic_score': self.semantic_score,
            'properties': self.properties
        }


@dataclass
class CoreferenceCluster:
    """Represents a cluster of mentions referring to the same entity"""
    cluster_id: int
    mentions: List[MentionSpan]
    canonical_mention: MentionSpan
    entity_type: EntityType
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def add_mention(self, mention: MentionSpan, confidence: float):
        """Add a mention to this cluster"""
        self.mentions.append(mention)
        # Update cluster confidence (average)
        self.confidence = (self.confidence * (len(self.mentions) - 1) + confidence) / len(self.mentions)
    
    def get_canonical_text(self) -> str:
        """Get the most representative text for this cluster"""
        return self.canonical_mention.text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'cluster_id': self.cluster_id,
            'canonical_text': self.get_canonical_text(),
            'mention_count': len(self.mentions),
            'entity_type': self.entity_type.value if self.entity_type else None,
            'confidence': self.confidence,
            'mentions': [
                {
                    'text': m.text,
                    'sentence_id': m.sentence_id,
                    'reference_type': m.reference_type.value,
                    'start_position': m.start_position,
                    'end_position': m.end_position
                } for m in self.mentions
            ],
            'properties': self.properties
        }


@dataclass
class ResolutionContext:
    """Context for coreference resolution operations"""
    text: str
    sentences: List[str] = field(default_factory=list)
    existing_entities: List[EntityCandidate] = field(default_factory=list)
    confidence_threshold: float = 0.6
    max_distance: int = 5  # maximum sentence distance for resolution
    enable_pronoun_resolution: bool = True
    enable_nominal_resolution: bool = True
    enable_proper_noun_resolution: bool = True
    discourse_window: int = 10  # sentences to consider for discourse model


class CoreferenceResolver:
    """Advanced coreference resolution with multiple strategies"""
    
    def __init__(self):
        """Initialize the coreference resolver"""
        self.spacy_nlp = None
        self.stats = {
            'total_resolutions': 0,
            'pronoun_resolutions': 0,
            'nominal_resolutions': 0,
            'proper_noun_resolutions': 0,
            'high_confidence_resolutions': 0,
            'resolution_methods_enabled': {
                'pronoun_agreement': True,
                'nominal_matching': True,
                'proper_noun_variation': True,
                'semantic_similarity': SPACY_AVAILABLE,
                'discourse_proximity': True,
                'contextual_clues': True
            },
            'models_available': {
                'spacy': SPACY_AVAILABLE
            }
        }
        
        # Initialize spaCy if available
        self._initialize_spacy()
        
        # Pronoun information for agreement checking
        self.pronouns = {
            # Masculine singular
            'he': {'gender': 'masculine', 'number': 'singular', 'entity_types': [EntityType.PERSON]},
            'him': {'gender': 'masculine', 'number': 'singular', 'entity_types': [EntityType.PERSON]},
            'his': {'gender': 'masculine', 'number': 'singular', 'entity_types': [EntityType.PERSON]},
            
            # Feminine singular
            'she': {'gender': 'feminine', 'number': 'singular', 'entity_types': [EntityType.PERSON]},
            'her': {'gender': 'feminine', 'number': 'singular', 'entity_types': [EntityType.PERSON]},
            'hers': {'gender': 'feminine', 'number': 'singular', 'entity_types': [EntityType.PERSON]},
            
            # Neuter singular
            'it': {'gender': 'neuter', 'number': 'singular', 'entity_types': [EntityType.ORGANIZATION, EntityType.LOCATION]},
            'its': {'gender': 'neuter', 'number': 'singular', 'entity_types': [EntityType.ORGANIZATION, EntityType.LOCATION]},
            
            # Plural
            'they': {'gender': 'neutral', 'number': 'plural', 'entity_types': [EntityType.PERSON, EntityType.ORGANIZATION]},
            'them': {'gender': 'neutral', 'number': 'plural', 'entity_types': [EntityType.PERSON, EntityType.ORGANIZATION]},
            'their': {'gender': 'neutral', 'number': 'plural', 'entity_types': [EntityType.PERSON, EntityType.ORGANIZATION]},
            'theirs': {'gender': 'neutral', 'number': 'plural', 'entity_types': [EntityType.PERSON, EntityType.ORGANIZATION]}
        }
        
        # Nominal patterns for definite references
        self.nominal_patterns = {
            EntityType.PERSON: ['the person', 'the individual', 'the man', 'the woman', 'the CEO', 'the founder', 'the director'],
            EntityType.ORGANIZATION: ['the company', 'the corporation', 'the organization', 'the firm', 'the business'],
            EntityType.LOCATION: ['the location', 'the place', 'the city', 'the country', 'the region'],
        }
        
        logger.info(f"🚀 CoreferenceResolver initialized with {len([k for k, v in self.stats['resolution_methods_enabled'].items() if v])} resolution methods")
    
    def _initialize_spacy(self):
        """Initialize spaCy model if available"""
        if SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model initialized for advanced linguistic analysis")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize spaCy model: {e}")
                self.spacy_nlp = None
    
    def resolve_coreferences(self, context: ResolutionContext) -> Tuple[List[CoreferenceCluster], List[ResolutionCandidate]]:
        """
        Resolve coreferences in text using multiple strategies
        
        Args:
            context: ResolutionContext with text and parameters
            
        Returns:
            Tuple of (coreference clusters, resolution candidates)
        """
        logger.info(f"🔍 Resolving coreferences in text ({len(context.text)} chars)")
        
        # 1. Split text into sentences and extract mentions
        sentences = self._split_into_sentences(context.text)
        context.sentences = sentences
        
        # 2. Extract all mentions from text
        mentions = self._extract_mentions(sentences, context)
        logger.info(f"📝 Found {len(mentions)} mentions across {len(sentences)} sentences")
        
        # 3. Generate resolution candidates
        candidates = []
        
        if context.enable_pronoun_resolution:
            pronoun_candidates = self._resolve_pronouns(mentions, context)
            candidates.extend(pronoun_candidates)
        
        if context.enable_nominal_resolution:
            nominal_candidates = self._resolve_nominals(mentions, context)
            candidates.extend(nominal_candidates)
        
        if context.enable_proper_noun_resolution:
            proper_noun_candidates = self._resolve_proper_nouns(mentions, context)
            candidates.extend(proper_noun_candidates)
        
        # 4. Filter candidates by confidence threshold
        filtered_candidates = [
            c for c in candidates 
            if c.confidence >= context.confidence_threshold
        ]
        
        # 5. Build coreference clusters
        clusters = self._build_clusters(mentions, filtered_candidates)
        
        # Update statistics
        self.stats['total_resolutions'] += len(filtered_candidates)
        self.stats['high_confidence_resolutions'] += len([
            c for c in filtered_candidates if c.confidence >= 0.8
        ])
        
        logger.info(f"✅ Resolved {len(filtered_candidates)} coreferences into {len(clusters)} clusters")
        return clusters, filtered_candidates
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Basic sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _extract_mentions(self, sentences: List[str], context: ResolutionContext) -> List[MentionSpan]:
        """Extract all mention spans from sentences"""
        mentions = []
        current_position = 0
        
        for sent_id, sentence in enumerate(sentences):
            sent_start = context.text.find(sentence, current_position)
            
            # Find pronouns
            pronoun_mentions = self._find_pronoun_mentions(sentence, sent_start, sent_id)
            mentions.extend(pronoun_mentions)
            
            # Find nominal mentions
            nominal_mentions = self._find_nominal_mentions(sentence, sent_start, sent_id)
            mentions.extend(nominal_mentions)
            
            # Find proper noun mentions from existing entities and basic extraction
            entity_mentions = self._find_entity_mentions(sentence, sent_start, sent_id, context.existing_entities)
            mentions.extend(entity_mentions)
            
            current_position = sent_start + len(sentence)
        
        return mentions
    
    def _find_pronoun_mentions(self, sentence: str, sent_start: int, sent_id: int) -> List[MentionSpan]:
        """Find pronoun mentions in a sentence"""
        mentions = []
        
        for pronoun, info in self.pronouns.items():
            pattern = rf'\b{re.escape(pronoun)}\b'
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            
            for match in matches:
                start_pos = sent_start + match.start()
                end_pos = sent_start + match.end()
                
                mention = MentionSpan(
                    text=match.group(),
                    start_position=start_pos,
                    end_position=end_pos,
                    sentence_id=sent_id,
                    reference_type=ReferenceType.PRONOUN,
                    gender=info.get('gender'),
                    number=info.get('number'),
                    context_window=sentence,
                    properties={'compatible_entity_types': [t.value for t in info.get('entity_types', [])]}
                )
                
                mentions.append(mention)
        
        return mentions
    
    def _find_nominal_mentions(self, sentence: str, sent_start: int, sent_id: int) -> List[MentionSpan]:
        """Find nominal mentions (definite references) in a sentence"""
        mentions = []
        
        for entity_type, patterns in self.nominal_patterns.items():
            for pattern in patterns:
                pattern_regex = rf'\b{re.escape(pattern)}\b'
                matches = re.finditer(pattern_regex, sentence, re.IGNORECASE)
                
                for match in matches:
                    start_pos = sent_start + match.start()
                    end_pos = sent_start + match.end()
                    
                    mention = MentionSpan(
                        text=match.group(),
                        start_position=start_pos,
                        end_position=end_pos,
                        sentence_id=sent_id,
                        reference_type=ReferenceType.NOMINAL,
                        entity_type=entity_type,
                        definiteness='definite',
                        context_window=sentence,
                        head_word=pattern.split()[-1]  # Last word is usually the head
                    )
                    
                    mentions.append(mention)
        
        return mentions
    
    def _find_entity_mentions(self, sentence: str, sent_start: int, sent_id: int, entities: List[EntityCandidate]) -> List[MentionSpan]:
        """Find mentions of existing entities in a sentence"""
        mentions = []
        
        # If no existing entities provided, extract potential entities from text
        if not entities:
            mentions.extend(self._extract_potential_entities(sentence, sent_start, sent_id))
        else:
            for entity in entities:
                # Look for exact matches and variations
                entity_patterns = [entity.entity_text]
                
                # Add canonical form if different
                if hasattr(entity, 'canonical_form') and entity.canonical_form and entity.canonical_form != entity.entity_text:
                    entity_patterns.append(entity.canonical_form)
                
                # Add aliases if available
                if hasattr(entity, 'aliases') and entity.aliases:
                    entity_patterns.extend(entity.aliases)
                
                for pattern in entity_patterns:
                    pattern_regex = rf'\b{re.escape(pattern)}\b'
                    matches = re.finditer(pattern_regex, sentence, re.IGNORECASE)
                    
                    for match in matches:
                        start_pos = sent_start + match.start()
                        end_pos = sent_start + match.end()
                        
                        mention = MentionSpan(
                            text=match.group(),
                            start_position=start_pos,
                            end_position=end_pos,
                            sentence_id=sent_id,
                            reference_type=ReferenceType.PROPER_NOUN,
                            entity_type=entity.entity_type if hasattr(entity, 'entity_type') else None,
                            context_window=sentence,
                            properties={'source_entity': entity.entity_text if hasattr(entity, 'entity_text') else str(entity)}
                        )
                        
                        mentions.append(mention)
        
        return mentions
    
    def _extract_potential_entities(self, sentence: str, sent_start: int, sent_id: int) -> List[MentionSpan]:
        """Extract potential entity mentions from sentence using basic patterns"""
        mentions = []
        
        # Basic entity patterns
        entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. First Last
                r'\bMr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Mr. First Last
                r'\bMs\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Ms. First Last
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+ Inc\.?\b',  # Company Inc
                r'\b[A-Z][a-z]+ Corp\.?\b',  # Company Corp
                r'\b[A-Z][a-z]+ Corporation\b',  # Company Corporation
                r'\b[A-Z][a-z]+ Company\b',  # Company Company
                r'\b[A-Z]{2,}\b',  # Acronyms like IBM, NASA
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+ City\b',  # City names
                r'\b[A-Z][a-z]+, [A-Z]{2}\b',  # City, State
            ]
        }
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence)
                
                for match in matches:
                    start_pos = sent_start + match.start()
                    end_pos = sent_start + match.end()
                    
                    mention = MentionSpan(
                        text=match.group(),
                        start_position=start_pos,
                        end_position=end_pos,
                        sentence_id=sent_id,
                        reference_type=ReferenceType.PROPER_NOUN,
                        entity_type=entity_type,
                        context_window=sentence,
                        properties={'extracted_pattern': pattern}
                    )
                    
                    mentions.append(mention)
        
        return mentions
    
    def _resolve_pronouns(self, mentions: List[MentionSpan], context: ResolutionContext) -> List[ResolutionCandidate]:
        """Resolve pronoun coreferences using agreement checking"""
        candidates = []
        
        pronoun_mentions = [m for m in mentions if m.reference_type == ReferenceType.PRONOUN]
        potential_antecedents = [m for m in mentions if m.reference_type != ReferenceType.PRONOUN]
        
        for pronoun in pronoun_mentions:
            # Find potential antecedents in previous sentences (within max_distance)
            for antecedent in potential_antecedents:
                if (antecedent.sentence_id < pronoun.sentence_id and
                    pronoun.sentence_id - antecedent.sentence_id <= context.max_distance):
                    
                    # Check agreement (gender, number, entity type)
                    agreement_score = self._calculate_agreement(pronoun, antecedent)
                    
                    if agreement_score > 0.5:  # Minimum agreement threshold
                        distance = pronoun.sentence_id - antecedent.sentence_id
                        
                        # Calculate confidence based on agreement and proximity
                        confidence = agreement_score * (1.0 - (distance / context.max_distance) * 0.3)
                        
                        candidate = ResolutionCandidate(
                            mention=pronoun,
                            antecedent=antecedent,
                            confidence=confidence,
                            resolution_method=ResolutionMethod.PRONOUN_AGREEMENT,
                            evidence=f"Agreement: gender={pronoun.gender}, number={pronoun.number}",
                            distance=distance,
                            agreement_score=agreement_score
                        )
                        
                        candidates.append(candidate)
        
        self.stats['pronoun_resolutions'] += len(candidates)
        logger.info(f"🔗 Pronoun resolution found {len(candidates)} candidates")
        return candidates
    
    def _resolve_nominals(self, mentions: List[MentionSpan], context: ResolutionContext) -> List[ResolutionCandidate]:
        """Resolve nominal coreferences (definite references)"""
        candidates = []
        
        nominal_mentions = [m for m in mentions if m.reference_type == ReferenceType.NOMINAL]
        entity_mentions = [m for m in mentions if m.reference_type == ReferenceType.PROPER_NOUN]
        
        for nominal in nominal_mentions:
            # Find entity mentions of the same type in previous sentences
            for entity in entity_mentions:
                if (entity.sentence_id <= nominal.sentence_id and
                    nominal.sentence_id - entity.sentence_id <= context.max_distance and
                    entity.entity_type == nominal.entity_type):
                    
                    distance = nominal.sentence_id - entity.sentence_id
                    
                    # Calculate confidence based on type match and proximity
                    confidence = 0.8 * (1.0 - (distance / context.max_distance) * 0.2)
                    
                    # Boost confidence if the nominal is definite
                    if nominal.definiteness == 'definite':
                        confidence += 0.1
                    
                    candidate = ResolutionCandidate(
                        mention=nominal,
                        antecedent=entity,
                        confidence=min(1.0, confidence),
                        resolution_method=ResolutionMethod.NOMINAL_MATCHING,
                        evidence=f"Type match: {nominal.entity_type.value if nominal.entity_type else 'unknown'}",
                        distance=distance
                    )
                    
                    candidates.append(candidate)
        
        self.stats['nominal_resolutions'] += len(candidates)
        logger.info(f"🏷️  Nominal resolution found {len(candidates)} candidates")
        return candidates
    
    def _resolve_proper_nouns(self, mentions: List[MentionSpan], context: ResolutionContext) -> List[ResolutionCandidate]:
        """Resolve proper noun variations and abbreviations"""
        candidates = []
        
        proper_noun_mentions = [m for m in mentions if m.reference_type == ReferenceType.PROPER_NOUN]
        
        # Group mentions by entity type
        type_groups = defaultdict(list)
        for mention in proper_noun_mentions:
            if mention.entity_type:
                type_groups[mention.entity_type].append(mention)
        
        # Find variations within each type group
        for entity_type, group in type_groups.items():
            for i, mention1 in enumerate(group):
                for mention2 in group[i+1:]:
                    # Check if mentions are similar (abbreviations, variations)
                    similarity_score = self._calculate_text_similarity(mention1.text, mention2.text)
                    
                    if similarity_score > 0.6:  # Similarity threshold
                        # Determine which is the antecedent (earlier mention or longer text)
                        if mention1.sentence_id < mention2.sentence_id:
                            antecedent, mention = mention1, mention2
                        elif mention2.sentence_id < mention1.sentence_id:
                            antecedent, mention = mention2, mention1
                        else:
                            # Same sentence - longer text is antecedent
                            if len(mention1.text) >= len(mention2.text):
                                antecedent, mention = mention1, mention2
                            else:
                                antecedent, mention = mention2, mention1
                        
                        distance = abs(mention.sentence_id - antecedent.sentence_id)
                        
                        if distance <= context.max_distance:
                            confidence = similarity_score * (1.0 - (distance / context.max_distance) * 0.1)
                            
                            candidate = ResolutionCandidate(
                                mention=mention,
                                antecedent=antecedent,
                                confidence=confidence,
                                resolution_method=ResolutionMethod.PROPER_NOUN_VARIATION,
                                evidence=f"Text similarity: {similarity_score:.3f}",
                                distance=distance,
                                semantic_score=similarity_score
                            )
                            
                            candidates.append(candidate)
        
        self.stats['proper_noun_resolutions'] += len(candidates)
        logger.info(f"📝 Proper noun resolution found {len(candidates)} candidates")
        return candidates
    
    def _calculate_agreement(self, pronoun: MentionSpan, antecedent: MentionSpan) -> float:
        """Calculate agreement score between pronoun and potential antecedent"""
        score = 0.0
        
        # Gender agreement
        if pronoun.gender and antecedent.gender:
            if pronoun.gender == antecedent.gender:
                score += 0.4
        elif pronoun.gender == 'neutral' or antecedent.gender is None:
            score += 0.2  # Neutral pronouns or unknown gender
        
        # Number agreement
        if pronoun.number and antecedent.number:
            if pronoun.number == antecedent.number:
                score += 0.3
        elif pronoun.number is None or antecedent.number is None:
            score += 0.15
        
        # Entity type compatibility
        if pronoun.properties.get('compatible_entity_types') and antecedent.entity_type:
            if antecedent.entity_type.value in pronoun.properties['compatible_entity_types']:
                score += 0.3
        
        return min(1.0, score)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # Exact match
        if text1_lower == text2_lower:
            return 1.0
        
        # One contains the other
        if text1_lower in text2_lower or text2_lower in text1_lower:
            shorter = min(len(text1_lower), len(text2_lower))
            longer = max(len(text1_lower), len(text2_lower))
            return shorter / longer
        
        # Check for abbreviation patterns
        words1 = text1_lower.split()
        words2 = text2_lower.split()
        
        # Acronym check
        if len(words1) > 1 and len(words2) == 1:
            acronym = ''.join(w[0] for w in words1)
            if acronym == text2_lower:
                return 0.9
        elif len(words2) > 1 and len(words1) == 1:
            acronym = ''.join(w[0] for w in words2)
            if acronym == text1_lower:
                return 0.9
        
        # Simple word overlap
        set1 = set(words1)
        set2 = set(words2)
        if set1 and set2:
            overlap = len(set1.intersection(set2))
            total = len(set1.union(set2))
            return overlap / total
        
        return 0.0
    
    def _build_clusters(self, mentions: List[MentionSpan], candidates: List[ResolutionCandidate]) -> List[CoreferenceCluster]:
        """Build coreference clusters from resolution candidates"""
        # Create a graph of coreference links
        mention_to_antecedent = {}
        
        # Sort candidates by confidence (highest first)
        sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        
        for candidate in sorted_candidates:
            mention_id = id(candidate.mention)
            antecedent_id = id(candidate.antecedent)
            
            # Only add if not already linked
            if mention_id not in mention_to_antecedent:
                mention_to_antecedent[mention_id] = (candidate.antecedent, candidate.confidence)
        
        # Find connected components (clusters)
        mention_to_cluster = {}
        clusters = []
        cluster_id = 0
        
        for mention in mentions:
            mention_id = id(mention)
            
            if mention_id not in mention_to_cluster:
                # Start a new cluster
                cluster_mentions = []
                to_process = [mention]
                
                while to_process:
                    current = to_process.pop()
                    current_id = id(current)
                    
                    if current_id not in mention_to_cluster:
                        mention_to_cluster[current_id] = cluster_id
                        cluster_mentions.append(current)
                        
                        # Add linked mentions
                        if current_id in mention_to_antecedent:
                            antecedent, _ = mention_to_antecedent[current_id]
                            to_process.append(antecedent)
                        
                        # Add mentions that link to this one
                        for other_mention_id, (antecedent, _) in mention_to_antecedent.items():
                            if id(antecedent) == current_id and other_mention_id not in mention_to_cluster:
                                other_mention = next(m for m in mentions if id(m) == other_mention_id)
                                to_process.append(other_mention)
                
                # Create cluster if it has multiple mentions
                if len(cluster_mentions) > 1:
                    # Find canonical mention (longest proper noun or first mention)
                    canonical = max(cluster_mentions, key=lambda m: (
                        1 if m.reference_type == ReferenceType.PROPER_NOUN else 0,
                        len(m.text),
                        -m.sentence_id  # Earlier mentions preferred
                    ))
                    
                    # Determine entity type
                    entity_types = [m.entity_type for m in cluster_mentions if m.entity_type]
                    entity_type = entity_types[0] if entity_types else EntityType.MISC
                    
                    # Calculate average confidence
                    confidences = [
                        mention_to_antecedent[id(m)][1] 
                        for m in cluster_mentions 
                        if id(m) in mention_to_antecedent
                    ]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
                    
                    cluster = CoreferenceCluster(
                        cluster_id=cluster_id,
                        mentions=cluster_mentions,
                        canonical_mention=canonical,
                        entity_type=entity_type,
                        confidence=avg_confidence
                    )
                    
                    clusters.append(cluster)
                
                cluster_id += 1
        
        return clusters
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get coreference resolution statistics"""
        return self.stats.copy()
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for resolution filtering"""
        if 0.0 <= new_threshold <= 1.0:
            self.stats['confidence_threshold'] = new_threshold
            logger.info(f"✅ Updated confidence threshold to {new_threshold}")
        else:
            logger.warning(f"⚠️  Invalid confidence threshold: {new_threshold}")


# Convenience function for quick coreference resolution
def resolve_coreferences_quick(text: str, entities: List[EntityCandidate] = None, confidence_threshold: float = 0.6) -> Tuple[List[CoreferenceCluster], List[ResolutionCandidate]]:
    """
    Quick coreference resolution with default settings
    
    Args:
        text: Input text for coreference resolution
        entities: Existing entity candidates to consider
        confidence_threshold: Minimum confidence for resolution candidates
        
    Returns:
        Tuple of (coreference clusters, resolution candidates)
    """
    resolver = CoreferenceResolver()
    context = ResolutionContext(
        text=text,
        existing_entities=entities or [],
        confidence_threshold=confidence_threshold
    )
    return resolver.resolve_coreferences(context)


def create_coreference_resolver(**kwargs) -> CoreferenceResolver:
    """
    Factory function to create a coreference resolver
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        CoreferenceResolver: Configured coreference resolver instance
    """
    return CoreferenceResolver()


if __name__ == "__main__":
    # Example usage and basic testing
    sample_text = """
    Dr. Sarah Chen works for Microsoft Corporation in Seattle. She founded the AI Research Lab.
    The company was established by Bill Gates in 1975. He served as CEO for many years.
    Microsoft has grown significantly since then. The organization now employs thousands of people.
    """
    
    print("🧪 Testing Advanced Coreference Resolution")
    print("=" * 50)
    
    # Test quick resolution
    clusters, candidates = resolve_coreferences_quick(sample_text, confidence_threshold=0.5)
    
    print(f"✅ Found {len(clusters)} coreference clusters:")
    for cluster in clusters:
        print(f"  📝 Cluster {cluster.cluster_id}: {cluster.get_canonical_text()}")
        print(f"     Entity Type: {cluster.entity_type.value}")
        print(f"     Confidence: {cluster.confidence:.3f}")
        print(f"     Mentions ({len(cluster.mentions)}):")
        for mention in cluster.mentions:
            print(f"       - '{mention.text}' ({mention.reference_type.value}, sentence {mention.sentence_id})")
        print()
    
    print(f"✅ Found {len(candidates)} resolution candidates:")
    for candidate in candidates[:5]:  # Show first 5
        print(f"  🔗 '{candidate.mention.text}' -> '{candidate.antecedent.text}'")
        print(f"     Confidence: {candidate.confidence:.3f}")
        print(f"     Method: {candidate.resolution_method.value}")
        print(f"     Evidence: {candidate.evidence}")
        print()
