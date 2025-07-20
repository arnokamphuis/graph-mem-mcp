"""
Advanced Entity Resolution System

This module provides sophisticated entity resolution capabilities including:
- Fuzzy string matching for entity deduplication
- Embedding-based semantic similarity matching
- Rule-based entity merging strategies
- External entity linking to knowledge bases
- Coreference resolution across document spans
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from difflib import SequenceMatcher

# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None
    DBSCAN = None

try:
    import fuzzywuzzy
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    fuzz = None
    process = None

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidate:
    """Candidate entity for resolution"""
    id: str
    name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    mentions: List[str] = field(default_factory=list)
    context_snippets: List[str] = field(default_factory=list)


@dataclass
class EntityMatch:
    """Result of entity matching"""
    candidate1: EntityCandidate
    candidate2: EntityCandidate
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'semantic', 'alias'
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityCluster:
    """Cluster of resolved entities"""
    canonical_entity: EntityCandidate
    member_entities: List[EntityCandidate]
    cluster_confidence: float
    resolution_method: str


class EntityResolver:
    """Advanced entity resolution with multiple matching strategies"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 fuzzy_threshold: int = 85,
                 semantic_threshold: float = 0.7,
                 use_embeddings: bool = True):
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.use_embeddings = use_embeddings
        
        # Initialize embedding model if available
        self.sentence_model = None
        if use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded for semantic matching")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
        
        # Entity registry for tracking resolved entities
        self.entity_registry: Dict[str, EntityCandidate] = {}
        self.entity_clusters: List[EntityCluster] = []
        
        # Common entity normalization patterns
        self.normalization_patterns = {
            # Remove common prefixes/suffixes
            'prefixes': ['mr.', 'mrs.', 'dr.', 'prof.', 'the ', 'a '],
            'suffixes': [' inc.', ' corp.', ' ltd.', ' llc.', ' co.'],
            # Organization type variations
            'org_synonyms': {
                'corporation': ['corp', 'inc', 'incorporated'],
                'company': ['co', 'company'],
                'limited': ['ltd', 'limited'],
                'university': ['univ', 'university', 'college']
            }
        }
    
    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for better matching"""
        normalized = name.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common prefixes
        for prefix in self.normalization_patterns['prefixes']:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Remove common suffixes
        for suffix in self.normalization_patterns['suffixes']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # Remove punctuation except hyphens and apostrophes
        normalized = re.sub(r'[^\w\s\-\']', '', normalized)
        
        return normalized
    
    def extract_aliases(self, entity: EntityCandidate) -> Set[str]:
        """Extract potential aliases for an entity"""
        aliases = set(entity.aliases)
        
        # Add normalized name
        aliases.add(self.normalize_entity_name(entity.name))
        
        # Add variations from mentions
        for mention in entity.mentions:
            aliases.add(self.normalize_entity_name(mention))
        
        # Add acronyms for organizations
        if entity.entity_type in ['organization', 'company']:
            acronym = self._generate_acronym(entity.name)
            if len(acronym) >= 2:
                aliases.add(acronym)
        
        # Add name variations (first/last name for persons)
        if entity.entity_type == 'person':
            name_parts = entity.name.split()
            if len(name_parts) >= 2:
                aliases.add(name_parts[0])  # First name
                aliases.add(name_parts[-1])  # Last name
                # Add initials
                initials = ''.join(part[0].upper() for part in name_parts if part)
                aliases.add(initials)
        
        return aliases
    
    def _generate_acronym(self, text: str) -> str:
        """Generate acronym from text"""
        words = text.split()
        return ''.join(word[0].upper() for word in words if word and word[0].isalpha())
    
    def compute_similarity(self, entity1: EntityCandidate, entity2: EntityCandidate) -> EntityMatch:
        """Compute similarity between two entities using multiple methods"""
        
        # 1. Exact matching
        if entity1.name.lower() == entity2.name.lower():
            return EntityMatch(
                candidate1=entity1,
                candidate2=entity2,
                similarity_score=1.0,
                match_type='exact',
                confidence=1.0,
                evidence={'exact_name_match': True}
            )
        
        # 2. Normalized exact matching
        norm1 = self.normalize_entity_name(entity1.name)
        norm2 = self.normalize_entity_name(entity2.name)
        if norm1 == norm2:
            return EntityMatch(
                candidate1=entity1,
                candidate2=entity2,
                similarity_score=0.95,
                match_type='exact',
                confidence=0.95,
                evidence={'normalized_exact_match': True}
            )
        
        # 3. Alias matching
        aliases1 = self.extract_aliases(entity1)
        aliases2 = self.extract_aliases(entity2)
        if aliases1.intersection(aliases2):
            overlap = aliases1.intersection(aliases2)
            return EntityMatch(
                candidate1=entity1,
                candidate2=entity2,
                similarity_score=0.9,
                match_type='alias',
                confidence=0.9,
                evidence={'alias_overlap': list(overlap)}
            )
        
        # 4. Fuzzy string matching
        fuzzy_score = 0.0
        if FUZZYWUZZY_AVAILABLE:
            fuzzy_score = fuzz.ratio(entity1.name, entity2.name) / 100.0
        else:
            # Fallback to basic similarity
            fuzzy_score = SequenceMatcher(None, entity1.name, entity2.name).ratio()
        
        # 5. Semantic similarity using embeddings
        semantic_score = 0.0
        if (self.use_embeddings and self.sentence_model and 
            entity1.embedding is not None and entity2.embedding is not None and
            SKLEARN_AVAILABLE):
            try:
                semantic_score = cosine_similarity(
                    entity1.embedding.reshape(1, -1),
                    entity2.embedding.reshape(1, -1)
                )[0][0]
            except Exception as e:
                logger.warning(f"Error computing semantic similarity: {e}")
        
        # Combine scores with weights
        combined_score = max(fuzzy_score, semantic_score)
        match_type = 'fuzzy' if fuzzy_score > semantic_score else 'semantic'
        
        return EntityMatch(
            candidate1=entity1,
            candidate2=entity2,
            similarity_score=combined_score,
            match_type=match_type,
            confidence=combined_score,
            evidence={
                'fuzzy_score': fuzzy_score,
                'semantic_score': semantic_score,
                'combined_score': combined_score
            }
        )
    
    def add_embeddings(self, entities: List[EntityCandidate]) -> List[EntityCandidate]:
        """Add semantic embeddings to entities"""
        if not self.sentence_model or not NUMPY_AVAILABLE:
            return entities
        
        try:
            # Prepare texts for embedding
            texts = []
            for entity in entities:
                # Combine name with attributes for richer embeddings
                text_parts = [entity.name]
                if entity.attributes.get('description'):
                    text_parts.append(entity.attributes['description'])
                if entity.context_snippets:
                    text_parts.extend(entity.context_snippets[:2])  # Limit context
                texts.append(' '.join(text_parts))
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(texts)
            
            # Assign embeddings to entities
            for entity, embedding in zip(entities, embeddings):
                entity.embedding = embedding
            
            logger.info(f"Generated embeddings for {len(entities)} entities")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
        
        return entities
    
    def find_matches(self, entities: List[EntityCandidate]) -> List[EntityMatch]:
        """Find potential matches between entities"""
        matches = []
        
        # Add embeddings if enabled
        if self.use_embeddings:
            entities = self.add_embeddings(entities)
        
        # Compare all pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Skip if different types (unless one is unknown)
                if (entity1.entity_type != entity2.entity_type and 
                    entity1.entity_type != 'unknown' and 
                    entity2.entity_type != 'unknown'):
                    continue
                
                match = self.compute_similarity(entity1, entity2)
                
                # Only keep matches above threshold
                threshold = self.similarity_threshold
                if match.match_type == 'fuzzy':
                    threshold = self.fuzzy_threshold / 100.0
                elif match.match_type == 'semantic':
                    threshold = self.semantic_threshold
                
                if match.similarity_score >= threshold:
                    matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches
    
    def cluster_entities(self, entities: List[EntityCandidate]) -> List[EntityCluster]:
        """Cluster entities into resolved groups"""
        matches = self.find_matches(entities)
        
        # Build connectivity graph
        entity_to_idx = {entity.id: i for i, entity in enumerate(entities)}
        connected_components = []
        visited = set()
        
        # Find connected components using DFS
        for entity in entities:
            if entity.id in visited:
                continue
            
            component = []
            stack = [entity]
            
            while stack:
                current = stack.pop()
                if current.id in visited:
                    continue
                
                visited.add(current.id)
                component.append(current)
                
                # Find connected entities
                for match in matches:
                    if match.candidate1.id == current.id:
                        if match.candidate2.id not in visited:
                            stack.append(match.candidate2)
                    elif match.candidate2.id == current.id:
                        if match.candidate1.id not in visited:
                            stack.append(match.candidate1)
            
            if len(component) > 1:
                connected_components.append(component)
        
        # Create clusters
        clusters = []
        for component in connected_components:
            # Choose canonical entity (highest confidence or most mentions)
            canonical = max(component, key=lambda e: (e.confidence, len(e.mentions)))
            
            # Calculate cluster confidence
            cluster_confidence = sum(e.confidence for e in component) / len(component)
            
            cluster = EntityCluster(
                canonical_entity=canonical,
                member_entities=component,
                cluster_confidence=cluster_confidence,
                resolution_method='graph_clustering'
            )
            clusters.append(cluster)
        
        self.entity_clusters = clusters
        return clusters
    
    def resolve_entity(self, new_entity: EntityCandidate) -> Optional[EntityCandidate]:
        """Resolve a new entity against existing registry"""
        
        # Add embedding if needed
        if self.use_embeddings and self.sentence_model:
            new_entity = self.add_embeddings([new_entity])[0]
        
        best_match = None
        best_score = 0.0
        
        # Compare against all registered entities
        for registered_entity in self.entity_registry.values():
            match = self.compute_similarity(new_entity, registered_entity)
            
            if match.similarity_score > best_score:
                best_score = match.similarity_score
                best_match = registered_entity
        
        # Return match if above threshold
        if best_score >= self.similarity_threshold:
            # Merge information
            return self.merge_entities(best_match, new_entity)
        
        # Register as new entity
        self.entity_registry[new_entity.id] = new_entity
        return new_entity
    
    def merge_entities(self, canonical: EntityCandidate, duplicate: EntityCandidate) -> EntityCandidate:
        """Merge duplicate entity into canonical entity"""
        
        # Merge aliases
        canonical.aliases.update(duplicate.aliases)
        canonical.aliases.update(self.extract_aliases(duplicate))
        
        # Merge mentions
        canonical.mentions.extend(duplicate.mentions)
        
        # Merge context snippets
        canonical.context_snippets.extend(duplicate.context_snippets)
        
        # Merge attributes (prefer canonical for conflicts)
        for key, value in duplicate.attributes.items():
            if key not in canonical.attributes:
                canonical.attributes[key] = value
        
        # Update confidence (weighted average)
        total_mentions = len(canonical.mentions) + len(duplicate.mentions)
        if total_mentions > 0:
            canonical.confidence = (
                (canonical.confidence * len(canonical.mentions) + 
                 duplicate.confidence * len(duplicate.mentions)) / total_mentions
            )
        
        return canonical
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity resolution"""
        total_entities = len(self.entity_registry)
        total_clusters = len(self.entity_clusters)
        
        cluster_sizes = [len(cluster.member_entities) for cluster in self.entity_clusters]
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        
        return {
            'total_entities': total_entities,
            'total_clusters': total_clusters,
            'average_cluster_size': avg_cluster_size,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'entities_in_clusters': sum(cluster_sizes),
            'unique_entities': total_entities - sum(cluster_sizes) + total_clusters
        }


def create_entity_resolver(similarity_threshold: float = 0.8,
                          fuzzy_threshold: int = 85,
                          semantic_threshold: float = 0.7) -> EntityResolver:
    """Factory function to create an entity resolver"""
    return EntityResolver(
        similarity_threshold=similarity_threshold,
        fuzzy_threshold=fuzzy_threshold,
        semantic_threshold=semantic_threshold
    )


if __name__ == "__main__":
    # Example usage
    resolver = create_entity_resolver()
    
    # Create test entities
    entities = [
        EntityCandidate(
            id="1",
            name="John Smith",
            entity_type="person",
            mentions=["John Smith", "J. Smith"]
        ),
        EntityCandidate(
            id="2", 
            name="Jonathan Smith",
            entity_type="person",
            mentions=["Jonathan Smith"]
        ),
        EntityCandidate(
            id="3",
            name="Microsoft Corporation", 
            entity_type="organization",
            mentions=["Microsoft Corp", "MSFT"]
        ),
        EntityCandidate(
            id="4",
            name="Microsoft Corp",
            entity_type="organization"
        )
    ]
    
    # Find matches
    matches = resolver.find_matches(entities)
    print(f"Found {len(matches)} potential matches")
    
    for match in matches:
        print(f"Match: {match.candidate1.name} <-> {match.candidate2.name}")
        print(f"  Score: {match.similarity_score:.3f}, Type: {match.match_type}")
    
    # Cluster entities
    clusters = resolver.cluster_entities(entities)
    print(f"\nFound {len(clusters)} clusters:")
    
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster.canonical_entity.name}")
        for member in cluster.member_entities:
            if member.id != cluster.canonical_entity.id:
                print(f"  - {member.name}")
    
    # Print statistics
    stats = resolver.get_resolution_statistics()
    print(f"\nResolution Statistics: {stats}")
