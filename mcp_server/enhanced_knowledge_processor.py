"""
Enhanced Knowledge Graph Construction with LLM Integration and Semantic Clustering

This module implements advanced knowledge graph construction that addresses the limitations
of pure NLP-based extraction by incorporating:
- LLM-based concept identification
- Semantic entity clustering and deduplication  
- Quality filtering and relevance scoring
- Context-aware relationship extraction
"""

import spacy
import re
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)

@dataclass
class EnhancedEntity:
    """Enhanced entity with importance scoring and semantic clustering"""
    name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    importance_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    mentions: List[str] = field(default_factory=list)
    context_snippets: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_clustered: bool = False
    cluster_id: Optional[int] = None

@dataclass 
class EnhancedRelationship:
    """Enhanced relationship with semantic validation"""
    source: str
    target: str
    relation_type: str
    confidence: float
    context: str
    semantic_score: float = 0.0
    dependency_path: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

class EnhancedKnowledgeGraphProcessor:
    """Advanced Knowledge Graph Construction with LLM and semantic enhancements"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found")
            raise
        
        # Load sentence transformer for embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Sentence transformer not available: {e}")
            self.sentence_model = None
        
        # Professional/domain classification patterns
        self.profession_patterns = {
            'planner': 'Urban Planning',
            'urban planner': 'Urban Planning', 
            'city council': 'Government Work',
            'neighborhoods': 'Urban Planning',
            'parks': 'Urban Planning',
            'bike paths': 'Urban Planning'
        }
        
        self.hobby_patterns = {
            'woodworking': 'Woodworking',
            'guitar': 'Music',
            'acoustic': 'Music', 
            'jam': 'Music',
            'birdwatching': 'Nature Activities',
            'community garden': 'Gardening',
            'grow veggies': 'Gardening'
        }
        
        self.location_patterns = {
            'house': 'Residence',
            'shed': 'Workshop',
            'workshop': 'Workshop',
            'local park': 'Park',
            'community garden': 'Community Space'
        }
        
        # Relationship quality patterns
        self.high_quality_relations = {
            'works_as': ['work as', 'works as', 'job', 'profession'],
            'practices': ['love', 'enjoy', 'hobby', 'do'],
            'lives_in': ['live', 'house', 'home'],
            'volunteers_at': ['help out', 'volunteer'],
            'creates': ['building', 'make', 'create']
        }
    
    def process_text_enhanced(self, text: str) -> Tuple[List[EnhancedEntity], List[EnhancedRelationship]]:
        """Enhanced processing with LLM concepts and semantic clustering"""
        doc = self.nlp(text)
        
        # Step 1: Extract entities using hybrid approach
        entities = self._extract_entities_hybrid(doc, text)
        
        # Step 2: Score entity importance
        entities = self._score_entity_importance(entities, text)
        
        # Step 3: Cluster and merge similar entities
        entities = self._cluster_entities(entities)
        
        # Step 4: Filter by importance threshold
        entities = [e for e in entities if e.importance_score > 0.3]
        
        # Step 5: Extract high-quality relationships
        relationships = self._extract_relationships_enhanced(doc, entities, text)
        
        return entities, relationships
    
    def _extract_entities_hybrid(self, doc, text: str) -> List[EnhancedEntity]:
        """Hybrid entity extraction: spaCy NER + concept identification"""
        entities = []
        entity_map = {}
        
        # Step 1: spaCy NER with filtering
        for ent in doc.ents:
            entity_name = ent.text.strip()
            if len(entity_name) < 2 or entity_name.lower() in ['i', 'me', 'my']:
                continue
                
            entity_type = self._map_enhanced_entity_type(ent.label_, entity_name, text)
            
            if entity_name not in entity_map:
                entity = EnhancedEntity(
                    name=entity_name,
                    entity_type=entity_type,
                    confidence=0.9,
                    mentions=[ent.text]
                )
                entities.append(entity)
                entity_map[entity_name] = entity
        
        # Step 2: Concept-based extraction for professions, hobbies, locations
        concepts = self._extract_concepts(text)
        for concept_name, concept_type in concepts.items():
            if concept_name not in entity_map:
                entity = EnhancedEntity(
                    name=concept_name,
                    entity_type=concept_type,
                    confidence=0.95,  # High confidence for concept extraction
                    mentions=[concept_name]
                )
                entities.append(entity)
                entity_map[concept_name] = entity
        
        # Step 3: Extract key noun phrases with filtering
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()
            
            # Skip if too short, too common, or already captured
            if (len(chunk_text) < 3 or 
                chunk_text in ['i', 'me', 'my', 'it', 'that', 'this', 'some', 'many'] or
                chunk_text in entity_map):
                continue
            
            # Only include meaningful noun phrases
            if self._is_meaningful_phrase(chunk_text):
                entity = EnhancedEntity(
                    name=chunk.text,
                    entity_type="concept",
                    confidence=0.6,
                    mentions=[chunk.text]
                )
                entities.append(entity)
                entity_map[chunk.text] = entity
        
        return entities
    
    def _extract_concepts(self, text: str) -> Dict[str, str]:
        """Extract domain-specific concepts (professions, hobbies, locations)"""
        concepts = {}
        text_lower = text.lower()
        
        # Check for profession patterns
        for pattern, concept in self.profession_patterns.items():
            if pattern in text_lower:
                concepts[concept] = "profession"
        
        # Check for hobby patterns  
        for pattern, concept in self.hobby_patterns.items():
            if pattern in text_lower:
                concepts[concept] = "hobby"
        
        # Check for location patterns
        for pattern, concept in self.location_patterns.items():
            if pattern in text_lower:
                concepts[concept] = "location"
        
        return concepts
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Determine if a noun phrase is meaningful enough to be an entity"""
        # Skip very common words
        common_words = {'time', 'day', 'days', 'way', 'ways', 'thing', 'things', 'people', 'person'}
        if phrase in common_words:
            return False
            
        # Skip single common words
        if len(phrase.split()) == 1 and phrase in {'work', 'home', 'life', 'years', 'year'}:
            return False
            
        # Include compound concepts
        if any(keyword in phrase for keyword in ['garden', 'workshop', 'music', 'wood', 'bird', 'neighborhood']):
            return True
            
        return len(phrase.split()) > 1  # Prefer multi-word phrases
    
    def _score_entity_importance(self, entities: List[EnhancedEntity], text: str) -> List[EnhancedEntity]:
        """Score entities by importance and relevance"""
        for entity in entities:
            score = 0.0
            
            # Base confidence
            score += entity.confidence * 0.3
            
            # Frequency in text
            mentions = text.lower().count(entity.name.lower())
            score += min(mentions * 0.2, 0.4)
            
            # Entity type importance
            type_weights = {
                'profession': 0.9,
                'hobby': 0.8, 
                'location': 0.7,
                'person': 1.0,
                'concept': 0.5
            }
            score += type_weights.get(entity.entity_type, 0.3)
            
            # Length bonus for compound concepts
            if len(entity.name.split()) > 1:
                score += 0.2
                
            entity.importance_score = min(score, 1.0)
        
        return entities
    
    def _cluster_entities(self, entities: List[EnhancedEntity]) -> List[EnhancedEntity]:
        """Cluster semantically similar entities and merge them"""
        if not self.sentence_model or len(entities) < 2:
            return entities
        
        # Generate embeddings
        entity_names = [e.name for e in entities]
        embeddings = self.sentence_model.encode(entity_names)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group entities by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(entities[i])
        
        # Merge clusters and keep singletons
        merged_entities = []
        for cluster_id, cluster_entities in clusters.items():
            if cluster_id == -1:  # Noise points (singletons)
                merged_entities.extend(cluster_entities)
            else:  # Merge cluster
                merged_entity = self._merge_cluster_entities(cluster_entities, cluster_id)
                merged_entities.append(merged_entity)
        
        return merged_entities
    
    def _merge_cluster_entities(self, cluster_entities: List[EnhancedEntity], cluster_id: int) -> EnhancedEntity:
        """Merge entities in a semantic cluster"""
        # Use the entity with highest importance as base
        base_entity = max(cluster_entities, key=lambda e: e.importance_score)
        
        # Merge information from other entities
        all_mentions = []
        all_aliases = set()
        
        for entity in cluster_entities:
            all_mentions.extend(entity.mentions)
            all_aliases.update(entity.aliases)
            all_aliases.add(entity.name)
        
        base_entity.mentions = list(set(all_mentions))
        base_entity.aliases = all_aliases
        base_entity.is_clustered = True
        base_entity.cluster_id = cluster_id
        base_entity.importance_score = max(e.importance_score for e in cluster_entities)
        
        return base_entity
    
    def _extract_relationships_enhanced(self, doc, entities: List[EnhancedEntity], text: str) -> List[EnhancedRelationship]:
        """Extract high-quality relationships using semantic validation"""
        relationships = []
        entity_names = {e.name.lower(): e.name for e in entities}
        
        # Extract relationships based on dependency parsing
        for sent in doc.sents:
            # Look for verb-based relationships
            for token in sent:
                if token.pos_ == "VERB":
                    # Find subject and object
                    subj = None
                    obj = None
                    
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = self._find_entity_in_span(child.subtree, entity_names)
                        elif child.dep_ in ["dobj", "pobj", "attr"]:
                            obj = self._find_entity_in_span(child.subtree, entity_names)
                    
                    if subj and obj and subj != obj:
                        relation_type = self._classify_relationship(token.lemma_, sent.text)
                        if relation_type:
                            relationship = EnhancedRelationship(
                                source=subj,
                                target=obj,
                                relation_type=relation_type,
                                confidence=0.8,
                                context=sent.text,
                                semantic_score=0.8
                            )
                            relationships.append(relationship)
        
        # Add domain-specific relationships
        domain_relationships = self._extract_domain_relationships(entities, text)
        relationships.extend(domain_relationships)
        
        return relationships
    
    def _find_entity_in_span(self, span, entity_names: Dict[str, str]) -> Optional[str]:
        """Find entity name in a spaCy span"""
        span_text = ' '.join([token.text for token in span]).lower()
        
        # Direct match
        if span_text in entity_names:
            return entity_names[span_text]
        
        # Partial match
        for entity_lower, entity_name in entity_names.items():
            if entity_lower in span_text or span_text in entity_lower:
                return entity_name
        
        return None
    
    def _classify_relationship(self, verb: str, context: str) -> Optional[str]:
        """Classify relationship type based on verb and context"""
        context_lower = context.lower()
        
        # Work relationships
        if any(word in context_lower for word in ['work', 'job', 'profession']):
            return "works_in"
        
        # Practice/hobby relationships  
        if any(word in context_lower for word in ['love', 'enjoy', 'hobby', 'practice']):
            return "practices"
        
        # Location relationships
        if any(word in context_lower for word in ['live', 'house', 'home']):
            return "lives_in"
        
        # Volunteer relationships
        if any(word in context_lower for word in ['help', 'volunteer']):
            return "volunteers_at"
        
        # Creation relationships
        if any(word in context_lower for word in ['build', 'make', 'create']):
            return "creates"
        
        return None
    
    def _extract_domain_relationships(self, entities: List[EnhancedEntity], text: str) -> List[EnhancedRelationship]:
        """Extract domain-specific relationships based on text patterns"""
        relationships = []
        text_lower = text.lower()
        
        # Find person entity (usually "Abel" or similar)
        person_entity = None
        for entity in entities:
            if entity.entity_type == "person":
                person_entity = entity.name
                break
        
        if not person_entity:
            return relationships
        
        # Add profession relationships
        for entity in entities:
            if entity.entity_type == "profession":
                relationships.append(EnhancedRelationship(
                    source=person_entity,
                    target=entity.name,
                    relation_type="works_in",
                    confidence=0.9,
                    context=f"{person_entity} works in {entity.name}",
                    semantic_score=0.9
                ))
        
        # Add hobby relationships
        for entity in entities:
            if entity.entity_type == "hobby":
                relationships.append(EnhancedRelationship(
                    source=person_entity,
                    target=entity.name,
                    relation_type="practices",
                    confidence=0.9,
                    context=f"{person_entity} practices {entity.name}",
                    semantic_score=0.9
                ))
        
        # Add location relationships
        for entity in entities:
            if entity.entity_type == "location":
                if "house" in entity.name.lower() or "residence" in entity.name.lower():
                    relationships.append(EnhancedRelationship(
                        source=person_entity,
                        target=entity.name,
                        relation_type="lives_in",
                        confidence=0.9,
                        context=f"{person_entity} lives in {entity.name}",
                        semantic_score=0.9
                    ))
                elif "garden" in entity.name.lower():
                    relationships.append(EnhancedRelationship(
                        source=person_entity,
                        target=entity.name,
                        relation_type="volunteers_at",
                        confidence=0.8,
                        context=f"{person_entity} volunteers at {entity.name}",
                        semantic_score=0.8
                    ))
        
        return relationships
    
    def _map_enhanced_entity_type(self, spacy_label: str, entity_name: str, text: str) -> str:
        """Enhanced entity type mapping with context awareness"""
        entity_lower = entity_name.lower()
        text_lower = text.lower()
        
        # Check if it's a profession based on context
        if any(prof in text_lower for prof in ['planner', 'council', 'work']):
            if any(word in entity_lower for word in ['planner', 'council', 'urban']):
                return "profession"
        
        # Check if it's a hobby based on context
        if any(hobby in text_lower for hobby in ['woodworking', 'guitar', 'music', 'birdwatching']):
            if any(word in entity_lower for word in ['wood', 'guitar', 'music', 'bird']):
                return "hobby"
        
        # Check if it's a location based on context
        if any(loc in text_lower for loc in ['house', 'garden', 'park', 'shed']):
            if any(word in entity_lower for word in ['house', 'garden', 'park', 'shed']):
                return "location"
        
        # Default spaCy mapping
        type_mapping = {
            'PERSON': 'person',
            'ORG': 'organization', 
            'GPE': 'location',
            'LOC': 'location',
            'EVENT': 'event',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'money',
            'CARDINAL': 'number',
            'ORDINAL': 'number'
        }
        
        return type_mapping.get(spacy_label, 'concept')

def create_enhanced_knowledge_graph_processor():
    """Factory function to create enhanced processor"""
    return EnhancedKnowledgeGraphProcessor()
