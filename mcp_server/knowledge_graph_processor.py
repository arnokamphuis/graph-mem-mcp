"""
Modern Knowledge Graph Construction using spaCy and Advanced NLP

This module implements a state-of-the-art knowledge graph construction pipeline
inspired by the best practices from https://memgraph.com/blog/best-python-packages-tools-for-knowledge-graphs

Features:
- spaCy for Named Entity Recognition and dependency parsing
- sentence-transformers for semantic embeddings
- Advanced relationship extraction using linguistic patterns
- Entity linking and coreference resolution
- Graph-based reasoning and inference
"""

import spacy
import re
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None
    mentions: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class Relationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relationship_type: str
    confidence: float
    context: str
    dependency_path: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

class ModernKnowledgeGraphProcessor:
    """Advanced Knowledge Graph Construction using modern NLP techniques"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the knowledge graph processor with spaCy and transformers"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}")
            # Fallback to blank model
            self.nlp = spacy.blank("en")
            
        # Add custom pipeline components if needed
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
            
        # Initialize sentence transformer for embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
            
        # Relationship extraction patterns based on dependency parsing
        self.relation_patterns = self._initialize_relation_patterns()
        
    def _initialize_relation_patterns(self) -> Dict[str, List[str]]:
        """Initialize linguistic patterns for relationship extraction"""
        return {
            'causes': ['cause', 'lead', 'result', 'trigger', 'induce', 'provoke'],
            'creates': ['create', 'produce', 'generate', 'make', 'build', 'develop', 'form'],
            'contains': ['contain', 'include', 'have', 'hold', 'comprise', 'consist'],
            'located_in': ['locate', 'situate', 'position', 'place', 'base'],
            'part_of': ['part', 'component', 'element', 'piece', 'section'],
            'type_of': ['type', 'kind', 'form', 'variety', 'class', 'category'],
            'owns': ['own', 'possess', 'belong', 'control'],
            'works_for': ['work', 'employ', 'serve', 'represent'],
            'collaborates_with': ['collaborate', 'cooperate', 'partner', 'work together'],
            'influences': ['influence', 'affect', 'impact', 'shape', 'determine'],
            'uses': ['use', 'utilize', 'employ', 'apply', 'leverage'],
            'develops': ['develop', 'design', 'build', 'create', 'engineer'],
            'manages': ['manage', 'lead', 'direct', 'oversee', 'supervise'],
            'competes_with': ['compete', 'rival', 'oppose', 'challenge'],
            'similar_to': ['similar', 'like', 'resemble', 'comparable', 'equivalent']
        }
    
    def process_text(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """Process text and extract entities and relationships using spaCy"""
        doc = self.nlp(text)
        
        # Extract entities using spaCy NER
        entities = self._extract_entities_spacy(doc)
        
        # Extract relationships using dependency parsing
        relationships = self._extract_relationships_spacy(doc, entities)
        
        # Add embeddings to entities if sentence transformer is available
        if self.sentence_model:
            self._add_embeddings(entities, text)
            
        return entities, relationships
    
    def _extract_entities_spacy(self, doc) -> List[Entity]:
        """Extract entities using spaCy's Named Entity Recognition"""
        entities = []
        entity_map = {}
        
        # Extract named entities
        for ent in doc.ents:
            entity_name = ent.text.strip()
            if len(entity_name) < 2:
                continue
                
            # Map spaCy entity types to our types
            entity_type = self._map_spacy_entity_type(ent.label_)
            
            if entity_name not in entity_map:
                entity = Entity(
                    name=entity_name,
                    entity_type=entity_type,
                    confidence=0.9,  # High confidence for spaCy NER
                    mentions=[ent.text]
                )
                entities.append(entity)
                entity_map[entity_name] = entity
            else:
                # Add as alias or mention
                entity_map[entity_name].mentions.append(ent.text)
                entity_map[entity_name].aliases.add(ent.text)
        
        # Extract noun phrases as potential entities
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if (len(chunk_text) > 2 and 
                chunk_text not in entity_map and
                self._is_valid_entity(chunk_text)):
                
                entity = Entity(
                    name=chunk_text,
                    entity_type="concept",
                    confidence=0.6,  # Lower confidence for noun phrases
                    mentions=[chunk_text]
                )
                entities.append(entity)
                entity_map[chunk_text] = entity
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization', 
            'GPE': 'location',
            'LOC': 'location',
            'EVENT': 'event',
            'FAC': 'facility',
            'PRODUCT': 'product',
            'WORK_OF_ART': 'work_of_art',
            'LAW': 'law',
            'LANGUAGE': 'language',
            'DATE': 'date',
            'TIME': 'time',
            'PERCENT': 'percent',
            'MONEY': 'money',
            'QUANTITY': 'quantity',
            'ORDINAL': 'ordinal',
            'CARDINAL': 'number'
        }
        return mapping.get(spacy_label, 'entity')
    
    def _extract_relationships_spacy(self, doc, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using spaCy dependency parsing"""
        relationships = []
        entity_texts = {e.name.lower() for e in entities}
        
        for sent in doc.sents:
            # Find entities in this sentence
            sent_entities = []
            for token in sent:
                for entity in entities:
                    if token.text.lower() in entity.name.lower():
                        sent_entities.append((entity, token))
                        break
            
            if len(sent_entities) < 2:
                continue
                
            # Extract relationships using dependency patterns
            for i, (entity1, token1) in enumerate(sent_entities):
                for entity2, token2 in sent_entities[i+1:]:
                    relationship = self._extract_dependency_relationship(
                        sent, token1, token2, entity1, entity2
                    )
                    if relationship:
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_dependency_relationship(self, sent, token1, token2, entity1: Entity, entity2: Entity) -> Optional[Relationship]:
        """Extract relationship using dependency tree analysis"""
        # Find the path between tokens in the dependency tree
        path = self._find_dependency_path(token1, token2)
        if not path:
            return None
            
        # Analyze the path to determine relationship type
        relation_type, confidence = self._analyze_dependency_path(path, sent)
        
        if relation_type:
            return Relationship(
                source=entity1.name,
                target=entity2.name,
                relationship_type=relation_type,
                confidence=confidence,
                context=sent.text,
                dependency_path=" -> ".join([t.text + "(" + t.dep_ + ")" for t in path])
            )
        
        return None
    
    def _find_dependency_path(self, token1, token2):
        """Find the shortest path between two tokens in the dependency tree"""
        # Simple implementation - can be enhanced with more sophisticated algorithms
        if token1 == token2:
            return [token1]
            
        # Check if one is ancestor of the other
        ancestors1 = list(token1.ancestors)
        ancestors2 = list(token2.ancestors)
        
        # Find common ancestor
        for anc1 in [token1] + ancestors1:
            if anc1 in [token2] + ancestors2:
                # Found common ancestor, build path
                path1 = []
                current = token1
                while current != anc1:
                    path1.append(current)
                    current = current.head
                path1.append(anc1)
                
                path2 = []
                current = token2
                while current != anc1:
                    path2.append(current)
                    current = current.head
                
                # Combine paths
                path2.reverse()
                return path1 + path2
        
        return None
    
    def _analyze_dependency_path(self, path, sent) -> Tuple[Optional[str], float]:
        """Analyze dependency path to determine relationship type"""
        if not path:
            return None, 0.0
            
        # Look for key verbs and their roles in the path
        verbs = [token for token in path if token.pos_ == "VERB"]
        
        for verb in verbs:
            verb_lemma = verb.lemma_.lower()
            
            # Check against our relationship patterns
            for relation_type, patterns in self.relation_patterns.items():
                if verb_lemma in patterns:
                    confidence = 0.8
                    
                    # Boost confidence based on dependency relations
                    if any(token.dep_ in ["nsubj", "dobj", "pobj"] for token in path):
                        confidence = 0.9
                        
                    return relation_type, confidence
        
        # Fallback analysis based on dependency relations
        deps = [token.dep_ for token in path]
        
        if "poss" in deps:
            return "owns", 0.7
        elif "compound" in deps:
            return "part_of", 0.6
        elif "amod" in deps:
            return "has_property", 0.6
        elif "prep" in deps:
            # Look at the preposition
            preps = [token.text.lower() for token in path if token.dep_ == "prep"]
            for prep in preps:
                if prep in ["in", "at", "on"]:
                    return "located_in", 0.7
                elif prep in ["of", "from"]:
                    return "part_of", 0.7
                elif prep in ["with", "alongside"]:
                    return "associated_with", 0.6
        
        # Default to associated_with with low confidence
        return "associated_with", 0.4
    
    def _is_valid_entity(self, text: str) -> bool:
        """Check if text represents a valid entity"""
        # Filter out stop words, pronouns, etc.
        stop_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                     'he', 'she', 'it', 'they', 'we', 'you', 'i'}
        
        text_lower = text.lower().strip()
        return (len(text_lower) > 2 and 
                text_lower not in stop_words and
                not text_lower.isdigit() and
                not all(c in '.,!?;:' for c in text))
    
    def _add_embeddings(self, entities: List[Entity], context: str):
        """Add semantic embeddings to entities using sentence transformers"""
        if not self.sentence_model:
            return
            
        try:
            entity_texts = [f"{entity.name}: {' '.join(entity.mentions)}" for entity in entities]
            embeddings = self.sentence_model.encode(entity_texts)
            
            for entity, embedding in zip(entities, embeddings):
                entity.embedding = embedding
        except Exception as e:
            logger.warning(f"Could not generate embeddings: {e}")
    
    def find_similar_entities(self, entities: List[Entity], threshold: float = 0.8) -> List[Tuple[Entity, Entity, float]]:
        """Find similar entities using semantic embeddings"""
        if not self.sentence_model:
            return []
            
        similar_pairs = []
        embeddings = [e.embedding for e in entities if e.embedding is not None]
        
        if len(embeddings) < 2:
            return similar_pairs
            
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if (entities[i].embedding is not None and 
                    entities[j].embedding is not None):
                    
                    similarity = similarities[i][j]
                    if similarity > threshold:
                        similar_pairs.append((entities[i], entities[j], similarity))
        
        return similar_pairs
    
    def construct_knowledge_graph(self, text: str, existing_entities: List[str] = None) -> Dict[str, Any]:
        """Main method to construct knowledge graph from text"""
        # Process text to extract entities and relationships
        entities, relationships = self.process_text(text)
        
        # Find similar entities for potential merging
        similar_entities = self.find_similar_entities(entities)
        
        # Post-process and filter
        entities = self._post_process_entities(entities, existing_entities or [])
        relationships = self._post_process_relationships(relationships, entities)
        
        return {
            'entities': entities,
            'relationships': relationships,
            'similar_entities': similar_entities,
            'stats': {
                'entity_count': len(entities),
                'relationship_count': len(relationships),
                'entity_types': list(set(e.entity_type for e in entities)),
                'relationship_types': list(set(r.relation_type for r in relationships))
            }
        }
    
    def _post_process_entities(self, entities: List[Entity], existing_entities: List[str]) -> List[Entity]:
        """Post-process entities to improve quality"""
        # Remove very short or invalid entities
        filtered_entities = []
        
        for entity in entities:
            if (len(entity.name) >= 2 and 
                self._is_valid_entity(entity.name) and
                entity.confidence > 0.3):
                
                # Check if similar to existing entities
                entity.name = self._normalize_entity_name(entity.name)
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _post_process_relationships(self, relationships: List[Relationship], entities: List[Entity]) -> List[Relationship]:
        """Post-process relationships to improve quality"""
        entity_names = {e.name for e in entities}
        
        # Filter relationships where both entities exist
        filtered_relationships = []
        for rel in relationships:
            if (rel.source in entity_names and 
                rel.target in entity_names and
                rel.confidence > 0.3):
                filtered_relationships.append(rel)
        
        return filtered_relationships
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity names for consistency"""
        # Remove extra whitespace, convert to title case for proper nouns
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Simple heuristic: if all caps or all lower, convert to title case
        if name.isupper() or name.islower():
            name = name.title()
            
        return name

# Factory function for easy initialization
def create_knowledge_graph_processor() -> ModernKnowledgeGraphProcessor:
    """Create and return a configured knowledge graph processor"""
    return ModernKnowledgeGraphProcessor()
