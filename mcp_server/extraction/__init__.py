"""
Extraction Module - Phase 2 Integration

This module provides enhanced entity and relationship extraction capabilities
that integrate with Phase 1 core architecture. All extraction components
use the established schema management and entity resolution systems.

Key Components:
- EnhancedEntityExtractor: Multi-model ensemble entity extraction
- (Future) SophisticatedRelationExtractor: Advanced relationship extraction  
- (Future) CoreferenceResolver: Entity coreference resolution

Architecture Compliance:
- Integrates with core/graph_schema.py for schema management
- Uses core/entity_resolution.py for entity deduplication
- Follows dependency management patterns with graceful fallbacks
- Outputs schema-compatible EntityInstance and RelationInstance objects
"""

from typing import Dict, List, Optional, Any
import logging

# Core Phase 1 integration
try:
    from ..core.graph_schema import SchemaManager, EntityInstance, RelationshipInstance
    from ..core.entity_resolution import EntityResolver
    CORE_INTEGRATION_AVAILABLE = True
    logging.info("✅ Core Phase 1 integration available")
except ImportError as e:
    CORE_INTEGRATION_AVAILABLE = False
    logging.error(f"❌ Core Phase 1 integration not available: {e}")
    raise ImportError("Phase 2 extraction requires Phase 1 core components")

# Enhanced extraction components
try:
    from .enhanced_entity_extractor import (
        EnhancedEntityExtractor, 
        ExtractionContext,
        ExtractionStrategy,
        extract_entities_from_text
    )
    ENHANCED_ENTITY_EXTRACTION_AVAILABLE = True
    logging.info("✅ Enhanced entity extraction available")
except ImportError as e:
    ENHANCED_ENTITY_EXTRACTION_AVAILABLE = False
    logging.warning(f"⚠️  Enhanced entity extraction not available: {e}")

# Legacy extraction components (to be refactored)
try:
    from .relation_extractor import (
        RelationshipCandidate,
        RelationshipExtractor,
        create_relationship_extractor
    )
    LEGACY_RELATION_EXTRACTION_AVAILABLE = True
    logging.warning("⚠️  Legacy relationship extraction available - needs refactoring per plan")
except ImportError as e:
    LEGACY_RELATION_EXTRACTION_AVAILABLE = False

# Future extraction components (Phase 2.1, 2.3)
SOPHISTICATED_RELATION_EXTRACTION_AVAILABLE = False
COREFERENCE_RESOLUTION_AVAILABLE = False

# Export key components for easy import
__all__ = [
    # Core integration
    'CORE_INTEGRATION_AVAILABLE',
    
    # Enhanced entity extraction (Phase 2.2)
    'EnhancedEntityExtractor',
    'ExtractionContext', 
    'ExtractionStrategy',
    'extract_entities_from_text',
    'ENHANCED_ENTITY_EXTRACTION_AVAILABLE',
    
    # Legacy components (to be refactored)
    'RelationshipCandidate',
    'RelationshipExtractor', 
    'create_relationship_extractor',
    'LEGACY_RELATION_EXTRACTION_AVAILABLE',
    
    # Future components
    'SOPHISTICATED_RELATION_EXTRACTION_AVAILABLE',
    'COREFERENCE_RESOLUTION_AVAILABLE',
    
    # Integration functions
    'create_extraction_pipeline',
    'get_extraction_capabilities'
]


def create_extraction_pipeline(schema_manager: SchemaManager, 
                             entity_resolver: Optional[EntityResolver] = None) -> Dict[str, Any]:
    """
    Create an integrated extraction pipeline using Phase 1 and Phase 2 components
    
    Args:
        schema_manager: Schema manager from Phase 1
        entity_resolver: Optional entity resolver from Phase 1
        
    Returns:
        Dictionary containing extraction pipeline components
    """
    pipeline = {
        'schema_manager': schema_manager,
        'entity_resolver': entity_resolver,
        'components': {},
        'capabilities': get_extraction_capabilities()
    }
    
    # Initialize available extraction components
    if ENHANCED_ENTITY_EXTRACTION_AVAILABLE:
        pipeline['components']['entity_extractor'] = EnhancedEntityExtractor(schema_manager)
        logging.info("✅ Entity extractor added to pipeline")
    
    # Legacy relationship extractor (needs refactoring)
    if LEGACY_RELATION_EXTRACTION_AVAILABLE:
        pipeline['components']['legacy_relation_extractor'] = create_relationship_extractor()
        logging.warning("⚠️  Legacy relationship extractor added - needs refactoring per plan")
    
    # Future: Add relationship extractor when Phase 2.1 is implemented
    # if SOPHISTICATED_RELATION_EXTRACTION_AVAILABLE:
    #     pipeline['components']['relation_extractor'] = SophisticatedRelationExtractor(schema_manager)
    
    # Future: Add coreference resolver when Phase 2.3 is implemented  
    # if COREFERENCE_RESOLUTION_AVAILABLE:
    #     pipeline['components']['coreference_resolver'] = CoreferenceResolver(schema_manager)
    
    return pipeline


def get_extraction_capabilities() -> Dict[str, bool]:
    """
    Get current extraction capabilities based on available components
    
    Returns:
        Dictionary of capability flags
    """
    return {
        'core_integration': CORE_INTEGRATION_AVAILABLE,
        'enhanced_entity_extraction': ENHANCED_ENTITY_EXTRACTION_AVAILABLE,
        'legacy_relation_extraction': LEGACY_RELATION_EXTRACTION_AVAILABLE,
        'sophisticated_relation_extraction': SOPHISTICATED_RELATION_EXTRACTION_AVAILABLE,
        'coreference_resolution': COREFERENCE_RESOLUTION_AVAILABLE,
        'multi_model_ensemble': ENHANCED_ENTITY_EXTRACTION_AVAILABLE,
        'schema_guided_extraction': ENHANCED_ENTITY_EXTRACTION_AVAILABLE and CORE_INTEGRATION_AVAILABLE,
        'entity_resolution_integration': CORE_INTEGRATION_AVAILABLE
    }


def extract_full_knowledge_graph(text: str, schema_manager: SchemaManager,
                               entity_resolver: Optional[EntityResolver] = None,
                               confidence_threshold: float = 0.7) -> Dict[str, List[Any]]:
    """
    Extract complete knowledge graph from text using all available extraction components
    
    Args:
        text: Input text for extraction
        schema_manager: Schema manager from Phase 1
        entity_resolver: Optional entity resolver from Phase 1
        confidence_threshold: Minimum confidence for extraction
        
    Returns:
        Dictionary containing extracted entities and relationships
    """
    results = {
        'entities': [],
        'relationships': [],
        'coreferences': [],
        'extraction_stats': {}
    }
    
    # Extract entities (Phase 2.2)
    if ENHANCED_ENTITY_EXTRACTION_AVAILABLE:
        try:
            entities = extract_entities_from_text(
                text=text,
                schema_manager=schema_manager,
                entity_resolver=entity_resolver,
                confidence_threshold=confidence_threshold
            )
            results['entities'] = entities
            results['extraction_stats']['entities_extracted'] = len(entities)
            logging.info(f"✅ Extracted {len(entities)} entities")
        except Exception as e:
            logging.error(f"❌ Entity extraction failed: {e}")
            results['extraction_stats']['entity_extraction_error'] = str(e)
    
    # Future: Extract relationships (Phase 2.1)
    # if SOPHISTICATED_RELATION_EXTRACTION_AVAILABLE:
    #     try:
    #         relationships = extract_relationships_from_text(text, entities, schema_manager)
    #         results['relationships'] = relationships
    #     except Exception as e:
    #         logging.error(f"❌ Relationship extraction failed: {e}")
    
    # Future: Resolve coreferences (Phase 2.3)
    # if COREFERENCE_RESOLUTION_AVAILABLE:
    #     try:
    #         coreferences = resolve_coreferences(text, entities)
    #         results['coreferences'] = coreferences
    #     except Exception as e:
    #         logging.error(f"❌ Coreference resolution failed: {e}")
    
    return results
