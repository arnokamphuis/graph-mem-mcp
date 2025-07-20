"""
Advanced NLP & ML Extraction Components

This package provides sophisticated extraction capabilities for knowledge graphs,
including relationship extraction, enhanced entity extraction, and coreference resolution.
"""

from .relation_extractor import (
    RelationshipCandidate,
    RelationshipExtractor,
    create_relationship_extractor
)

__all__ = [
    'RelationshipCandidate',
    'RelationshipExtractor', 
    'create_relationship_extractor'
]
