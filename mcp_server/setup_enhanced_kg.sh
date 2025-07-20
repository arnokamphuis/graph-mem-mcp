#!/bin/bash

# Enhanced Knowledge Graph Setup Script
# This script installs all required packages for the enhanced knowledge graph construction

echo "Setting up Enhanced Knowledge Graph Construction..."

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Download spaCy English model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download additional models if needed
echo "Downloading additional spaCy models..."
python -m spacy download en_core_web_md  # Medium model with word vectors

echo "Setup complete!"
echo ""
echo "The enhanced knowledge graph construction is now available with:"
echo "- spaCy for Named Entity Recognition and dependency parsing"
echo "- sentence-transformers for semantic similarity"
echo "- Advanced relationship extraction"
echo "- Concept clustering and deduplication"
echo "- Semantic observation linking"
echo ""
echo "To use the enhanced features, restart your MCP server."
