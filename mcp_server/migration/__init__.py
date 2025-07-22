"""
Migration Module for MCP Integration

This module provides utilities for migrating from the legacy memory_banks system 
to the new Phase 3.2 storage abstraction layer.

Migration Components:
- legacy_migrator.py - Utilities for migrating existing data
- compatibility.py - Backwards compatibility layer
- validation.py - Migration validation and testing

Usage:
    from migration import migrate_legacy_data, validate_migration
    
    # Migrate existing data
    success = migrate_legacy_data(legacy_banks, new_storage)
    
    # Validate migration
    validation_results = validate_migration(legacy_banks, new_storage)
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot"
