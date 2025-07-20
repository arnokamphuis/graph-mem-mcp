# Coding Standards for Knowledge Graph Refactoring

## Dependency Management Pattern

**Always use this pattern for optional dependencies:**

```python
try:
    import optional_library
    OPTIONAL_AVAILABLE = True
    logger.info("Optional library loaded successfully")
except ImportError:
    OPTIONAL_AVAILABLE = False
    optional_library = None
    logger.warning("Optional library not available - using fallback")
```

## Error Handling Requirements

**Implement comprehensive error handling:**

```python
# Log warnings for missing optional features
# Provide meaningful error messages with recovery suggestions  
# Use try-catch blocks for external library calls
# Validate inputs before processing
# Include fallback mechanisms when dependencies missing
```

## Testing Standards

**For each new module, create:**

- Unit tests for all public methods (>90% coverage required)
- Integration tests for component interactions
- Performance benchmarks for critical operations
- Example usage scripts demonstrating functionality
- Error case testing for missing dependencies

## Type Safety Requirements

**Use comprehensive type hints:**

```python
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

# Define clear data structures with validation
# Include type hints for all function parameters and returns
# Use dataclasses for structured data
# Validate types at runtime when appropriate
```

## Documentation Standards

**Every module must include:**

- Comprehensive module docstring with purpose and usage examples
- Docstrings for all classes and public methods
- Type hints for all parameters and return values
- Usage examples in docstrings
- Clear error handling documentation

## Code Organization

**Follow existing patterns from `/mcp_server/core/`:**

- Graceful dependency handling
- Consistent error patterns
- Type safety throughout
- Comprehensive logging
- Performance monitoring where applicable

## Commit Message Format

```
[Phase X.Y] Component: Brief description

- Detailed description of changes
- Features implemented  
- Tests added/updated
- Performance impact
- Breaking changes (if any)
```

## Quality Gates

**All code must pass:**

- Type checking (mypy or similar)
- Code formatting (black or similar)
- Linting (flake8 or similar) 
- Unit tests with >90% coverage
- Integration tests
- Documentation completeness check
