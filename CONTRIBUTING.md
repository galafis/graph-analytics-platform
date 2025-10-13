# Contributing to Graph Analytics Platform

Thank you for your interest in contributing to the Graph Analytics Platform! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Your environment (OS, Python/Julia versions)
- Code samples if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Possible implementation approach

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Submit a pull request**

## 📝 Development Setup

### Prerequisites

- Python 3.8+
- Julia 1.9+
- Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/galafis/graph-analytics-platform.git
cd graph-analytics-platform

# Install Python dependencies
pip install -r requirements.txt

# Install Julia packages
julia -e 'using Pkg; Pkg.add(["Graphs", "LinearAlgebra", "SparseArrays", "DataStructures"])'

# Run tests
python tests/test_python_integration.py
julia tests/test_centrality.jl
```

## 🎨 Coding Standards

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Write docstrings for all public functions (NumPy style)
- Maximum line length: 100 characters

Example:
```python
def calculate_centrality(
    self,
    G: Optional[nx.Graph] = None,
    metric: str = 'betweenness'
) -> Dict[int, float]:
    """
    Calculate node centrality.
    
    Parameters
    ----------
    G : nx.Graph, optional
        Graph to analyze
    metric : str
        Centrality metric
        
    Returns
    -------
    centrality : dict
        Centrality scores
    """
    # Implementation
    pass
```

### Julia

- Follow [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Use meaningful variable names
- Write docstrings for all public functions
- Use 4 spaces for indentation

Example:
```julia
"""
Calculate betweenness centrality for all nodes.

# Arguments
- `g::AbstractGraph`: Input graph

# Returns
- `Dict{Int, Float64}`: Betweenness centrality scores
"""
function betweenness_centrality(g::AbstractGraph)
    # Implementation
end
```

## 🧪 Testing

### Running Tests

```bash
# Python tests
python tests/test_python_integration.py

# Julia tests
julia tests/test_centrality.jl
julia tests/test_community.jl
julia tests/test_pagerank.jl
```

### Writing Tests

- Add tests for all new features
- Ensure edge cases are covered
- Test both valid and invalid inputs
- Aim for high code coverage

## 📚 Documentation

### Code Documentation

- All public functions must have docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

### README Updates

If your contribution affects user-facing features:
- Update the README.md
- Add usage examples
- Update feature lists

## 🔄 Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements

### Commit Messages

Write clear, descriptive commit messages:
```
Add betweenness centrality calculation

- Implement Brandes' algorithm
- Add unit tests
- Update documentation
```

## 🏷️ Release Process

1. Update version in relevant files
2. Update CHANGELOG.md
3. Create a release tag
4. Build and test release
5. Publish release notes

## 📋 Checklist

Before submitting a pull request:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No merge conflicts with master
- [ ] PR description clearly explains changes

## 💬 Communication

- Use GitHub issues for bug reports and feature requests
- Be respectful and constructive in discussions
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- CHANGELOG.md
- Release notes
- README.md (for significant contributions)

Thank you for contributing to Graph Analytics Platform!
