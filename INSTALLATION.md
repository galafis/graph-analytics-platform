# Installation Guide

Complete installation guide for Graph Analytics Platform.

## Prerequisites

### System Requirements
- **Operating System:** Linux, macOS, or Windows
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 2GB free space

### Software Requirements
- **Python:** 3.8 or higher
- **Julia:** 1.9 or higher
- **Git:** For cloning the repository

## Step-by-Step Installation

### 1. Install Python

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip
```

#### macOS
```bash
brew install python3
```

#### Windows
Download and install from [python.org](https://www.python.org/downloads/)

### 2. Install Julia

#### Linux
```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar -xvzf julia-1.9.4-linux-x86_64.tar.gz
sudo mv julia-1.9.4 /opt/
sudo ln -s /opt/julia-1.9.4/bin/julia /usr/local/bin/julia
```

#### macOS
```bash
brew install julia
```

#### Windows
Download and install from [julialang.org](https://julialang.org/downloads/)

### 3. Clone Repository

```bash
git clone https://github.com/galafis/graph-analytics-platform.git
cd graph-analytics-platform
```

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Install Julia Packages

```bash
julia -e 'using Pkg; Pkg.add(["Graphs", "LinearAlgebra", "SparseArrays", "DataStructures", "Statistics", "Random"])'
```

Or interactively:
```julia
julia
> using Pkg
> Pkg.add("Graphs")
> Pkg.add("LinearAlgebra")
> Pkg.add("SparseArrays")
> Pkg.add("DataStructures")
> Pkg.add("Statistics")
> Pkg.add("Random")
```

### 6. Verify Installation

#### Test Python
```bash
python3 tests/test_python_integration.py
```

Expected output:
```
============================================================
Running Python Integration Tests
============================================================
...
Ran 15 tests in X.XXs
OK
✓ All Python tests completed!
```

#### Test Julia
```bash
julia tests/test_centrality.jl
```

Expected output:
```
Test Summary:  | Pass  Total  Time
Centrality Tests |   10     10  X.Xs
✓ All centrality tests passed!
```

### 7. Generate Sample Data

```bash
python3 data/generate_sample_data.py
```

## Optional: Julia-Python Integration

For seamless Julia-Python integration:

```bash
pip install julia
python3 -c "import julia; julia.install()"
```

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Julia 1.9+ installed
- [ ] Repository cloned
- [ ] Python dependencies installed
- [ ] Julia packages installed
- [ ] Python tests pass
- [ ] Julia tests pass
- [ ] Sample data generated

## Troubleshooting

### Issue: Julia not in PATH
```bash
# Find Julia installation
which julia

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$PATH:/path/to/julia/bin"
```

### Issue: Python package conflicts
```bash
# Use virtual environment
python3 -m venv venv --clear
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Julia package installation fails
```julia
# Update Julia packages
using Pkg
Pkg.update()
Pkg.gc()  # Clean up old packages
Pkg.add("Graphs")  # Try again
```

### Issue: ImportError in Python
```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Next Steps

1. **Run Examples**
   ```bash
   python3 examples/social_network_analysis.py
   julia examples/citation_network.jl
   ```

2. **Read Documentation**
   - [README.md](README.md) - Main documentation
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
   - [examples/](examples/) - Code examples

3. **Explore Sample Data**
   ```bash
   cd data/sample_networks
   ls -la
   ```

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/galafis/graph-analytics-platform/issues)
- **Discussions:** [GitHub Discussions](https://github.com/galafis/graph-analytics-platform/discussions)

## Uninstallation

### Remove Python packages
```bash
pip uninstall -r requirements.txt -y
```

### Remove Julia packages
```julia
using Pkg
Pkg.rm(["Graphs", "LinearAlgebra", "SparseArrays", "DataStructures"])
```

### Remove repository
```bash
rm -rf graph-analytics-platform
```
