#!/bin/bash
#
# WALKING Project Setup Script
# Adapted for CUDA 11.5 (local) + Python 3.12
# Uses PyTorch with CUDA 11.8 (forward compatible)
#
# Usage: bash setup_walking_project.sh

set -e  # Exit on error

# Configuration variables
PROJECT_NAME="walking3"
VENV_NAME="walking_env"
FORCE_REINSTALL=${FORCE_REINSTALL:-false}
PROJECT_DIR="/data/jiacheng/system/cache/temp/icml2026/WALKING"

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        log_info "Conda is available"
        return 0
    else
        log_warn "Conda is not available"
        return 1
    fi
}

# Main setup function
setup_environment() {
    log_step "Setting up WALKING project environment..."
    
    # Navigate to project directory
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project directory $PROJECT_DIR not found!"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log_info "Working in directory: $(pwd)"
    
    # Load system modules
    log_step "Loading system modules..."
    if command -v module &> /dev/null; then
        module purge 2>/dev/null || true
        module load binutils/2.38 gcc/10.4.0-5erhxvw 2>/dev/null || log_warn "Could not load binutils/gcc modules"
        
        log_info "Loaded modules:"
        module list 2>/dev/null || true
    else
        log_warn "Module system not available, using system defaults"
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "Local CUDA version: $cuda_version"
    else
        log_warn "NVCC not found in PATH"
    fi
    
    # Check NVIDIA driver
    if command -v nvidia-smi &> /dev/null; then
        driver_cuda=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log_info "Driver supports CUDA: $driver_cuda"
    fi
    
    # Check Rust
    if command -v rustc &> /dev/null; then
        rust_version=$(rustc --version)
        log_info "Rust version: $rust_version"
    else
        log_error "Rust not found! Please install Rust first:"
        log_error "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
}

# Function to setup Python environment with conda
setup_python_env_conda() {
    log_step "Setting up Python environment with Conda..."
    
    local conda_env_name="walking3"
    
    # Check if environment exists
    if conda env list | grep -q "^${conda_env_name} "; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            log_info "Removing existing conda environment..."
            conda env remove -n $conda_env_name -y
        else
            log_info "Conda environment '$conda_env_name' already exists"
            log_info "Activate it with: conda activate $conda_env_name"
            log_info "To force reinstall, run: FORCE_REINSTALL=true bash setup_walking_project.sh"
            return 0
        fi
    fi
    
    log_step "Creating new conda environment with Python 3.12..."
    conda create -n $conda_env_name python=3.12 -y
    
    log_info "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate $conda_env_name
    
    # Verify Python version
    python_version=$(python --version 2>&1)
    log_info "Python version: $python_version"
    
    # Install PyTorch with CUDA 11.8 (compatible with CUDA 11.5 runtime)
    log_step "Installing PyTorch with CUDA 11.8 support..."
    log_info "Note: CUDA 11.8 binaries are forward compatible with CUDA 11.5 runtime"
    log_info "Using pip to avoid conda dependency conflicts..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other dependencies
    log_step "Installing other dependencies..."
    pip install pyyaml toml tqdm tensorboard
    
    # Install any additional packages from environment.yml
    if [ -f "environment.yml" ]; then
        log_info "Found environment.yml, parsing pip dependencies..."
        python -c "
import yaml
import sys

try:
    with open('environment.yml', 'r') as f:
        env_data = yaml.safe_load(f)
    
    pip_deps = []
    if 'dependencies' in env_data:
        for dep in env_data['dependencies']:
            if isinstance(dep, dict) and 'pip' in dep:
                pip_deps.extend(dep['pip'])
    
    if pip_deps:
        print('Installing pip dependencies from environment.yml:')
        for dep in pip_deps:
            print(f'  {dep}')
        
        with open('.pip_deps_temp.txt', 'w') as f:
            for dep in pip_deps:
                f.write(dep + '\n')
    else:
        print('No pip dependencies found in environment.yml')
        
except Exception as e:
    print(f'Could not parse environment.yml: {e}')
    sys.exit(1)
" && pip install -r .pip_deps_temp.txt && rm -f .pip_deps_temp.txt || log_warn "Could not install environment.yml dependencies"
    fi
    
    log_info "Python environment setup completed"
}

# Function to setup Python environment with venv (fallback)
setup_python_env_venv() {
    log_step "Setting up Python virtual environment (venv)..."
    
    # Check for Python 3.12
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$python_version >= 3.12" | bc -l) -eq 1 ]]; then
            PYTHON_CMD="python3"
        else
            log_error "Python 3.12+ required, found Python $python_version"
            log_error "Please install Python 3.12 or use conda"
            exit 1
        fi
    else
        log_error "Python 3 not found!"
        exit 1
    fi
    
    python_version=$($PYTHON_CMD --version 2>&1)
    log_info "Python version: $python_version"
    
    # Remove old environment if needed
    if [ "$FORCE_REINSTALL" = "true" ] && [ -d "$VENV_NAME" ]; then
        log_info "Removing old virtual environment..."
        rm -rf $VENV_NAME
    fi
    
    # Create virtual environment
    if [ ! -d "$VENV_NAME" ]; then
        log_info "Creating new virtual environment..."
        $PYTHON_CMD -m venv $VENV_NAME
    fi
    
    source $VENV_NAME/bin/activate
    
    # Install pip
    log_step "Installing/upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Set pip cache directory
    export PIP_CACHE_DIR=$PWD/.pip_cache
    mkdir -p $PIP_CACHE_DIR
    
    # Install PyTorch with CUDA 11.8 (compatible with CUDA 11.5)
    log_step "Installing PyTorch with CUDA 11.8 support..."
    log_info "Note: CUDA 11.8 binaries are forward compatible with CUDA 11.5 runtime"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other dependencies
    log_step "Installing other dependencies..."
    pip install pyyaml toml tqdm tensorboard
    
    log_info "Python environment setup completed"
}

# Function to setup project configuration
setup_project_config() {
    log_step "Setting up project configuration..."
    
    # Copy config file if it doesn't exist
    if [ ! -f "config.toml" ]; then
        if [ -f "walking/config.example.toml" ]; then
            cp walking/config.example.toml config.toml
            log_info "Created config.toml from example"
        elif [ -f "walking/config.toml" ]; then
            cp walking/config.toml config.toml
            log_info "Copied config.toml from walking directory"
        else
            log_warn "No config file found, you may need to create config.toml manually"
        fi
    else
        log_info "config.toml already exists"
    fi
    
    # Ensure workspace directories exist
    log_step "Creating workspace directories..."
    mkdir -p workdir/{1v3,buffer,checkpoints,dataset,drain,logs,tensorboard,test_play,train_play}
    log_info "Workspace directories created"
}

# Function to build Rust components
build_rust_components() {
    log_step "Building Rust components..."
    
    # Set environment variables for Rust compilation
    export CARGO_TARGET_DIR=target
    
    # Build libriichi with release profile
    log_info "Building libriichi library..."
    if cargo build --release -p libriichi --lib; then
        log_info "Rust build completed successfully"
        
        # Copy library to Python directory
        if [ -f "target/release/libriichi.so" ]; then
            cp target/release/libriichi.so walking/libriichi.so
            log_info "libriichi.so copied to walking directory"
        elif [ -f "target/release/libriichi.dll" ]; then
            cp target/release/libriichi.dll walking/libriichi.pyd
            log_info "libriichi.dll copied as walking/libriichi.pyd"
        else
            log_error "Could not find compiled libriichi library"
            log_error "Expected at: target/release/libriichi.so"
            exit 1
        fi
    else
        log_error "Rust build failed"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    log_step "Verifying installation..."
    
    python -c "
import sys
print('=' * 50)
print('WALKING Project Installation Verification')
print('=' * 50)
print(f'Python version: {sys.version}')

# Check core dependencies
deps_status = {}
critical_deps = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('yaml', 'PyYAML'),
    ('toml', 'TOML'),
]

for module, name in critical_deps:
    try:
        if module == 'yaml':
            import yaml as mod
        elif module == 'toml':
            import toml as mod
        else:
            mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
        deps_status[module] = True
    except Exception as e:
        print(f'✗ {name}: {e}')
        deps_status[module] = False

# PyTorch CUDA check
if deps_status.get('torch', False):
    import torch
    print(f'✓ PyTorch CUDA available: {torch.cuda.is_available()}')
    print(f'✓ PyTorch built with CUDA: {torch.version.cuda}')
    if torch.cuda.is_available():
        print(f'✓ CUDA device count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                compute_cap = f'sm_{props.major}{props.minor}'
                print(f'  GPU {i}: {name} ({memory_gb:.1f} GB, {compute_cap})')
            except Exception as e:
                print(f'  GPU {i}: Error - {e}')
    else:
        print('⚠️  CUDA not available - check drivers and CUDA installation')

# libriichi check
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.getcwd(), 'walking'))
    import libriichi
    print('✓ libriichi imported successfully')
    print('✓ All core components verified!')
except Exception as e:
    print(f'✗ libriichi import failed: {e}')
    print('   This error indicates Python version incompatibility')
    print('   Make sure you are using Python 3.12+')

print('=' * 50)
"
}

# Function to display usage instructions
show_usage() {
    echo ""
    log_info "WALKING Project Setup Complete!"
    echo ""
    
    if check_conda; then
        echo "To use the project (Conda):"
        echo "1. Activate the environment:"
        echo "   conda activate walking3"
        echo ""
        echo "2. Navigate to project directory:"
        echo "   cd $PROJECT_DIR"
    else
        echo "To use the project (venv):"
        echo "1. Activate the environment:"
        echo "   cd $PROJECT_DIR"
        echo "   source $VENV_NAME/bin/activate"
    fi
    
    echo ""
    echo "3. Run the project:"
    echo "   python walking/train.py        # Start training"
    echo "   python walking/server.py       # Start server"
    echo "   python walking/client.py       # Start client"
    echo ""
    echo "4. Configuration:"
    echo "   Edit config.toml to customize settings"
    echo ""
    echo "5. Logs and outputs:"
    echo "   Check workdir/ for training outputs"
    echo "   Check logs/ for training logs"
}

# Main execution
main() {
    echo "======================================================================"
    log_info "WALKING Project Setup Script"
    log_info "Mortal Mahjong AI - CUDA 11.5 + Python 3.12"
    log_info "Using PyTorch with CUDA 11.8 (forward compatible)"
    echo "======================================================================"
    echo ""
    
    setup_environment
    
    # Choose setup method based on conda availability
    if check_conda; then
        log_info "Using Conda for environment management"
        setup_python_env_conda
    else
        log_info "Using venv for environment management"
        setup_python_env_venv
    fi
    
    setup_project_config
    build_rust_components
    verify_installation
    show_usage
    
    log_info "Setup completed successfully!"
}

# Error handling
trap 'log_error "Setup failed at line $LINENO. Check error messages above."' ERR

# Run main function
main "$@"