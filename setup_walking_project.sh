# #!/bin/bash
# #
# # WALKING Project Setup Script
# # Adapted for CUDA 11.5 (local) + Python 3.12
# # Uses PyTorch with CUDA 11.8 (forward compatible)
# #
# # Usage: bash setup_walking_project.sh

# set -e  # Exit on error

# # Configuration variables
# PROJECT_NAME="walking3"
# VENV_NAME="walking_env"
# FORCE_REINSTALL=${FORCE_REINSTALL:-false}
# PROJECT_DIR="/root/dylan/icml2026/WALKING"

# # Color output for better readability
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# # Logging functions
# log_info() {
#     echo -e "${GREEN}[INFO]${NC} $1"
# }

# log_warn() {
#     echo -e "${YELLOW}[WARN]${NC} $1"
# }

# log_error() {
#     echo -e "${RED}[ERROR]${NC} $1"
# }

# log_step() {
#     echo -e "${BLUE}[STEP]${NC} $1"
# }

# # Function to check if conda is available
# check_conda() {
#     if command -v conda &> /dev/null; then
#         log_info "Conda is available"
#         return 0
#     else
#         log_warn "Conda is not available"
#         return 1
#     fi
# }

# # Main setup function
# setup_environment() {
#     log_step "Setting up WALKING project environment..."
    
#     # Navigate to project directory
#     if [ ! -d "$PROJECT_DIR" ]; then
#         log_error "Project directory $PROJECT_DIR not found!"
#         exit 1
#     fi
    
#     cd "$PROJECT_DIR"
#     log_info "Working in directory: $(pwd)"
    
#     # Load system modules
#     log_step "Loading system modules..."
#     if command -v module &> /dev/null; then
#         module purge 2>/dev/null || true
#         module load binutils/2.38 gcc/10.4.0-5erhxvw 2>/dev/null || log_warn "Could not load binutils/gcc modules"
        
#         log_info "Loaded modules:"
#         module list 2>/dev/null || true
#     else
#         log_warn "Module system not available, using system defaults"
#     fi
    
#     # Check CUDA
#     if command -v nvcc &> /dev/null; then
#         cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
#         log_info "Local CUDA version: $cuda_version"
#     else
#         log_warn "NVCC not found in PATH"
#     fi
    
#     # Check NVIDIA driver
#     if command -v nvidia-smi &> /dev/null; then
#         driver_cuda=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
#         log_info "Driver supports CUDA: $driver_cuda"
#     fi
    
#     # Check Rust
#     if command -v rustc &> /dev/null; then
#         rust_version=$(rustc --version)
#         log_info "Rust version: $rust_version"
#     else
#         log_error "Rust not found! Please install Rust first:"
#         log_error "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
#         exit 1
#     fi
# }

# # Function to setup Python environment with conda
# setup_python_env_conda() {
#     log_step "Setting up Python environment with Conda..."
    
#     local conda_env_name="walking3"
    
#     # Check if environment exists
#     if conda env list | grep -q "^${conda_env_name} "; then
#         if [ "$FORCE_REINSTALL" = "true" ]; then
#             log_info "Removing existing conda environment..."
#             conda env remove -n $conda_env_name -y
#         else
#             log_info "Conda environment '$conda_env_name' already exists"
#             log_info "Activate it with: conda activate $conda_env_name"
#             log_info "To force reinstall, run: FORCE_REINSTALL=true bash setup_walking_project.sh"
#             return 0
#         fi
#     fi
    
#     log_step "Creating new conda environment with Python 3.12..."
#     conda create -n $conda_env_name python=3.12 -y
    
#     log_info "Activating conda environment..."
#     eval "$(conda shell.bash hook)"
#     conda activate $conda_env_name
    
#     # Verify Python version
#     python_version=$(python --version 2>&1)
#     log_info "Python version: $python_version"
    
#     # Install PyTorch with CUDA 11.8 (compatible with CUDA 11.5 runtime)
#     log_step "Installing PyTorch with CUDA 11.8 support..."
#     log_info "Note: CUDA 11.8 binaries are forward compatible with CUDA 11.5 runtime"
#     log_info "Using pip to avoid conda dependency conflicts..."
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
#     # Install other dependencies
#     log_step "Installing other dependencies..."
#     pip install pyyaml toml tqdm tensorboard
    
#     # Install any additional packages from environment.yml
#     if [ -f "environment.yml" ]; then
#         log_info "Found environment.yml, parsing pip dependencies..."
#         python -c "
# import yaml
# import sys

# try:
#     with open('environment.yml', 'r') as f:
#         env_data = yaml.safe_load(f)
    
#     pip_deps = []
#     if 'dependencies' in env_data:
#         for dep in env_data['dependencies']:
#             if isinstance(dep, dict) and 'pip' in dep:
#                 pip_deps.extend(dep['pip'])
    
#     if pip_deps:
#         print('Installing pip dependencies from environment.yml:')
#         for dep in pip_deps:
#             print(f'  {dep}')
        
#         with open('.pip_deps_temp.txt', 'w') as f:
#             for dep in pip_deps:
#                 f.write(dep + '\n')
#     else:
#         print('No pip dependencies found in environment.yml')
        
# except Exception as e:
#     print(f'Could not parse environment.yml: {e}')
#     sys.exit(1)
# " && pip install -r .pip_deps_temp.txt && rm -f .pip_deps_temp.txt || log_warn "Could not install environment.yml dependencies"
#     fi
    
#     log_info "Python environment setup completed"
# }

# # Function to setup Python environment with venv (fallback)
# setup_python_env_venv() {
#     log_step "Setting up Python virtual environment (venv)..."
    
#     # Check for Python 3.12
#     if command -v python3.12 &> /dev/null; then
#         PYTHON_CMD="python3.12"
#     elif command -v python3 &> /dev/null; then
#         python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
#         if [[ $(echo "$python_version >= 3.12" | bc -l) -eq 1 ]]; then
#             PYTHON_CMD="python3"
#         else
#             log_error "Python 3.12+ required, found Python $python_version"
#             log_error "Please install Python 3.12 or use conda"
#             exit 1
#         fi
#     else
#         log_error "Python 3 not found!"
#         exit 1
#     fi
    
#     python_version=$($PYTHON_CMD --version 2>&1)
#     log_info "Python version: $python_version"
    
#     # Remove old environment if needed
#     if [ "$FORCE_REINSTALL" = "true" ] && [ -d "$VENV_NAME" ]; then
#         log_info "Removing old virtual environment..."
#         rm -rf $VENV_NAME
#     fi
    
#     # Create virtual environment
#     if [ ! -d "$VENV_NAME" ]; then
#         log_info "Creating new virtual environment..."
#         $PYTHON_CMD -m venv $VENV_NAME
#     fi
    
#     source $VENV_NAME/bin/activate
    
#     # Install pip
#     log_step "Installing/upgrading pip..."
#     python -m pip install --upgrade pip setuptools wheel
    
#     # Set pip cache directory
#     export PIP_CACHE_DIR=$PWD/.pip_cache
#     mkdir -p $PIP_CACHE_DIR
    
#     # Install PyTorch with CUDA 11.8 (compatible with CUDA 11.5)
#     log_step "Installing PyTorch with CUDA 11.8 support..."
#     log_info "Note: CUDA 11.8 binaries are forward compatible with CUDA 11.5 runtime"
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
#     # Install other dependencies
#     log_step "Installing other dependencies..."
#     pip install pyyaml toml tqdm tensorboard
    
#     log_info "Python environment setup completed"
# }

# # Function to setup project configuration
# setup_project_config() {
#     log_step "Setting up project configuration..."
    
#     # Copy config file if it doesn't exist
#     if [ ! -f "config.toml" ]; then
#         if [ -f "walking/config.example.toml" ]; then
#             cp walking/config.example.toml config.toml
#             log_info "Created config.toml from example"
#         elif [ -f "walking/config.toml" ]; then
#             cp walking/config.toml config.toml
#             log_info "Copied config.toml from walking directory"
#         else
#             log_warn "No config file found, you may need to create config.toml manually"
#         fi
#     else
#         log_info "config.toml already exists"
#     fi
    
#     # Ensure workspace directories exist
#     log_step "Creating workspace directories..."
#     mkdir -p workdir/{1v3,buffer,checkpoints,dataset,drain,logs,tensorboard,test_play,train_play}
#     log_info "Workspace directories created"
# }

# # Function to build Rust components
# build_rust_components() {
#     log_step "Building Rust components..."
    
#     # Set environment variables for Rust compilation
#     export CARGO_TARGET_DIR=target
    
#     # Build libriichi with release profile
#     log_info "Building libriichi library..."
#     if cargo build --release -p libriichi --lib; then
#         log_info "Rust build completed successfully"
        
#         # Copy library to Python directory
#         if [ -f "target/release/libriichi.so" ]; then
#             cp target/release/libriichi.so walking/libriichi.so
#             log_info "libriichi.so copied to walking directory"
#         elif [ -f "target/release/libriichi.dll" ]; then
#             cp target/release/libriichi.dll walking/libriichi.pyd
#             log_info "libriichi.dll copied as walking/libriichi.pyd"
#         else
#             log_error "Could not find compiled libriichi library"
#             log_error "Expected at: target/release/libriichi.so"
#             exit 1
#         fi
#     else
#         log_error "Rust build failed"
#         exit 1
#     fi
# }

# # Function to verify installation
# verify_installation() {
#     log_step "Verifying installation..."
    
#     python -c "
# import sys
# print('=' * 50)
# print('WALKING Project Installation Verification')
# print('=' * 50)
# print(f'Python version: {sys.version}')

# # Check core dependencies
# deps_status = {}
# critical_deps = [
#     ('torch', 'PyTorch'),
#     ('numpy', 'NumPy'),
#     ('yaml', 'PyYAML'),
#     ('toml', 'TOML'),
# ]

# for module, name in critical_deps:
#     try:
#         if module == 'yaml':
#             import yaml as mod
#         elif module == 'toml':
#             import toml as mod
#         else:
#             mod = __import__(module)
#         version = getattr(mod, '__version__', 'unknown')
#         print(f'✓ {name}: {version}')
#         deps_status[module] = True
#     except Exception as e:
#         print(f'✗ {name}: {e}')
#         deps_status[module] = False

# # PyTorch CUDA check
# if deps_status.get('torch', False):
#     import torch
#     print(f'✓ PyTorch CUDA available: {torch.cuda.is_available()}')
#     print(f'✓ PyTorch built with CUDA: {torch.version.cuda}')
#     if torch.cuda.is_available():
#         print(f'✓ CUDA device count: {torch.cuda.device_count()}')
#         for i in range(torch.cuda.device_count()):
#             try:
#                 name = torch.cuda.get_device_name(i)
#                 props = torch.cuda.get_device_properties(i)
#                 memory_gb = props.total_memory / 1024**3
#                 compute_cap = f'sm_{props.major}{props.minor}'
#                 print(f'  GPU {i}: {name} ({memory_gb:.1f} GB, {compute_cap})')
#             except Exception as e:
#                 print(f'  GPU {i}: Error - {e}')
#     else:
#         print('⚠️  CUDA not available - check drivers and CUDA installation')

# # libriichi check
# try:
#     import sys
#     import os
#     sys.path.insert(0, os.path.join(os.getcwd(), 'walking'))
#     import libriichi
#     print('✓ libriichi imported successfully')
#     print('✓ All core components verified!')
# except Exception as e:
#     print(f'✗ libriichi import failed: {e}')
#     print('   This error indicates Python version incompatibility')
#     print('   Make sure you are using Python 3.12+')

# print('=' * 50)
# "
# }

# # Function to display usage instructions
# show_usage() {
#     echo ""
#     log_info "WALKING Project Setup Complete!"
#     echo ""
    
#     if check_conda; then
#         echo "To use the project (Conda):"
#         echo "1. Activate the environment:"
#         echo "   conda activate walking3"
#         echo ""
#         echo "2. Navigate to project directory:"
#         echo "   cd $PROJECT_DIR"
#     else
#         echo "To use the project (venv):"
#         echo "1. Activate the environment:"
#         echo "   cd $PROJECT_DIR"
#         echo "   source $VENV_NAME/bin/activate"
#     fi
    
#     echo ""
#     echo "3. Run the project:"
#     echo "   python walking/train.py        # Start training"
#     echo "   python walking/server.py       # Start server"
#     echo "   python walking/client.py       # Start client"
#     echo ""
#     echo "4. Configuration:"
#     echo "   Edit config.toml to customize settings"
#     echo ""
#     echo "5. Logs and outputs:"
#     echo "   Check workdir/ for training outputs"
#     echo "   Check logs/ for training logs"
# }

# # Main execution
# main() {
#     echo "======================================================================"
#     log_info "WALKING Project Setup Script"
#     log_info "Mortal Mahjong AI - CUDA 11.5 + Python 3.12"
#     log_info "Using PyTorch with CUDA 11.8 (forward compatible)"
#     echo "======================================================================"
#     echo ""
    
#     setup_environment
    
#     # Choose setup method based on conda availability
#     if check_conda; then
#         log_info "Using Conda for environment management"
#         setup_python_env_conda
#     else
#         log_info "Using venv for environment management"
#         setup_python_env_venv
#     fi
    
#     setup_project_config
#     build_rust_components
#     verify_installation
#     show_usage
    
#     log_info "Setup completed successfully!"
# }

# # Error handling
# trap 'log_error "Setup failed at line $LINENO. Check error messages above."' ERR

# # Run main function
# main "$@"


#!/bin/bash
#
# WALKING Project Setup - Fixed Version for Ubuntu
# 修正了 Python 安装和 GPU 驱动问题
# 
# Usage: sudo bash setup_walking_project.sh

set -e

PROJECT_DIR="$HOME/dylan/icml2026/WALKING"
VENV_NAME="walking_env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# ============================================================
# Check if running as root
# ============================================================
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        log_error "Please run as root or with sudo"
        exit 1
    fi
}

# ============================================================
# Detect GPU
# ============================================================
detect_gpu() {
    log_step "Detecting GPU hardware..."
    
    # Check for NVIDIA GPU
    if lspci | grep -i nvidia > /dev/null 2>&1; then
        GPU_MODEL=$(lspci | grep -i nvidia | grep -i '3D\|VGA\|Display' | head -1)
        log_info "NVIDIA GPU detected: $GPU_MODEL"
        HAS_NVIDIA_GPU=true
        
        # Check device ID
        DEVICE_ID=$(lspci -n | grep -i nvidia | grep -i '3D\|VGA' | awk '{print $3}' | cut -d':' -f2)
        log_info "Device ID: $DEVICE_ID"
    else
        log_warn "No NVIDIA GPU detected"
        HAS_NVIDIA_GPU=false
    fi
}

# ============================================================
# Install NVIDIA Driver
# ============================================================
install_nvidia_driver() {
    if [ "$HAS_NVIDIA_GPU" != "true" ]; then
        log_warn "Skipping NVIDIA driver installation (no GPU detected)"
        return 0
    fi
    
    log_step "Installing NVIDIA driver..."
    
    # Check if driver already installed
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        log_info "NVIDIA driver already working"
        nvidia-smi
        return 0
    fi
    
    # Update package list
    apt-get update
    
    # Install ubuntu-drivers-common
    apt-get install -y ubuntu-drivers-common
    
    # Detect recommended driver
    log_info "Detecting recommended NVIDIA driver..."
    RECOMMENDED_DRIVER=$(ubuntu-drivers devices 2>/dev/null | grep 'recommended' | awk '{print $3}' | head -1)
    
    if [ -z "$RECOMMENDED_DRIVER" ]; then
        log_warn "Could not auto-detect driver, installing nvidia-driver-535-server"
        RECOMMENDED_DRIVER="nvidia-driver-535-server"
    fi
    
    log_info "Installing driver: $RECOMMENDED_DRIVER"
    
    # Install the driver
    apt-get install -y "$RECOMMENDED_DRIVER" || {
        log_error "Failed to install $RECOMMENDED_DRIVER"
        log_info "Trying alternative installation method..."
        ubuntu-drivers autoinstall
    }
    
    log_warn "NVIDIA driver installed. System REBOOT REQUIRED!"
    log_warn "After reboot, run 'nvidia-smi' to verify installation"
}

# ============================================================
# Install System Dependencies
# ============================================================
install_system_deps() {
    log_step "Installing system dependencies..."
    
    apt-get update
    apt-get install -y \
        software-properties-common \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        git \
        curl \
        wget \
        ca-certificates \
        libssl-dev \
        pkg-config \
        libffi-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        zlib1g-dev
    
    log_info "System dependencies installed"
}

# ============================================================
# Install Python 3.12 via deadsnakes PPA
# ============================================================
install_python312() {
    log_step "Installing Python 3.12..."
    
    if command -v python3.12 &> /dev/null; then
        log_info "Python 3.12 already installed: $(python3.12 --version)"
        return 0
    fi
    
    # Add deadsnakes PPA
    log_info "Adding deadsnakes PPA for Python 3.12..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    
    # Install Python 3.12
    apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3.12-distutils
    
    # Install pip for Python 3.12
    log_info "Installing pip for Python 3.12..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3.12 /tmp/get-pip.py
    rm /tmp/get-pip.py
    
    log_info "Python 3.12 installed: $(python3.12 --version)"
}

# ============================================================
# Install Rust
# ============================================================
install_rust() {
    log_step "Installing Rust..."
    
    # Get the actual user (not root)
    ACTUAL_USER=${SUDO_USER:-$USER}
    ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
    
    # Check if already installed
    if su - $ACTUAL_USER -c "command -v rustc" &> /dev/null; then
        RUST_VERSION=$(su - $ACTUAL_USER -c "rustc --version")
        log_info "Rust already installed: $RUST_VERSION"
        return 0
    fi
    
    log_info "Installing Rust for user: $ACTUAL_USER"
    
    # Install Rust as the actual user
    su - $ACTUAL_USER -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
    
    RUST_VERSION=$(su - $ACTUAL_USER -c "source $ACTUAL_HOME/.cargo/env && rustc --version")
    log_info "Rust installed: $RUST_VERSION"
}

# ============================================================
# Setup Virtual Environment
# ============================================================
setup_venv() {
    log_step "Setting up Python virtual environment..."
    
    # Get actual user
    ACTUAL_USER=${SUDO_USER:-$USER}
    
    cd "$PROJECT_DIR"
    
    # Create venv as actual user
    if [ ! -d "$VENV_NAME" ]; then
        su - $ACTUAL_USER -c "cd $PROJECT_DIR && python3.12 -m venv $VENV_NAME"
        log_info "Virtual environment created"
    fi
    
    log_info "Virtual environment ready at: $PROJECT_DIR/$VENV_NAME"
}

# ============================================================
# Install PyTorch
# ============================================================
install_pytorch() {
    log_step "Installing PyTorch..."
    
    ACTUAL_USER=${SUDO_USER:-$USER}
    
    cd "$PROJECT_DIR"
    
    # Determine CUDA support
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        log_info "GPU detected, installing PyTorch with CUDA 11.8..."
        PIP_CMD="$PROJECT_DIR/$VENV_NAME/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else
        log_info "No working GPU, installing CPU-only PyTorch..."
        PIP_CMD="$PROJECT_DIR/$VENV_NAME/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    fi
    
    su - $ACTUAL_USER -c "cd $PROJECT_DIR && $PIP_CMD"
    log_info "PyTorch installed"
}

# ============================================================
# Install Python Dependencies
# ============================================================
install_python_deps() {
    log_step "Installing Python dependencies..."
    
    ACTUAL_USER=${SUDO_USER:-$USER}
    cd "$PROJECT_DIR"
    
    # Core dependencies
    su - $ACTUAL_USER -c "cd $PROJECT_DIR && $PROJECT_DIR/$VENV_NAME/bin/pip install --upgrade pip setuptools wheel"
    su - $ACTUAL_USER -c "cd $PROJECT_DIR && $PROJECT_DIR/$VENV_NAME/bin/pip install numpy pyyaml toml tqdm tensorboard"
    
    # Install from requirements.txt if exists
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        su - $ACTUAL_USER -c "cd $PROJECT_DIR && $PROJECT_DIR/$VENV_NAME/bin/pip install -r requirements.txt" || log_warn "Some requirements failed"
    fi
    
    log_info "Dependencies installed"
}

# ============================================================
# Build Rust Components
# ============================================================
build_rust() {
    log_step "Building Rust components..."
    
    ACTUAL_USER=${SUDO_USER:-$USER}
    ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
    
    cd "$PROJECT_DIR"
    
    export CARGO_TARGET_DIR="$PROJECT_DIR/target"
    
    log_info "Building libriichi..."
    su - $ACTUAL_USER -c "cd $PROJECT_DIR && source $ACTUAL_HOME/.cargo/env && CARGO_TARGET_DIR=$PROJECT_DIR/target cargo build --release -p libriichi --lib"
    
    # Find and copy library
    if [ -f "target/release/libriichi.so" ]; then
        cp target/release/libriichi.so walking/libriichi.so
        log_info "Copied libriichi.so to walking/"
    elif [ -f "target/release/liblibriichi.so" ]; then
        cp target/release/liblibriichi.so walking/libriichi.so
        log_info "Copied liblibriichi.so to walking/"
    else
        log_error "Could not find compiled library"
        ls -la target/release/ | grep libriichi || true
        exit 1
    fi
}

# ============================================================
# Setup Project Structure
# ============================================================
setup_project() {
    log_step "Setting up project structure..."
    
    ACTUAL_USER=${SUDO_USER:-$USER}
    cd "$PROJECT_DIR"
    
    # Create directories
    su - $ACTUAL_USER -c "cd $PROJECT_DIR && mkdir -p workdir/{1v3,buffer,checkpoints,dataset,drain,logs,tensorboard,test_play,train_play}"
    
    # Copy config
    if [ ! -f "config.toml" ]; then
        if [ -f "walking/config.example.toml" ]; then
            su - $ACTUAL_USER -c "cd $PROJECT_DIR && cp walking/config.example.toml config.toml"
            log_info "Created config.toml from example"
        elif [ -f "walking/config.toml" ]; then
            su - $ACTUAL_USER -c "cd $PROJECT_DIR && cp walking/config.toml ."
            log_info "Copied config.toml"
        fi
    fi
    
    log_info "Project structure ready"
}

# ============================================================
# Verify Installation
# ============================================================
verify_installation() {
    log_step "Verifying installation..."
    
    ACTUAL_USER=${SUDO_USER:-$USER}
    cd "$PROJECT_DIR"
    
    su - $ACTUAL_USER -c "cd $PROJECT_DIR && $PROJECT_DIR/$VENV_NAME/bin/python3 << 'PYEOF'
import sys
print('=' * 60)
print('WALKING Project Installation Verification')
print('=' * 60)
print(f'Python: {sys.version}')
print()

# Check dependencies
deps = [('torch', 'PyTorch'), ('numpy', 'NumPy'), ('yaml', 'PyYAML'), ('toml', 'TOML')]
for module, name in deps:
    try:
        if module == 'yaml':
            import yaml as mod
        else:
            mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f'✓ {name}: {version}')
    except Exception as e:
        print(f'✗ {name}: {e}')

# Check PyTorch CUDA
try:
    import torch
    print(f'\nPyTorch CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except Exception as e:
    print(f'PyTorch check error: {e}')

# Check libriichi
try:
    sys.path.insert(0, 'walking')
    import libriichi
    print('\n✓ libriichi imported successfully')
    print('✓ All core components verified!')
except Exception as e:
    print(f'\n✗ libriichi import failed: {e}')
    print('   Make sure you are using Python 3.12+')

print('=' * 60)
PYEOF
"
}

# ============================================================
# Show Usage
# ============================================================
show_usage() {
    echo ""
    log_info "=== WALKING Project Setup Complete ==="
    echo ""
    echo "To activate and use:"
    echo "  cd $PROJECT_DIR"
    echo "  source $VENV_NAME/bin/activate"
    echo ""
    echo "To run:"
    echo "  python walking/train.py       # Start training"
    echo "  python walking/server.py      # Start server"
    echo "  python walking/client.py      # Start client"
    echo ""
    
    if ! nvidia-smi &> /dev/null 2>&1; then
        log_warn "============================================"
        log_warn "GPU driver installed but not active yet"
        log_warn "Please REBOOT the system to activate:"
        log_warn "  sudo reboot"
        log_warn "After reboot, verify with: nvidia-smi"
        log_warn "============================================"
    else
        log_info "GPU is ready!"
        nvidia-smi
    fi
}

# ============================================================
# Main
# ============================================================
main() {
    echo "======================================================================"
    log_info "WALKING Project Setup Script - Fixed Version"
    log_info "For Ubuntu with NVIDIA GPU Support"
    echo "======================================================================"
    echo ""
    
    # Check root
    check_root
    
    # Detect GPU
    detect_gpu
    
    # Check project directory
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project directory not found: $PROJECT_DIR"
        log_error "Please clone the repository first"
        exit 1
    fi
    
    # Install system dependencies
    install_system_deps
    
    # Install NVIDIA driver
    install_nvidia_driver
    
    # Install Python 3.12
    install_python312
    
    # Install Rust
    install_rust
    
    # Setup virtual environment
    setup_venv
    
    # Install PyTorch
    install_pytorch
    
    # Install Python dependencies
    install_python_deps
    
    # Build Rust components
    build_rust
    
    # Setup project structure
    setup_project
    
    # Verify installation
    verify_installation
    
    # Show usage
    show_usage
    
    log_info "Setup completed successfully!"
}

# Error handling
trap 'log_error "Setup failed at line $LINENO. Check error messages above."' ERR

# Run main
main "$@"