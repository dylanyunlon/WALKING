# #!/bin/bash
# #
# # WALKING Project Setup - Conda Version (Optimized)
# # 使用 Miniconda 管理 Python 环境，支持 NVIDIA GPU
# #
# # Usage: sudo bash setup_walking_conda.sh

# set -e

# # ============================================================
# # Configuration
# # ============================================================
# PROJECT_NAME="walking3"
# CONDA_ENV_NAME="walking3"
# PROJECT_DIR="$HOME/dylan/icml2026/WALKING"
# FORCE_REINSTALL=${FORCE_REINSTALL:-false}
# PYTHON_VERSION="3.12"

# # Colors
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m'

# log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
# log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
# log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
# log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# # ============================================================
# # Check if running as root
# # ============================================================
# check_root() {
#     if [ "$EUID" -ne 0 ]; then 
#         log_error "Please run as root or with sudo"
#         exit 1
#     fi
# }

# # ============================================================
# # Detect GPU
# # ============================================================
# detect_gpu() {
#     log_step "Detecting GPU hardware..."
    
#     if lspci | grep -i nvidia > /dev/null 2>&1; then
#         GPU_MODEL=$(lspci | grep -i nvidia | grep -i '3D\|VGA\|Display' | head -1)
#         log_info "NVIDIA GPU detected: $GPU_MODEL"
#         HAS_NVIDIA_GPU=true
        
#         # Check device ID (0x2236 is Ada Lovelace architecture)
#         DEVICE_ID=$(lspci -n | grep -i nvidia | grep -i '3D\|VGA' | awk '{print $3}' | cut -d':' -f2)
#         log_info "Device ID: $DEVICE_ID"
#     else
#         log_warn "No NVIDIA GPU detected"
#         HAS_NVIDIA_GPU=false
#     fi
    
#     # Check driver status
#     if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
#         log_info "NVIDIA driver is working"
#         GPU_AVAILABLE=true
#         nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
#     else
#         log_warn "NVIDIA driver not working or not installed"
#         GPU_AVAILABLE=false
#     fi
# }

# # ============================================================
# # Install System Dependencies
# # ============================================================
# install_system_deps() {
#     log_step "Installing system dependencies..."
    
#     apt-get update
#     apt-get install -y \
#         build-essential \
#         gcc \
#         g++ \
#         make \
#         cmake \
#         git \
#         curl \
#         wget \
#         ca-certificates \
#         libssl-dev \
#         pkg-config \
#         ubuntu-drivers-common
    
#     log_info "System dependencies installed"
# }

# # ============================================================
# # Install NVIDIA Driver
# # ============================================================
# install_nvidia_driver() {
#     if [ "$HAS_NVIDIA_GPU" != "true" ]; then
#         log_warn "Skipping NVIDIA driver installation (no GPU detected)"
#         return 0
#     fi
    
#     if [ "$GPU_AVAILABLE" = "true" ]; then
#         log_info "NVIDIA driver already working, skipping installation"
#         return 0
#     fi
    
#     log_step "Installing NVIDIA driver..."
    
#     # Detect recommended driver
#     RECOMMENDED_DRIVER=$(ubuntu-drivers devices 2>/dev/null | grep 'recommended' | awk '{print $3}' | head -1)
    
#     if [ -z "$RECOMMENDED_DRIVER" ]; then
#         log_warn "Could not auto-detect driver, using nvidia-driver-550-server (Ada Lovelace support)"
#         RECOMMENDED_DRIVER="nvidia-driver-550-server"
#     fi
    
#     log_info "Installing driver: $RECOMMENDED_DRIVER"
#     apt-get install -y "$RECOMMENDED_DRIVER" || ubuntu-drivers autoinstall
    
#     log_warn "NVIDIA driver installed. System REBOOT REQUIRED!"
#     NEEDS_REBOOT=true
# }

# # ============================================================
# # Install Miniconda
# # ============================================================
# install_miniconda() {
#     log_step "Installing Miniconda..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
#     MINICONDA_DIR="$ACTUAL_HOME/miniconda3"
    
#     # Check if conda already exists
#     if su - $ACTUAL_USER -c "command -v conda" &> /dev/null; then
#         CONDA_VERSION=$(su - $ACTUAL_USER -c "conda --version")
#         log_info "Conda already installed: $CONDA_VERSION"
#         return 0
#     fi
    
#     # Check for existing installation
#     if [ -d "$MINICONDA_DIR" ]; then
#         log_info "Found existing Miniconda at $MINICONDA_DIR"
#         su - $ACTUAL_USER -c "$MINICONDA_DIR/bin/conda init bash"
#         return 0
#     fi
    
#     # Download and install Miniconda
#     log_info "Downloading latest Miniconda..."
#     mkdir -p "$MINICONDA_DIR"
    
#     su - $ACTUAL_USER -c "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $MINICONDA_DIR/miniconda.sh"
    
#     log_info "Installing Miniconda to $MINICONDA_DIR..."
#     su - $ACTUAL_USER -c "bash $MINICONDA_DIR/miniconda.sh -b -u -p $MINICONDA_DIR"
#     su - $ACTUAL_USER -c "rm $MINICONDA_DIR/miniconda.sh"
    
#     # Initialize conda
#     su - $ACTUAL_USER -c "$MINICONDA_DIR/bin/conda init bash"
    
#     log_info "Miniconda installed successfully"
#     log_warn "Shell configuration updated. Changes will take effect in new terminals."
# }

# # ============================================================
# # Install Rust
# # ============================================================
# install_rust() {
#     log_step "Installing Rust..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
    
#     if su - $ACTUAL_USER -c "command -v rustc" &> /dev/null; then
#         RUST_VERSION=$(su - $ACTUAL_USER -c "rustc --version")
#         log_info "Rust already installed: $RUST_VERSION"
#         return 0
#     fi
    
#     log_info "Installing Rust for user: $ACTUAL_USER"
#     su - $ACTUAL_USER -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
    
#     RUST_VERSION=$(su - $ACTUAL_USER -c "source $ACTUAL_HOME/.cargo/env && rustc --version")
#     log_info "Rust installed: $RUST_VERSION"
# }

# # ============================================================
# # Setup Conda Environment
# # ============================================================
# setup_conda_env() {
#     log_step "Setting up Conda environment..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
    
#     # Source conda
#     CONDA_INIT="source $ACTUAL_HOME/miniconda3/etc/profile.d/conda.sh"
    
#     # Check if environment exists
#     ENV_EXISTS=$(su - $ACTUAL_USER -c "$CONDA_INIT && conda env list" | grep "^${CONDA_ENV_NAME} " || true)
    
#     if [ -n "$ENV_EXISTS" ]; then
#         if [ "$FORCE_REINSTALL" = "true" ]; then
#             log_info "Removing existing conda environment..."
#             su - $ACTUAL_USER -c "$CONDA_INIT && conda env remove -n $CONDA_ENV_NAME -y"
#         else
#             log_info "Conda environment '$CONDA_ENV_NAME' already exists"
#             log_info "To force reinstall: FORCE_REINSTALL=true sudo bash $0"
#             return 0
#         fi
#     fi
    
#     # Create new environment
#     log_info "Creating conda environment with Python $PYTHON_VERSION..."
#     su - $ACTUAL_USER -c "$CONDA_INIT && conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y"
    
#     log_info "Conda environment created successfully"
# }

# # ============================================================
# # Install PyTorch
# # ============================================================
# install_pytorch() {
#     log_step "Installing PyTorch..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
#     CONDA_INIT="source $ACTUAL_HOME/miniconda3/etc/profile.d/conda.sh"
    
#     if [ "$GPU_AVAILABLE" = "true" ] || [ "$HAS_NVIDIA_GPU" = "true" ]; then
#         log_info "Installing PyTorch with CUDA 11.8 support..."
#         log_info "Note: CUDA 11.8 binaries are forward compatible with newer drivers"
#         PYTORCH_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
#     else
#         log_info "Installing CPU-only PyTorch..."
#         PYTORCH_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
#     fi
    
#     su - $ACTUAL_USER -c "$CONDA_INIT && conda activate $CONDA_ENV_NAME && $PYTORCH_CMD"
#     log_info "PyTorch installed"
# }

# # ============================================================
# # Install Dependencies
# # ============================================================
# install_dependencies() {
#     log_step "Installing Python dependencies..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
#     CONDA_INIT="source $ACTUAL_HOME/miniconda3/etc/profile.d/conda.sh"
    
#     cd "$PROJECT_DIR"
    
#     # Core dependencies
#     su - $ACTUAL_USER -c "$CONDA_INIT && conda activate $CONDA_ENV_NAME && pip install pyyaml toml tqdm tensorboard"
    
#     # Install from environment.yml if exists
#     if [ -f "environment.yml" ]; then
#         log_info "Processing environment.yml..."
#         su - $ACTUAL_USER -c "$CONDA_INIT && conda activate $CONDA_ENV_NAME && python -c \"
# import yaml
# import subprocess
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
#         # Filter out torch packages (already installed)
#         filtered_deps = [d for d in pip_deps if 'torch' not in d.lower()]
#         for dep in filtered_deps:
#             print(f'  {dep}')
        
#         if filtered_deps:
#             subprocess.run(['pip', 'install'] + filtered_deps, check=True)
    
# except Exception as e:
#     print(f'Warning: Could not fully process environment.yml: {e}', file=sys.stderr)
# \"" || log_warn "Some environment.yml dependencies may have failed"
#     fi
    
#     log_info "Dependencies installed"
# }

# # ============================================================
# # Build Rust Components
# # ============================================================
# build_rust_components() {
#     log_step "Building Rust components..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
    
#     cd "$PROJECT_DIR"
    
#     export CARGO_TARGET_DIR="$PROJECT_DIR/target"
    
#     log_info "Building libriichi library..."
#     su - $ACTUAL_USER -c "cd $PROJECT_DIR && source $ACTUAL_HOME/.cargo/env && CARGO_TARGET_DIR=$PROJECT_DIR/target cargo build --release -p libriichi --lib"
    
#     # Copy library
#     if [ -f "target/release/libriichi.so" ]; then
#         cp target/release/libriichi.so walking/libriichi.so
#         log_info "libriichi.so copied to walking directory"
#     elif [ -f "target/release/liblibriichi.so" ]; then
#         cp target/release/liblibriichi.so walking/libriichi.so
#         log_info "liblibriichi.so copied to walking directory"
#     else
#         log_error "Could not find compiled libriichi library"
#         ls -la target/release/ | grep libriichi || true
#         exit 1
#     fi
# }

# # ============================================================
# # Setup Project Configuration
# # ============================================================
# setup_project_config() {
#     log_step "Setting up project configuration..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     cd "$PROJECT_DIR"
    
#     # Copy config file
#     if [ ! -f "config.toml" ]; then
#         if [ -f "walking/config.example.toml" ]; then
#             su - $ACTUAL_USER -c "cd $PROJECT_DIR && cp walking/config.example.toml config.toml"
#             log_info "Created config.toml from example"
#         elif [ -f "walking/config.toml" ]; then
#             su - $ACTUAL_USER -c "cd $PROJECT_DIR && cp walking/config.toml ."
#             log_info "Copied config.toml"
#         fi
#     fi
    
#     # Create workspace directories
#     log_info "Creating workspace directories..."
#     su - $ACTUAL_USER -c "cd $PROJECT_DIR && mkdir -p workdir/{1v3,buffer,checkpoints,dataset,drain,logs,tensorboard,test_play,train_play}"
    
#     log_info "Project configuration complete"
# }

# # ============================================================
# # Verify Installation
# # ============================================================
# verify_installation() {
#     log_step "Verifying installation..."
    
#     ACTUAL_USER=${SUDO_USER:-$USER}
#     ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
#     CONDA_INIT="source $ACTUAL_HOME/miniconda3/etc/profile.d/conda.sh"
    
#     cd "$PROJECT_DIR"
    
#     su - $ACTUAL_USER -c "$CONDA_INIT && conda activate $CONDA_ENV_NAME && python << 'PYEOF'
# import sys
# print('=' * 60)
# print('WALKING Project Installation Verification')
# print('=' * 60)
# print(f'Python version: {sys.version}')
# print()

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
#     print(f'\nPyTorch CUDA available: {torch.cuda.is_available()}')
#     print(f'PyTorch built with CUDA: {torch.version.cuda}')
#     if torch.cuda.is_available():
#         print(f'CUDA device count: {torch.cuda.device_count()}')
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
#         print('⚠️  CUDA not available - will be available after reboot')

# # libriichi check
# try:
#     import os
#     sys.path.insert(0, os.path.join(os.getcwd(), 'walking'))
#     import libriichi
#     print('\n✓ libriichi imported successfully')
#     print('✓ All core components verified!')
# except Exception as e:
#     print(f'\n✗ libriichi import failed: {e}')
#     print('   This may indicate Python version incompatibility')

# print('=' * 60)
# PYEOF
# "
# }

# # ============================================================
# # Show Usage
# # ============================================================
# show_usage() {
#     echo ""
#     log_info "=== WALKING Project Setup Complete ==="
#     echo ""
    
#     echo "To use the project with Conda:"
#     echo "1. Activate the environment:"
#     echo "   conda activate $CONDA_ENV_NAME"
#     echo ""
#     echo "2. Navigate to project directory:"
#     echo "   cd $PROJECT_DIR"
#     echo ""
#     echo "3. Run the project:"
#     echo "   python walking/train.py        # Start training"
#     echo "   python walking/server.py       # Start server"
#     echo "   python walking/client.py       # Start client"
#     echo ""
#     echo "4. Configuration:"
#     echo "   Edit config.toml to customize settings"
#     echo ""
    
#     if [ "${NEEDS_REBOOT:-false}" = "true" ]; then
#         log_warn "============================================"
#         log_warn "GPU driver installed - REBOOT REQUIRED"
#         log_warn "Please reboot the system:"
#         log_warn "  sudo reboot"
#         log_warn "After reboot, verify GPU with: nvidia-smi"
#         log_warn "============================================"
#     elif [ "$GPU_AVAILABLE" = "true" ]; then
#         log_info "GPU is ready!"
#         nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
#     elif [ "$HAS_NVIDIA_GPU" = "true" ]; then
#         log_warn "GPU hardware detected but driver needs activation (reboot required)"
#     fi
# }

# # ============================================================
# # Main
# # ============================================================
# main() {
#     echo "======================================================================"
#     log_info "WALKING Project Setup - Conda Version (Optimized)"
#     log_info "Mortal Mahjong AI with NVIDIA GPU Support"
#     echo "======================================================================"
#     echo ""
    
#     # Check root
#     check_root
    
#     # Detect GPU
#     detect_gpu
    
#     # Check project directory
#     if [ ! -d "$PROJECT_DIR" ]; then
#         log_error "Project directory not found: $PROJECT_DIR"
#         log_error "Please clone the repository first"
#         exit 1
#     fi
    
#     # Install system dependencies
#     install_system_deps
    
#     # Install NVIDIA driver if needed
#     install_nvidia_driver
    
#     # Install Miniconda
#     install_miniconda
    
#     # Install Rust
#     install_rust
    
#     # Setup Conda environment
#     setup_conda_env
    
#     # Install PyTorch
#     install_pytorch
    
#     # Install other dependencies
#     install_dependencies
    
#     # Setup project config
#     setup_project_config
    
#     # Build Rust components
#     build_rust_components
    
#     # Verify installation
#     verify_installation
    
#     # Show usage
#     show_usage
    
#     log_info "Setup completed successfully!"
# }

# # Error handling
# trap 'log_error "Setup failed at line $LINENO. Check error messages above."' ERR

# # Run main
# main "$@"

#!/bin/bash
#
# WALKING Project Setup - Fixed Version (Conda)
# 修复问题:
# 1. Rust安装卡住 - 使用RUSTUP_USE_CURL=1环境变量
# 2. Conda激活问题 - 正确处理shell初始化
# 3. 避免重复安装已安装的组件
#
# 阿里云 A10 GPU 服务器优化
#
# Usage: bash setup_walking_conda_fixed.sh

set -e

# ============================================================
# Configuration
# ============================================================
PROJECT_NAME="walking3"
CONDA_ENV_NAME="walking3"
PROJECT_DIR="$HOME/dylan/icml2026/WALKING"
FORCE_REINSTALL=${FORCE_REINSTALL:-false}

# Rust 下载超时修复 - 使用curl后端
export RUSTUP_USE_CURL=1

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
# Check GPU availability
# ============================================================
check_gpu() {
    log_step "Checking GPU availability..."
    
    if lspci | grep -i nvidia > /dev/null 2>&1; then
        log_info "NVIDIA GPU hardware detected"
        HAS_GPU_HARDWARE=true
    else
        log_warn "No NVIDIA GPU hardware found"
        HAS_GPU_HARDWARE=false
    fi
    
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        log_info "NVIDIA driver is working"
        GPU_AVAILABLE=true
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        log_warn "NVIDIA driver not working"
        GPU_AVAILABLE=false
    fi
}

# ============================================================
# Install System Dependencies
# ============================================================
install_system_deps() {
    log_step "Checking/Installing system dependencies..."
    
    # 检查是否已安装关键依赖
    local need_install=false
    for pkg in build-essential gcc g++ make cmake git curl wget; do
        if ! dpkg -l | grep -q "^ii  $pkg"; then
            need_install=true
            break
        fi
    done
    
    if [ "$need_install" = true ]; then
        apt-get update
        apt-get install -y \
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
            pkg-config
        log_info "System dependencies installed"
    else
        log_info "System dependencies already installed"
    fi
}

# ============================================================
# Setup Conda (初始化conda到当前shell)
# ============================================================
setup_conda() {
    log_step "Setting up Conda..."
    
    # 检查多个可能的conda位置
    CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "/opt/conda/bin/conda"
        "/root/miniconda3/bin/conda"
    )
    
    CONDA_BIN=""
    for path in "${CONDA_PATHS[@]}"; do
        if [ -f "$path" ]; then
            CONDA_BIN="$path"
            break
        fi
    done
    
    if [ -z "$CONDA_BIN" ]; then
        log_error "Conda not found! Please install Miniconda first."
        log_info "Run: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p \$HOME/miniconda3"
        exit 1
    fi
    
    log_info "Found conda at: $CONDA_BIN"
    
    # 获取conda目录
    CONDA_DIR=$(dirname $(dirname "$CONDA_BIN"))
    
    # 初始化conda到当前shell
    eval "$($CONDA_BIN shell.bash hook)"
    
    # 确保conda命令可用
    if ! command -v conda &> /dev/null; then
        export PATH="$CONDA_DIR/bin:$PATH"
        source "$CONDA_DIR/etc/profile.d/conda.sh"
    fi
    
    log_info "Conda version: $(conda --version)"
}

# ============================================================
# Install Rust (with timeout fix)
# ============================================================
install_rust() {
    log_step "Checking/Installing Rust..."
    
    # 检查是否已安装
    if [ -f "$HOME/.cargo/bin/rustc" ]; then
        source "$HOME/.cargo/env" 2>/dev/null || true
    fi
    
    if command -v rustc &> /dev/null; then
        log_info "Rust already installed: $(rustc --version)"
        log_info "Cargo version: $(cargo --version)"
        return 0
    fi
    
    log_info "Installing Rust..."
    log_info "Using RUSTUP_USE_CURL=1 to prevent timeout issues"
    
    # 设置环境变量防止超时
    export RUSTUP_USE_CURL=1
    
    # 使用 --profile minimal 减少下载量,加快安装
    # 使用 --default-toolchain stable 确保安装稳定版
    # 使用 -y 非交互式安装
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
        -y \
        --profile minimal \
        --default-toolchain stable
    
    # 加载环境
    source "$HOME/.cargo/env"
    
    log_info "Rust installed: $(rustc --version)"
    log_info "Cargo version: $(cargo --version)"
}

# ============================================================
# Setup Conda Environment
# ============================================================
setup_conda_env() {
    log_step "Setting up Python environment..."
    
    # 检查环境是否存在
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        if [ "$FORCE_REINSTALL" = "true" ]; then
            log_info "Removing existing conda environment..."
            conda env remove -n $CONDA_ENV_NAME -y
        else
            log_info "Conda environment '$CONDA_ENV_NAME' already exists"
            conda activate $CONDA_ENV_NAME
            log_info "Python version: $(python --version)"
            return 0
        fi
    fi
    
    # 创建新环境
    log_step "Creating new conda environment with Python 3.12..."
    conda create -n $CONDA_ENV_NAME python=3.12 -y
    
    conda activate $CONDA_ENV_NAME
    log_info "Python version: $(python --version)"
}

# ============================================================
# Install PyTorch
# ============================================================
install_pytorch() {
    log_step "Checking/Installing PyTorch..."
    
    # 检查是否已安装
    if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        log_info "PyTorch already installed: $TORCH_VERSION (CUDA: $CUDA_AVAILABLE)"
        
        # 如果已安装但CUDA不可用，且GPU可用，则重新安装
        if [ "$GPU_AVAILABLE" = "true" ] && [ "$CUDA_AVAILABLE" = "False" ]; then
            log_warn "Reinstalling PyTorch with CUDA support..."
        else
            return 0
        fi
    fi
    
    if [ "$GPU_AVAILABLE" = "true" ]; then
        log_info "Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installing PyTorch CPU-only version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_info "PyTorch installed"
}

# ============================================================
# Install Other Dependencies
# ============================================================
install_dependencies() {
    log_step "Installing other dependencies..."
    
    # 检查基本依赖
    local missing_deps=()
    for dep in pyyaml toml tqdm tensorboard numpy; do
        if ! python -c "import ${dep/pyyaml/yaml}" 2>/dev/null; then
            missing_deps+=($dep)
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_info "Installing: ${missing_deps[*]}"
        pip install ${missing_deps[*]}
    else
        log_info "Basic dependencies already installed"
    fi
    
    # 检查 environment.yml
    cd "$PROJECT_DIR"
    if [ -f "environment.yml" ]; then
        log_info "Parsing environment.yml for additional dependencies..."
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
        with open('.pip_deps_temp.txt', 'w') as f:
            for dep in pip_deps:
                if 'torch' not in dep.lower():
                    f.write(dep + '\n')
        print(f'Found {len(pip_deps)} pip dependencies')
except Exception as e:
    print(f'Could not parse environment.yml: {e}')
" 
        if [ -f .pip_deps_temp.txt ] && [ -s .pip_deps_temp.txt ]; then
            pip install -r .pip_deps_temp.txt || log_warn "Some environment.yml deps failed"
            rm -f .pip_deps_temp.txt
        fi
    fi
    
    log_info "Dependencies installed"
}

# ============================================================
# Setup Project Configuration
# ============================================================
setup_project_config() {
    log_step "Setting up project configuration..."
    
    cd "$PROJECT_DIR"
    
    # 配置文件
    if [ ! -f "config.toml" ]; then
        if [ -f "walking/config.example.toml" ]; then
            cp walking/config.example.toml config.toml
            log_info "Created config.toml from example"
        elif [ -f "walking/config.toml" ]; then
            cp walking/config.toml .
            log_info "Copied config.toml"
        fi
    else
        log_info "config.toml already exists"
    fi
    
    # 创建工作目录
    mkdir -p workdir/{1v3,buffer,checkpoints,dataset,drain,logs,tensorboard,test_play,train_play}
    log_info "Workspace directories ready"
}

# ============================================================
# Build Rust Components
# ============================================================
build_rust_components() {
    log_step "Building Rust components..."
    
    cd "$PROJECT_DIR"
    source "$HOME/.cargo/env" 2>/dev/null || true
    
    # 检查是否已编译
    if [ -f "walking/libriichi.so" ]; then
        log_info "libriichi.so already exists"
        
        # 验证是否可以加载
        if python -c "import sys; sys.path.insert(0, 'walking'); import libriichi" 2>/dev/null; then
            log_info "libriichi.so is valid"
            if [ "$FORCE_REINSTALL" != "true" ]; then
                return 0
            fi
        else
            log_warn "libriichi.so exists but cannot be loaded, rebuilding..."
        fi
    fi
    
    export CARGO_TARGET_DIR=target
    
    log_info "Building libriichi library..."
    if cargo build --release -p libriichi --lib; then
        log_info "Rust build completed"
        
        # 复制库文件
        if [ -f "target/release/libriichi.so" ]; then
            cp target/release/libriichi.so walking/libriichi.so
            log_info "Copied libriichi.so"
        elif [ -f "target/release/liblibriichi.so" ]; then
            cp target/release/liblibriichi.so walking/libriichi.so
            log_info "Copied liblibriichi.so as libriichi.so"
        else
            log_error "Could not find compiled library"
            ls -la target/release/*.so 2>/dev/null || log_error "No .so files found"
            exit 1
        fi
    else
        log_error "Rust build failed"
        exit 1
    fi
}

# ============================================================
# Verify Installation
# ============================================================
verify_installation() {
    log_step "Verifying installation..."
    
    cd "$PROJECT_DIR"
    
    python << 'PYEOF'
import sys
print("=" * 60)
print("WALKING Project Installation Verification")
print("=" * 60)
print(f"Python: {sys.version}")

# Core dependencies
deps = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('yaml', 'PyYAML'),
    ('toml', 'TOML'),
    ('tqdm', 'tqdm'),
]

all_ok = True
for module, name in deps:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f"✓ {name}: {version}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        all_ok = False

# PyTorch CUDA
try:
    import torch
    print(f"\n--- PyTorch GPU Status ---")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
except Exception as e:
    print(f"PyTorch error: {e}")

# libriichi
print(f"\n--- libriichi Status ---")
try:
    import os
    sys.path.insert(0, os.path.join(os.getcwd(), 'walking'))
    import libriichi
    print("✓ libriichi imported successfully")
except Exception as e:
    print(f"✗ libriichi: {e}")
    all_ok = False

print("=" * 60)
if all_ok:
    print("✓ All checks passed!")
else:
    print("⚠ Some checks failed")
print("=" * 60)
PYEOF
}

# ============================================================
# Create Activation Script
# ============================================================
create_activation_script() {
    log_step "Creating activation script..."
    
    cat > "$PROJECT_DIR/activate.sh" << 'EOF'
#!/bin/bash
# WALKING Project Activation Script
# Usage: source activate.sh

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 初始化conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 激活环境
conda activate walking3

# 加载Rust环境
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

# 切换到项目目录
cd "$SCRIPT_DIR"

echo "WALKING environment activated!"
echo "Python: $(python --version)"
echo "Project: $SCRIPT_DIR"
EOF
    
    chmod +x "$PROJECT_DIR/activate.sh"
    log_info "Created activate.sh"
}

# ============================================================
# Show Usage
# ============================================================
show_usage() {
    echo ""
    log_info "=========================================="
    log_info "WALKING Project Setup Complete!"
    log_info "=========================================="
    echo ""
    echo "To activate the environment, run:"
    echo ""
    echo "  source $PROJECT_DIR/activate.sh"
    echo ""
    echo "Or manually:"
    echo ""
    echo "  # 初始化conda (每次新终端需要)"
    echo "  source ~/miniconda3/etc/profile.d/conda.sh"
    echo "  conda activate walking3"
    echo "  cd $PROJECT_DIR"
    echo ""
    echo "Then run:"
    echo "  python walking/train.py        # 训练"
    echo "  python walking/server.py       # 服务器"
    echo ""
    
    if [ "$GPU_AVAILABLE" = "true" ]; then
        log_info "GPU: NVIDIA A10 detected and working"
    else
        log_warn "GPU not available"
    fi
}

# ============================================================
# Main
# ============================================================
main() {
    echo "======================================================================"
    log_info "WALKING Project Setup (Fixed Version)"
    log_info "Optimized for Aliyun A10 GPU Server"
    echo "======================================================================"
    echo ""
    
    # GPU检查
    check_gpu
    
    # 项目目录检查
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # 系统依赖
    install_system_deps
    
    # Conda设置
    setup_conda
    
    # Rust安装 (修复超时问题)
    install_rust
    source "$HOME/.cargo/env" 2>/dev/null || true
    
    # Conda环境
    setup_conda_env
    
    # PyTorch
    install_pytorch
    
    # 其他依赖
    install_dependencies
    
    # 项目配置
    setup_project_config
    
    # Rust编译
    build_rust_components
    
    # 验证
    verify_installation
    
    # 创建激活脚本
    create_activation_script
    
    # 使用说明
    show_usage
    
    log_info "Setup completed successfully!"
}

# 错误处理
trap 'log_error "Failed at line $LINENO"' ERR

main "$@"