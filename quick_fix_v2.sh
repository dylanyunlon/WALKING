#!/bin/bash
#
# WALKING Quick Fix v4 - 使用中国镜像源
# 清华大学 + 中科大镜像加速
#
# Usage: bash quick_fix_v4.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

PROJECT_DIR="$HOME/dylan/icml2026/WALKING"
CONDA_ENV_NAME="walking3"

# ============================================================
# 设置中国镜像源
# ============================================================
export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"

echo "======================================================================"
log_info "WALKING Quick Fix v4 - 中国镜像加速版"
log_info "Rustup镜像: 清华大学 (mirrors.tuna.tsinghua.edu.cn)"
echo "======================================================================"

# ============================================================
# Step 1: Conda
# ============================================================
log_step "Step 1: Setting up Conda..."

CONDA_DIR=""
for path in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [ -f "$path/etc/profile.d/conda.sh" ]; then
        CONDA_DIR="$path"
        break
    fi
done

[ -z "$CONDA_DIR" ] && { log_error "Conda not found!"; exit 1; }

source "$CONDA_DIR/etc/profile.d/conda.sh"
log_info "Conda: $(conda --version)"

# 接受TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 激活环境
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log_info "Environment exists"
else
    log_info "Creating environment..."
    conda create -n $CONDA_ENV_NAME python=3.12 -y
fi

conda activate $CONDA_ENV_NAME
log_info "Python: $(python --version)"

# ============================================================
# Step 2: 使用清华镜像安装Rust
# ============================================================
log_step "Step 2: Installing Rust (清华镜像)..."

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

if command -v rustc &> /dev/null; then
    log_info "Rust already installed: $(rustc --version)"
else
    log_info "使用清华镜像下载 rustup-init..."
    
    # 清华镜像的rustup-init地址
    RUSTUP_URL="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup/archive/1.28.1/x86_64-unknown-linux-gnu/rustup-init"
    RUSTUP_INIT="/tmp/rustup-init"
    
    rm -f "$RUSTUP_INIT"
    
    # 下载
    log_info "Downloading from: $RUSTUP_URL"
    if wget -q --show-progress -O "$RUSTUP_INIT" "$RUSTUP_URL"; then
        chmod +x "$RUSTUP_INIT"
        
        log_info "Installing Rust..."
        # 确保环境变量已设置
        export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
        export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"
        
        "$RUSTUP_INIT" -y --profile minimal --default-toolchain stable
        
        rm -f "$RUSTUP_INIT"
        source "$HOME/.cargo/env"
        
        # 将镜像配置写入env文件
        echo 'export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"' >> "$HOME/.cargo/env"
        echo 'export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"' >> "$HOME/.cargo/env"
        
        log_info "Rust: $(rustc --version)"
    else
        log_error "Download failed!"
        exit 1
    fi
fi

# 配置cargo使用国内镜像
log_info "Configuring cargo to use China mirrors..."
mkdir -p "$HOME/.cargo"
cat > "$HOME/.cargo/config" << 'EOF'
[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"
replace-with = 'tuna'

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

[source.ustc]
registry = "git://mirrors.ustc.edu.cn/crates.io-index"

[source.sjtu]
registry = "https://mirrors.sjtug.sjtu.edu.cn/git/crates.io-index"

[net]
git-fetch-with-cli = true
EOF
log_info "Cargo配置完成 (使用清华镜像)"

# ============================================================
# Step 3: PyTorch
# ============================================================
log_step "Step 3: Installing PyTorch..."

if python -c "import torch" 2>/dev/null; then
    log_info "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
else
    if nvidia-smi &>/dev/null; then
        log_info "Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# ============================================================
# Step 4: 依赖
# ============================================================
log_step "Step 4: Installing dependencies..."
pip install pyyaml toml tqdm tensorboard numpy -q

# ============================================================
# Step 5: 编译Rust (使用国内镜像)
# ============================================================
log_step "Step 5: Building Rust components (使用清华镜像)..."

cd "$PROJECT_DIR"
source "$HOME/.cargo/env" 2>/dev/null || true

if [ -f "walking/libriichi.so" ]; then
    if python -c "import sys; sys.path.insert(0,'walking'); import libriichi" 2>/dev/null; then
        log_info "libriichi.so OK"
    else
        rm -f walking/libriichi.so
    fi
fi

if [ ! -f "walking/libriichi.so" ]; then
    export CARGO_TARGET_DIR=target
    log_info "Building libriichi (crates从清华镜像下载)..."
    
    cargo build --release -p libriichi --lib
    
    for lib in target/release/libriichi.so target/release/liblibriichi.so; do
        [ -f "$lib" ] && { cp "$lib" walking/libriichi.so; log_info "Created libriichi.so"; break; }
    done
fi

# ============================================================
# Step 6: 配置
# ============================================================
log_step "Step 6: Setup workspace..."
mkdir -p workdir/{1v3,buffer,checkpoints,dataset,drain,logs,tensorboard,test_play,train_play}
[ ! -f "config.toml" ] && [ -f "walking/config.example.toml" ] && cp walking/config.example.toml config.toml

# ============================================================
# Step 7: 验证
# ============================================================
log_step "Step 7: Verification..."

python << 'PYEOF'
import sys
print("=" * 50)
print(f"Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    sys.path.insert(0, 'walking')
    import libriichi
    print("✓ libriichi: OK")
except Exception as e:
    print(f"✗ libriichi: {e}")

for m in ['numpy','yaml','toml','tqdm']:
    try:
        __import__(m)
        print(f"✓ {m}: OK")
    except:
        print(f"✗ {m}: MISSING")
print("=" * 50)
PYEOF

# ============================================================
# Step 8: 更新bashrc
# ============================================================
if ! grep -q "RUSTUP_DIST_SERVER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << EOF

# WALKING Environment + 中国镜像
source $CONDA_DIR/etc/profile.d/conda.sh
[ -f "\$HOME/.cargo/env" ] && source "\$HOME/.cargo/env"
export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
export RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"
EOF
    log_info "Updated ~/.bashrc"
fi

echo ""
log_info "=========================================="
log_info "Setup Complete!"
log_info "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate walking3"
echo "  cd $PROJECT_DIR"
echo "  python walking/train.py"
echo ""