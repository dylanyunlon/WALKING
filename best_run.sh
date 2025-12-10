#!/bin/bash
#
# WALKING/Mortal Mahjong AI - Complete Training & Evaluation Script
# 
# 这个脚本提供完整的训练、测试、对战流程
# 
# Usage:
#   bash best_run.sh setup          # 初始化环境和数据集
#   bash best_run.sh train          # 开始离线训练
#   bash best_run.sh eval           # 评估模型 (1v3 对战)
#   bash best_run.sh eval_fair      # 使用相同模型对战 (公平测试)
#   bash best_run.sh all            # 完整流程

set -e

# =============================================================================
# 配置变量
# =============================================================================
PROJECT_DIR="/root/dylan/icml2026/WALKING"
CONDA_ENV="walking3"
CONFIG_FILE="${PROJECT_DIR}/config.toml"
WALKING_DIR="${PROJECT_DIR}/walking"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }
log_cmd()   { echo -e "${CYAN}[CMD]${NC} $1"; }

# =============================================================================
# 环境激活
# =============================================================================
activate_env() {
    log_step "激活 Conda 环境: ${CONDA_ENV}"
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
    cd ${PROJECT_DIR}
    log_info "当前目录: $(pwd)"
    log_info "Python: $(which python)"
}

# =============================================================================
# 检查和准备
# =============================================================================
check_prerequisites() {
    log_step "检查前置条件..."
    
    # 检查 libriichi.so
    if [ ! -f "${WALKING_DIR}/libriichi.so" ]; then
        log_error "libriichi.so 不存在! 请先运行 setup_walking_project.sh"
        exit 1
    fi
    log_info "✓ libriichi.so 存在"
    
    # 检查 CUDA
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    log_info "✓ CUDA 可用"
    
    # 检查 GPU 数量
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    log_info "✓ GPU 数量: ${GPU_COUNT}"
    
    # 检查数据集
    DATASET_COUNT=$(find ${PROJECT_DIR}/workdir/dataset -name "*.json.gz" 2>/dev/null | wc -l)
    if [ "$DATASET_COUNT" -eq 0 ]; then
        log_warn "数据集为空! 需要先准备 mjai 格式的对局记录"
        log_warn "数据集目录: ${PROJECT_DIR}/workdir/dataset/"
        log_warn "可从 Kaggle 获取: https://www.kaggle.com/datasets/shokanekolouis/tenhou-to-mjai"
    else
        log_info "✓ 数据集文件数: ${DATASET_COUNT}"
    fi
}

# =============================================================================
# 数据集准备
# =============================================================================
setup_dataset() {
    log_step "准备数据集..."
    
    mkdir -p ${PROJECT_DIR}/workdir/dataset/{train,val}
    
    # 检查是否有 db 文件
    if [ -f "${PROJECT_DIR}/db/2023.db" ]; then
        log_info "发现 2023.db，正在转换为 mjai 格式..."
        cd ${PROJECT_DIR}
        
        # 使用项目自带的转换脚本（如果存在）
        if [ -f "convert_tenhou_to_mjai.py" ]; then
            log_cmd "python convert_tenhou_to_mjai.py"
            python convert_tenhou_to_mjai.py || log_warn "转换脚本执行失败，可能需要手动处理"
        fi
    fi
    
    # 如果有 scraw2023.zip，解压它
    if [ -f "${PROJECT_DIR}/scraw2023.zip" ]; then
        log_info "发现 scraw2023.zip，正在解压..."
        unzip -q -o ${PROJECT_DIR}/scraw2023.zip -d ${PROJECT_DIR}/workdir/dataset/ || true
    fi
    
    # 重新计数
    DATASET_COUNT=$(find ${PROJECT_DIR}/workdir/dataset -name "*.json.gz" 2>/dev/null | wc -l)
    log_info "数据集文件总数: ${DATASET_COUNT}"
    
    if [ "$DATASET_COUNT" -eq 0 ]; then
        log_warn "=========================================="
        log_warn "数据集为空！请按以下步骤准备数据："
        log_warn ""
        log_warn "方法1: 从 Kaggle 下载 Tenhou 日志"
        log_warn "  kaggle datasets download -d shokanekolouis/tenhou-to-mjai"
        log_warn "  unzip tenhou-to-mjai.zip -d ${PROJECT_DIR}/workdir/dataset/"
        log_warn ""
        log_warn "方法2: 转换现有 Tenhou XML 日志"
        log_warn "  python convert_tenhou_to_mjai.py"
        log_warn "=========================================="
    fi
}

# =============================================================================
# 模型初始化
# =============================================================================
init_models() {
    log_step "初始化模型..."
    
    CHECKPOINTS_DIR="${PROJECT_DIR}/workdir/checkpoints"
    mkdir -p ${CHECKPOINTS_DIR}
    
    # 检查是否有预训练模型
    if [ -f "${CHECKPOINTS_DIR}/walking.pth" ]; then
        log_info "✓ 已存在 walking.pth ($(ls -lh ${CHECKPOINTS_DIR}/walking.pth | awk '{print $5}'))"
    else
        log_warn "walking.pth 不存在，训练将从随机初始化开始"
    fi
    
    if [ -f "${CHECKPOINTS_DIR}/baseline.pth" ]; then
        # 检查 baseline 是否是真正训练过的模型
        BASELINE_INFO=$(python -c "
import torch
state = torch.load('${CHECKPOINTS_DIR}/baseline.pth', weights_only=True, map_location='cpu')
# 检查第一层的标准差
first_layer = list(state['walking'].values())[0]
std = first_layer.float().std().item()
if std < 0.001:
    print('UNTRAINED')
else:
    print('TRAINED')
" 2>/dev/null || echo "UNKNOWN")
        
        if [ "$BASELINE_INFO" = "UNTRAINED" ]; then
            log_warn "⚠ baseline.pth 看起来是未训练的初始模型!"
            log_warn "  这会导致 1v3 测试结果不可靠"
            log_warn "  建议获取官方 Mortal 模型作为 baseline"
        else
            log_info "✓ 已存在 baseline.pth"
        fi
    fi
    
    # 复制用于 1v3 测试的模型
    if [ -f "${CHECKPOINTS_DIR}/walking.pth" ]; then
        cp ${CHECKPOINTS_DIR}/walking.pth ${CHECKPOINTS_DIR}/challenger.pth
        log_info "✓ 复制 walking.pth -> challenger.pth"
    fi
    
    if [ -f "${CHECKPOINTS_DIR}/baseline.pth" ]; then
        cp ${CHECKPOINTS_DIR}/baseline.pth ${CHECKPOINTS_DIR}/champion.pth
        log_info "✓ 复制 baseline.pth -> champion.pth"
    fi
}

# =============================================================================
# 离线训练 (Offline RL with CQL)
# =============================================================================
run_training() {
    log_step "开始离线训练..."
    
    cd ${WALKING_DIR}
    
    # 检查数据集
    DATASET_COUNT=$(find ${PROJECT_DIR}/workdir/dataset -name "*.json.gz" 2>/dev/null | wc -l)
    if [ "$DATASET_COUNT" -eq 0 ]; then
        log_error "数据集为空! 请先运行: bash best_run.sh setup"
        exit 1
    fi
    
    log_info "配置信息:"
    log_info "  - batch_size: 512"
    log_info "  - conv_channels: 192"
    log_info "  - num_blocks: 40"
    log_info "  - device: cuda:0"
    
    log_cmd "python train.py"
    
    # 使用 nohup 运行训练，输出到日志
    LOG_FILE="${PROJECT_DIR}/workdir/logs/train_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p $(dirname ${LOG_FILE})
    
    echo ""
    log_info "训练日志: ${LOG_FILE}"
    log_info "TensorBoard: tensorboard --logdir=${PROJECT_DIR}/workdir/tensorboard"
    echo ""
    
    # 前台运行（可以看到输出）
    python train.py 2>&1 | tee ${LOG_FILE}
}

# =============================================================================
# 后台训练
# =============================================================================
run_training_background() {
    log_step "启动后台训练..."
    
    cd ${WALKING_DIR}
    
    LOG_FILE="${PROJECT_DIR}/workdir/logs/train_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p $(dirname ${LOG_FILE})
    
    nohup python train.py > ${LOG_FILE} 2>&1 &
    TRAIN_PID=$!
    
    log_info "训练进程 PID: ${TRAIN_PID}"
    log_info "训练日志: ${LOG_FILE}"
    log_info "查看日志: tail -f ${LOG_FILE}"
    log_info "停止训练: kill ${TRAIN_PID}"
    
    echo ${TRAIN_PID} > ${PROJECT_DIR}/workdir/train.pid
}

# =============================================================================
# 模型评估 (1v3 对战)
# =============================================================================
run_evaluation() {
    log_step "开始 1v3 模型评估..."
    
    cd ${WALKING_DIR}
    
    # 更新 challenger 和 champion 模型
    CHECKPOINTS_DIR="${PROJECT_DIR}/workdir/checkpoints"
    
    if [ -f "${CHECKPOINTS_DIR}/walking.pth" ]; then
        cp ${CHECKPOINTS_DIR}/walking.pth ${CHECKPOINTS_DIR}/challenger.pth
    fi
    
    log_info "对战配置:"
    log_info "  - Challenger: walking.pth (你的模型)"
    log_info "  - Champion: baseline.pth (对手)"
    log_info "  - 每轮游戏数: 2000"
    log_info "  - 总轮数: 500"
    
    log_warn "注意: 如果 baseline 是未训练的模型，结果将不可靠!"
    echo ""
    
    log_cmd "python one_vs_three.py"
    python one_vs_three.py
}

# =============================================================================
# 公平评估 (自我对战)
# =============================================================================
run_fair_evaluation() {
    log_step "开始公平评估 (自我对战)..."
    
    cd ${WALKING_DIR}
    
    CHECKPOINTS_DIR="${PROJECT_DIR}/workdir/checkpoints"
    
    # 使用同一个模型作为 challenger 和 champion
    if [ -f "${CHECKPOINTS_DIR}/walking.pth" ]; then
        cp ${CHECKPOINTS_DIR}/walking.pth ${CHECKPOINTS_DIR}/challenger.pth
        cp ${CHECKPOINTS_DIR}/walking.pth ${CHECKPOINTS_DIR}/champion.pth
        log_info "使用 walking.pth 进行自我对战"
    elif [ -f "${CHECKPOINTS_DIR}/best.pth" ]; then
        cp ${CHECKPOINTS_DIR}/best.pth ${CHECKPOINTS_DIR}/challenger.pth
        cp ${CHECKPOINTS_DIR}/best.pth ${CHECKPOINTS_DIR}/champion.pth
        log_info "使用 best.pth 进行自我对战"
    else
        log_error "没有可用的模型!"
        exit 1
    fi
    
    log_info "自我对战配置:"
    log_info "  - 使用相同模型进行对战"
    log_info "  - 预期结果: 平均排名约 2.5, 得分约 0pt"
    echo ""
    
    # 临时修改迭代次数
    log_cmd "python one_vs_three.py (快速测试: 10 轮)"
    
    python -c "
import prelude
import numpy as np
import torch
import secrets
from model import Brain, DQN
from engine import WalkingEngine
from libriichi.arena import OneVsThree
from config import config

cfg = config['1v3']
# 只跑 10 轮用于快速验证
iters = 10
games_per_iter = cfg['games_per_iter']
seeds_per_iter = games_per_iter // 4

key = secrets.randbits(64)

# 加载模型
state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location='cpu')
chal_cfg = state['config']
version = chal_cfg['control'].get('version', 1)
conv_channels = chal_cfg['resnet']['conv_channels']
num_blocks = chal_cfg['resnet']['num_blocks']

walking = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
dqn = DQN(version=version).eval()
walking.load_state_dict(state['walking'])
dqn.load_state_dict(state['current_dqn'])

# 同一个模型用于两方
engine = WalkingEngine(
    walking, dqn,
    is_oracle=False, version=version,
    device=torch.device(cfg['challenger']['device']),
    enable_amp=cfg['challenger']['enable_amp'],
    enable_rule_based_agari_guard=cfg['challenger']['enable_rule_based_agari_guard'],
    name='self_play',
)

seed_start = 10000
all_rankings = []

print('自我对战测试 (预期: 平均排名约 2.5)')
print('-' * 50)

for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
    env = OneVsThree(disable_progress_bar=False, log_dir=None)
    rankings = env.py_vs_py(
        challenger=engine,
        champion=engine,  # 使用相同的 engine
        seed_start=(seed, key),
        seed_count=seeds_per_iter,
    )
    rankings = np.array(rankings)
    all_rankings.append(rankings)
    avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
    avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
    print(f'#{i}: rankings={rankings}, avg_rank={avg_rank:.4f}, avg_pt={avg_pt:.2f}pt')

# 汇总
total = np.sum(all_rankings, axis=0)
final_avg_rank = total @ np.arange(1, 5) / total.sum()
final_avg_pt = total @ np.array([90, 45, 0, -135]) / total.sum()
print('=' * 50)
print(f'总计: rankings={total}')
print(f'平均排名: {final_avg_rank:.4f} (预期约 2.5)')
print(f'平均得分: {final_avg_pt:.2f}pt (预期约 0)')
print('')
if 2.4 < final_avg_rank < 2.6:
    print('✓ 自我对战结果正常!')
else:
    print('⚠ 自我对战结果异常，可能存在问题')
"
}

# =============================================================================
# 快速评估 (减少迭代次数)
# =============================================================================
run_quick_eval() {
    log_step "快速评估 (10 轮)..."
    
    cd ${WALKING_DIR}
    
    python -c "
import prelude
import numpy as np
import torch
import secrets
from model import Brain, DQN
from engine import WalkingEngine
from libriichi.arena import OneVsThree
from config import config

cfg = config['1v3']
iters = 10  # 只跑 10 轮
games_per_iter = cfg['games_per_iter']
seeds_per_iter = games_per_iter // 4

key = secrets.randbits(64)

# 加载 challenger
state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location='cpu')
chal_cfg = state['config']
version = chal_cfg['control'].get('version', 1)
conv_channels = chal_cfg['resnet']['conv_channels']
num_blocks = chal_cfg['resnet']['num_blocks']

walking_chal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
dqn_chal = DQN(version=version).eval()
walking_chal.load_state_dict(state['walking'])
dqn_chal.load_state_dict(state['current_dqn'])

engine_chal = WalkingEngine(
    walking_chal, dqn_chal,
    is_oracle=False, version=version,
    device=torch.device(cfg['challenger']['device']),
    enable_amp=cfg['challenger']['enable_amp'],
    enable_rule_based_agari_guard=cfg['challenger']['enable_rule_based_agari_guard'],
    name=cfg['challenger']['name'],
)

# 加载 champion
state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location='cpu')
cham_cfg = state['config']
version = cham_cfg['control'].get('version', 1)
conv_channels = cham_cfg['resnet']['conv_channels']
num_blocks = cham_cfg['resnet']['num_blocks']

walking_cham = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
dqn_cham = DQN(version=version).eval()
walking_cham.load_state_dict(state['walking'])
dqn_cham.load_state_dict(state['current_dqn'])

engine_cham = WalkingEngine(
    walking_cham, dqn_cham,
    is_oracle=False, version=version,
    device=torch.device(cfg['champion']['device']),
    enable_amp=cfg['champion']['enable_amp'],
    enable_rule_based_agari_guard=cfg['champion']['enable_rule_based_agari_guard'],
    name=cfg['champion']['name'],
)

seed_start = 10000
all_rankings = []

print('Challenger vs Champion 快速评估')
print('-' * 50)

for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
    env = OneVsThree(disable_progress_bar=False, log_dir=None)
    rankings = env.py_vs_py(
        challenger=engine_chal,
        champion=engine_cham,
        seed_start=(seed, key),
        seed_count=seeds_per_iter,
    )
    rankings = np.array(rankings)
    all_rankings.append(rankings)
    avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
    avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
    print(f'#{i}: {rankings}, rank={avg_rank:.4f}, pt={avg_pt:.2f}')

total = np.sum(all_rankings, axis=0)
final_avg_rank = total @ np.arange(1, 5) / total.sum()
final_avg_pt = total @ np.array([90, 45, 0, -135]) / total.sum()
print('=' * 50)
print(f'Challenger 排名分布: {total}')
print(f'Challenger 平均排名: {final_avg_rank:.4f}')
print(f'Challenger 平均得分: {final_avg_pt:.2f}pt')
"
}

# =============================================================================
# 状态查看
# =============================================================================
show_status() {
    log_step "项目状态"
    echo ""
    
    # 检查训练进程
    if [ -f "${PROJECT_DIR}/workdir/train.pid" ]; then
        TRAIN_PID=$(cat ${PROJECT_DIR}/workdir/train.pid)
        if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
            log_info "✓ 训练进程运行中 (PID: ${TRAIN_PID})"
        else
            log_warn "训练进程已停止"
        fi
    fi
    
    # 模型文件
    echo ""
    log_info "模型文件:"
    ls -lh ${PROJECT_DIR}/workdir/checkpoints/*.pth 2>/dev/null || log_warn "  没有模型文件"
    
    # 数据集
    echo ""
    DATASET_COUNT=$(find ${PROJECT_DIR}/workdir/dataset -name "*.json.gz" 2>/dev/null | wc -l)
    log_info "数据集文件数: ${DATASET_COUNT}"
    
    # 日志
    echo ""
    log_info "最近的日志:"
    ls -lt ${PROJECT_DIR}/workdir/logs/*.log 2>/dev/null | head -3 || log_warn "  没有日志文件"
    
    # GPU 状态
    echo ""
    log_info "GPU 状态:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
}

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    echo ""
    echo "WALKING/Mortal Mahjong AI - 训练和评估脚本"
    echo ""
    echo "用法: bash best_run.sh <command>"
    echo ""
    echo "命令:"
    echo "  setup           初始化环境，准备数据集"
    echo "  train           开始离线训练 (前台运行)"
    echo "  train-bg        开始离线训练 (后台运行)"
    echo "  eval            完整评估 (1v3 对战, 500轮)"
    echo "  quick-eval      快速评估 (10轮)"
    echo "  fair-eval       公平评估 (自我对战，验证代码正确性)"
    echo "  status          查看项目状态"
    echo "  all             完整流程 (setup -> train)"
    echo "  help            显示此帮助"
    echo ""
    echo "示例:"
    echo "  bash best_run.sh setup      # 首次运行"
    echo "  bash best_run.sh train      # 开始训练"
    echo "  bash best_run.sh quick-eval # 快速测试"
    echo ""
    echo "重要提示:"
    echo "  1. baseline.pth 应该是训练好的强模型，否则评估结果无意义"
    echo "  2. 官方 Mortal 模型需要从 Discord 获取"
    echo "  3. 使用 fair-eval 可以验证代码是否正确工作"
    echo ""
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    echo "======================================================================"
    echo "WALKING/Mortal Mahjong AI - Training & Evaluation"
    echo "======================================================================"
    echo ""
    
    case "${1:-help}" in
        setup)
            activate_env
            check_prerequisites
            setup_dataset
            init_models
            log_info "设置完成! 下一步: bash best_run.sh train"
            ;;
        train)
            activate_env
            check_prerequisites
            run_training
            ;;
        train-bg)
            activate_env
            check_prerequisites
            run_training_background
            ;;
        eval)
            activate_env
            check_prerequisites
            run_evaluation
            ;;
        quick-eval)
            activate_env
            check_prerequisites
            run_quick_eval
            ;;
        fair-eval)
            activate_env
            check_prerequisites
            run_fair_evaluation
            ;;
        status)
            activate_env
            show_status
            ;;
        all)
            activate_env
            check_prerequisites
            setup_dataset
            init_models
            run_training
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"