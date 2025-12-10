#!/bin/bash
#
# WALKING vs Mortal 对战脚本 (进化版)
# Generation: 1
# 
# 改进说明:
#   - 集成 Jeff Dean 调试信息写入
#   - 输出结构化 JSON 便于 LLM 解析
#   - 支持进化系统的自动改进
#   - [v1] 自动检测 CUDA 设备，支持单GPU环境
#   - [v1] 改进错误传播机制
#
# 用法:
#   bash walking_vs_mortal.sh          # 快速测试 (10轮)
#   bash walking_vs_mortal.sh full     # 完整测试 (100轮)
#   bash walking_vs_mortal.sh check    # 仅检查模型

set -e

# =============================================================================
# 配置变量
# =============================================================================
PROJECT_DIR="/root/dylan/icml2026/WALKING"
AKAGI_DIR="/root/dylan/icml2026/Akagi"
CONDA_ENV="walking3"
WALKING_DIR="${PROJECT_DIR}/walking"
EVOLUTION_DIR="${PROJECT_DIR}/evolution"

# Mortal 模型路径
MORTAL_PTH="${AKAGI_DIR}/mjai_bot/mortal/mortal.pth"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# =============================================================================
# 全局状态变量 (用于错误追踪)
# =============================================================================
declare -g SCRIPT_EXIT_CODE=0
declare -g SCRIPT_ERROR_MSG=""

# =============================================================================
# 进化系统集成 - 调试信息写入
# =============================================================================
init_debug() {
    mkdir -p ${EVOLUTION_DIR}
    export EVOLUTION_DIR=${EVOLUTION_DIR}
}

write_debug_metric() {
    # 写入指标到调试文件
    local NAME=$1
    local VALUE=$2
    python3 ${EVOLUTION_DIR}/debug_writer.py metric -n "${NAME}" -v "${VALUE}" 2>/dev/null || true
}

write_debug_status() {
    local STATUS=$1
    python3 ${EVOLUTION_DIR}/debug_writer.py status -v "${STATUS}" 2>/dev/null || true
}

write_debug_error() {
    local ERROR=$1
    python3 ${EVOLUTION_DIR}/debug_writer.py error -m "${ERROR}" 2>/dev/null || true
}

# =============================================================================
# CUDA 设备检测
# =============================================================================
detect_cuda_devices() {
    log_step "检测 CUDA 设备..."
    
    python3 << 'PYTHON_CUDA_CHECK'
import torch
import json

result = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": 0,
    "devices": [],
    "recommended": {
        "challenger": "cpu",
        "champion": "cpu"
    }
}

if torch.cuda.is_available():
    result["device_count"] = torch.cuda.device_count()
    
    for i in range(result["device_count"]):
        props = torch.cuda.get_device_properties(i)
        result["devices"].append({
            "index": i,
            "name": props.name,
            "memory_gb": round(props.total_memory / (1024**3), 2)
        })
    
    # 推荐设备分配策略
    if result["device_count"] >= 2:
        # 双GPU: 各用一个
        result["recommended"]["challenger"] = "cuda:0"
        result["recommended"]["champion"] = "cuda:1"
    elif result["device_count"] == 1:
        # 单GPU: 都用 cuda:0
        result["recommended"]["challenger"] = "cuda:0"
        result["recommended"]["champion"] = "cuda:0"
    
    print(f"CUDA 可用: {result['device_count']} 个设备")
    for dev in result["devices"]:
        print(f"  [{dev['index']}] {dev['name']} ({dev['memory_gb']} GB)")
    print(f"推荐配置: Challenger={result['recommended']['challenger']}, Champion={result['recommended']['champion']}")
else:
    print("CUDA 不可用，将使用 CPU")

# 输出 JSON 供 bash 解析
print("[CUDA_CONFIG_JSON]")
print(json.dumps(result))
print("[/CUDA_CONFIG_JSON]")
PYTHON_CUDA_CHECK
}

# =============================================================================
# 环境激活
# =============================================================================
activate_env() {
    log_step "激活 Conda 环境: ${CONDA_ENV}"
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
    cd ${PROJECT_DIR}
}

# =============================================================================
# 检查模型
# =============================================================================
check_models() {
    log_step "检查模型..."
    
    python3 << 'PYTHON_CHECK'
import torch
import json

print("=" * 70)
print("模型对比")
print("=" * 70)

result = {"walking": {}, "mortal": {}, "compatible": False}

# WALKING 模型
walking_path = "/root/dylan/icml2026/WALKING/workdir/checkpoints/walking.pth"
try:
    walking_state = torch.load(walking_path, weights_only=True, map_location='cpu')
    w_cfg = walking_state['config']
    
    result["walking"] = {
        "path": walking_path,
        "version": w_cfg['control'].get('version', 1),
        "conv_channels": w_cfg['resnet']['conv_channels'],
        "num_blocks": w_cfg['resnet']['num_blocks']
    }
    
    print("\n[WALKING 模型]")
    print(f"  路径: {walking_path}")
    print(f"  Version: {result['walking']['version']}")
    print(f"  Conv Channels: {result['walking']['conv_channels']}")
    print(f"  Num Blocks: {result['walking']['num_blocks']}")
except Exception as e:
    print(f"\n[ERROR] 加载 WALKING 模型失败: {e}")
    result["walking"]["error"] = str(e)

# Mortal 模型
mortal_path = "/root/dylan/icml2026/Akagi/mjai_bot/mortal/mortal.pth"
try:
    mortal_state = torch.load(mortal_path, weights_only=False, map_location='cpu')
    m_cfg = mortal_state['config']
    
    result["mortal"] = {
        "path": mortal_path,
        "version": m_cfg['control'].get('version', 1),
        "conv_channels": m_cfg['resnet']['conv_channels'],
        "num_blocks": m_cfg['resnet']['num_blocks']
    }
    
    print("\n[Mortal 模型 (Akagi)]")
    print(f"  路径: {mortal_path}")
    print(f"  Version: {result['mortal']['version']}")
    print(f"  Conv Channels: {result['mortal']['conv_channels']}")
    print(f"  Num Blocks: {result['mortal']['num_blocks']}")
except Exception as e:
    print(f"\n[ERROR] 加载 Mortal 模型失败: {e}")
    result["mortal"]["error"] = str(e)

# 兼容性检查
if result["walking"].get("version") and result["mortal"].get("version"):
    result["compatible"] = result["walking"]["version"] == result["mortal"]["version"]
    
    print("\n" + "=" * 70)
    print("兼容性检查")
    print("=" * 70)
    
    if result["compatible"]:
        print(f"✓ Version 兼容 (都是 v{result['walking']['version']})")
    else:
        print(f"✗ Version 不兼容: WALKING={result['walking']['version']}, Mortal={result['mortal']['version']}")

# 输出 JSON 结果 (供进化系统解析)
print("\n[DEBUG_JSON]")
print(json.dumps(result, indent=2))
print("[/DEBUG_JSON]")
PYTHON_CHECK
}

# =============================================================================
# 运行对战 (核心函数)
# =============================================================================
run_battle() {
    ITERS=${1:-10}
    
    log_step "开始 WALKING vs Mortal 对战 (${ITERS} 轮)..."
    
    cd ${WALKING_DIR}
    
    # 记录开始状态
    write_debug_status "running"
    
    # 使用 set +e 允许 Python 脚本返回非零退出码
    set +e
    
    python3 << PYTHON_BATTLE
import prelude
import numpy as np
import torch
import secrets
import json
import time
import sys
import os

# 添加进化系统路径
sys.path.insert(0, os.environ.get('EVOLUTION_DIR', '/root/dylan/icml2026/WALKING/evolution'))

from model import Brain, DQN
from engine import WalkingEngine
from libriichi.arena import OneVsThree

# 尝试导入调试写入器
try:
    from debug_writer import DebugWriter
    debug_writer = DebugWriter()
except ImportError:
    debug_writer = None
    print("[WARN] debug_writer not available")

# ===========================================
# 自动检测 CUDA 设备配置
# ===========================================
def get_device_config():
    """自动检测并返回最优设备配置"""
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，使用 CPU (性能会很慢)")
        return torch.device('cpu'), torch.device('cpu')
    
    device_count = torch.cuda.device_count()
    print(f"[INFO] 检测到 {device_count} 个 CUDA 设备")
    
    if device_count >= 2:
        # 双GPU: 各用一个
        challenger_device = torch.device('cuda:0')
        champion_device = torch.device('cuda:1')
        print(f"[INFO] 双GPU模式: Challenger=cuda:0, Champion=cuda:1")
    else:
        # 单GPU: 共享 cuda:0
        challenger_device = torch.device('cuda:0')
        champion_device = torch.device('cuda:0')
        print(f"[INFO] 单GPU模式: 两个模型共享 cuda:0")
    
    return challenger_device, champion_device

# 获取设备配置
CHALLENGER_DEVICE, CHAMPION_DEVICE = get_device_config()

# 配置
ITERS = ${ITERS}
GAMES_PER_ITER = 2000
SEEDS_PER_ITER = GAMES_PER_ITER // 4

key = secrets.randbits(64)
start_time = time.time()

print("=" * 70)
print("WALKING vs Mortal 对战")
print("=" * 70)

# ============================================
# 加载 Challenger (WALKING 模型)
# ============================================
print("\n[1] 加载 Challenger (WALKING)...")
walking_path = "/root/dylan/icml2026/WALKING/workdir/checkpoints/walking.pth"
try:
    state = torch.load(walking_path, weights_only=True, map_location='cpu')
    cfg = state['config']

    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']

    print(f"    Version: {version}")
    print(f"    Conv Channels: {conv_channels}")
    print(f"    Num Blocks: {num_blocks}")
    print(f"    Device: {CHALLENGER_DEVICE}")

    walking_brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    walking_dqn = DQN(version=version).eval()
    walking_brain.load_state_dict(state['walking'])
    walking_dqn.load_state_dict(state['current_dqn'])

    engine_challenger = WalkingEngine(
        walking_brain, walking_dqn,
        is_oracle=False,
        version=version,
        device=CHALLENGER_DEVICE,
        enable_amp=True,
        enable_rule_based_agari_guard=True,
        name='WALKING',
    )
    print("    ✓ WALKING 加载完成")
except Exception as e:
    import traceback
    error_msg = f"WALKING load failed: {e}"
    print(f"    ✗ {error_msg}")
    if debug_writer:
        # 记录错误，包含文件路径和完整异常信息
        debug_writer.log_error(
            error_msg, 
            fatal=True,
            file_path=walking_path,
            exception=e
        )
        debug_writer.save()
    sys.exit(1)

# ============================================
# 加载 Champion (Mortal 模型)
# ============================================
print("\n[2] 加载 Champion (Mortal from Akagi)...")
mortal_path = "/root/dylan/icml2026/Akagi/mjai_bot/mortal/mortal.pth"
try:
    state = torch.load(mortal_path, weights_only=False, map_location='cpu')
    cfg = state['config']

    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']

    print(f"    Version: {version}")
    print(f"    Conv Channels: {conv_channels}")
    print(f"    Num Blocks: {num_blocks}")
    print(f"    Device: {CHAMPION_DEVICE}")

    mortal_brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    mortal_dqn = DQN(version=version).eval()
    mortal_brain.load_state_dict(state['mortal'])
    mortal_dqn.load_state_dict(state['current_dqn'])

    engine_champion = WalkingEngine(
        mortal_brain, mortal_dqn,
        is_oracle=False,
        version=version,
        device=CHAMPION_DEVICE,
        enable_amp=True,
        enable_rule_based_agari_guard=True,
        name='Mortal',
    )
    print("    ✓ Mortal 加载完成")
except Exception as e:
    import traceback
    error_msg = f"Mortal load failed: {e}"
    print(f"    ✗ {error_msg}")
    if debug_writer:
        # 记录错误，包含文件路径和完整异常信息
        # 同时记录 engine.py 的路径，因为设备错误通常发生在那里
        engine_path = os.path.join(os.environ.get('WALKING_PROJECT_DIR', '/root/dylan/icml2026/WALKING'), 'walking', 'engine.py')
        debug_writer.log_error(
            error_msg, 
            fatal=True,
            file_path=engine_path,  # CUDA 错误通常在 engine.py 中
            exception=e
        )
        # 额外记录模型文件路径
        debug_writer.log(f"Model file: {mortal_path}")
        debug_writer.save()
    sys.exit(1)

# ============================================
# 开始对战
# ============================================
print("\n" + "=" * 70)
print(f"对战配置: {ITERS} 轮 × {GAMES_PER_ITER} 局 = {ITERS * GAMES_PER_ITER} 总局数")
print("=" * 70)

seed_start = 10000
all_rankings = []

for i, seed in enumerate(range(seed_start, seed_start + SEEDS_PER_ITER * ITERS, SEEDS_PER_ITER)):
    print(f"\n--- Round #{i+1}/{ITERS} ---")
    
    try:
        env = OneVsThree(disable_progress_bar=False, log_dir=None)
        rankings = env.py_vs_py(
            challenger=engine_challenger,
            champion=engine_champion,
            seed_start=(seed, key),
            seed_count=SEEDS_PER_ITER,
        )
        rankings = np.array(rankings)
        all_rankings.append(rankings)
        
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        
        # 累计统计
        cumulative = np.sum(all_rankings, axis=0)
        cum_avg_rank = cumulative @ np.arange(1, 5) / cumulative.sum()
        cum_avg_pt = cumulative @ np.array([90, 45, 0, -135]) / cumulative.sum()
        
        print(f"本轮: {rankings} | rank={avg_rank:.3f}, pt={avg_pt:+.1f}")
        print(f"累计: {cumulative} | rank={cum_avg_rank:.3f}, pt={cum_avg_pt:+.1f}")
        
        # 实时记录到调试信息
        if debug_writer:
            debug_writer.log_metric("current_round", i + 1)
            debug_writer.log_metric("cumulative_rank", float(cum_avg_rank))
            
    except Exception as e:
        print(f"    ✗ Round #{i+1} 失败: {e}")
        if debug_writer:
            debug_writer.log_error(f"Round {i+1} failed: {e}")

# ============================================
# 最终结果
# ============================================
end_time = time.time()
duration = end_time - start_time

# 检查是否有有效结果
if len(all_rankings) == 0:
    error_msg = "No valid battle results - all rounds failed"
    print(f"\n[ERROR] {error_msg}")
    if debug_writer:
        debug_writer.log_error(error_msg, fatal=True)
        debug_writer.save()
    sys.exit(1)

total = np.sum(all_rankings, axis=0)
final_avg_rank = total @ np.arange(1, 5) / total.sum()
final_avg_pt = total @ np.array([90, 45, 0, -135]) / total.sum()

print("\n" + "=" * 70)
print("最终结果")
print("=" * 70)

total_games = total.sum()
print(f"\nWALKING 排名分布 (共 {total_games} 局):")
print(f"  1位: {total[0]:4d} ({total[0]/total_games*100:5.1f}%)")
print(f"  2位: {total[1]:4d} ({total[1]/total_games*100:5.1f}%)")
print(f"  3位: {total[2]:4d} ({total[2]/total_games*100:5.1f}%)")
print(f"  4位: {total[3]:4d} ({total[3]/total_games*100:5.1f}%)")

print(f"\n  平均排名: {final_avg_rank:.4f}")
print(f"  平均得分: {final_avg_pt:+.2f}pt")
print(f"  运行时长: {duration:.1f}秒")

# 计算置信区间
std_rank = np.sqrt(np.sum([(r - final_avg_rank)**2 * total[r-1] for r in range(1,5)]) / total_games)
se_rank = std_rank / np.sqrt(total_games)

print(f"\n  排名标准误: ±{se_rank:.4f}")
print(f"  95%置信区间: [{final_avg_rank - 1.96*se_rank:.4f}, {final_avg_rank + 1.96*se_rank:.4f}]")

# 判断结果
print("\n" + "-" * 70)
if final_avg_rank < 2.4:
    verdict = "🏆 WALKING 显著优于 Mortal!"
    status = "excellent"
elif final_avg_rank < 2.5:
    verdict = "✓ WALKING 略优于 Mortal"
    status = "success"
elif final_avg_rank < 2.6:
    verdict = "= 两者表现相当"
    status = "needs_improvement"
elif final_avg_rank < 2.7:
    verdict = "✗ Mortal 略优于 WALKING"
    status = "poor"
else:
    verdict = "❌ Mortal 显著优于 WALKING"
    status = "poor"

print(verdict)
print("-" * 70)

# ============================================
# 写入调试信息 (Jeff Dean 思想)
# ============================================
if debug_writer:
    debug_writer.record_battle_result(
        rankings=total.tolist(),
        avg_rank=float(final_avg_rank),
        avg_pt=float(final_avg_pt),
        total_games=int(total_games),
        duration_seconds=duration
    )
    debug_writer.log(f"Battle completed: {verdict}")
    debug_writer.save()

# 输出结构化 JSON (供进化系统解析)
result_json = {
    "status": status,
    "rankings": total.tolist(),
    "avg_rank": float(final_avg_rank),
    "avg_pt": float(final_avg_pt),
    "total_games": int(total_games),
    "duration_seconds": duration,
    "confidence_interval": [
        float(final_avg_rank - 1.96*se_rank),
        float(final_avg_rank + 1.96*se_rank)
    ],
    "verdict": verdict
}

print("\n[RESULT_JSON]")
print(json.dumps(result_json, indent=2))
print("[/RESULT_JSON]")

# 成功退出
sys.exit(0)
PYTHON_BATTLE
    
    # 捕获 Python 脚本的退出码
    SCRIPT_EXIT_CODE=$?
    set -e
    
    # 如果 Python 脚本失败，传播错误
    if [ ${SCRIPT_EXIT_CODE} -ne 0 ]; then
        log_error "对战脚本执行失败 (exit code: ${SCRIPT_EXIT_CODE})"
        return ${SCRIPT_EXIT_CODE}
    fi
    
    return 0
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    echo "======================================================================"
    echo "WALKING vs Mortal 对战测试 (进化版 Generation 1)"
    echo "======================================================================"
    echo ""
    
    init_debug
    activate_env
    
    # 先检测 CUDA 设备
    detect_cuda_devices
    
    case "${1:-quick}" in
        check)
            check_models
            ;;
        quick|"")
            check_models
            run_battle 10
            ;;
        full)
            check_models
            run_battle 100
            ;;
        *)
            echo "用法: bash walking_vs_mortal.sh [check|quick|full]"
            echo "  check - 仅检查模型兼容性"
            echo "  quick - 快速测试 (10轮, 默认)"
            echo "  full  - 完整测试 (100轮)"
            exit 1
            ;;
    esac
}

main "$@"