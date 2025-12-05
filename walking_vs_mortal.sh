#!/bin/bash
#
# WALKING vs Mortal 对战脚本
# 使用 Akagi 仓库中的 mortal.pth 作为 champion
#
# 用法:
#   bash walking_vs_mortal.sh          # 快速测试 (10轮)
#   bash walking_vs_mortal.sh full     # 完整测试 (100轮)
#   bash walking_vs_mortal.sh check    # 仅检查模型

set -e

# =============================================================================
# 配置变量
# =============================================================================
PROJECT_DIR="/data/jiacheng/system/cache/temp/icml2026/WALKING"
AKAGI_DIR="/data/jiacheng/system/cache/temp/icml2026/Akagi"
CONDA_ENV="walking3"
WALKING_DIR="${PROJECT_DIR}/walking"

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
    
    python << 'PYTHON_CHECK'
import torch

print("=" * 70)
print("模型对比")
print("=" * 70)

# WALKING 模型
walking_path = "/data/jiacheng/system/cache/temp/icml2026/WALKING/workdir/checkpoints/walking.pth"
walking_state = torch.load(walking_path, weights_only=True, map_location='cpu')
w_cfg = walking_state['config']

print("\n[WALKING 模型]")
print(f"  路径: {walking_path}")
print(f"  Version: {w_cfg['control'].get('version', 1)}")
print(f"  Conv Channels: {w_cfg['resnet']['conv_channels']}")
print(f"  Num Blocks: {w_cfg['resnet']['num_blocks']}")
print(f"  权重 Key: 'walking'")

# Mortal 模型
mortal_path = "/data/jiacheng/system/cache/temp/icml2026/Akagi/mjai_bot/mortal/mortal.pth"
mortal_state = torch.load(mortal_path, weights_only=False, map_location='cpu')
m_cfg = mortal_state['config']

print("\n[Mortal 模型 (Akagi)]")
print(f"  路径: {mortal_path}")
print(f"  Version: {m_cfg['control'].get('version', 1)}")
print(f"  Conv Channels: {m_cfg['resnet']['conv_channels']}")
print(f"  Num Blocks: {m_cfg['resnet']['num_blocks']}")
print(f"  权重 Key: 'mortal'")

print("\n" + "=" * 70)
print("兼容性检查")
print("=" * 70)

w_ver = w_cfg['control'].get('version', 1)
m_ver = m_cfg['control'].get('version', 1)

if w_ver == m_ver:
    print(f"✓ Version 兼容 (都是 v{w_ver})")
else:
    print(f"✗ Version 不兼容: WALKING={w_ver}, Mortal={m_ver}")

print(f"\n模型大小对比:")
print(f"  WALKING: {w_cfg['resnet']['conv_channels']}ch × {w_cfg['resnet']['num_blocks']}blocks")
print(f"  Mortal:  {m_cfg['resnet']['conv_channels']}ch × {m_cfg['resnet']['num_blocks']}blocks")

# 计算参数量
def count_params(state_dict):
    return sum(p.numel() for p in state_dict.values())

w_params = count_params(walking_state['walking'])
m_params = count_params(mortal_state['mortal'])
print(f"\n参数量:")
print(f"  WALKING Brain: {w_params:,} ({w_params/1e6:.2f}M)")
print(f"  Mortal Brain:  {m_params:,} ({m_params/1e6:.2f}M)")
print(f"  比例: WALKING 是 Mortal 的 {w_params/m_params:.1f}x")
PYTHON_CHECK
}

# =============================================================================
# 运行对战
# =============================================================================
run_battle() {
    ITERS=${1:-10}
    
    log_step "开始 WALKING vs Mortal 对战 (${ITERS} 轮)..."
    
    cd ${WALKING_DIR}
    
    python << PYTHON_BATTLE
import prelude
import numpy as np
import torch
import secrets
from model import Brain, DQN
from engine import WalkingEngine
from libriichi.arena import OneVsThree

# 配置
ITERS = ${ITERS}
GAMES_PER_ITER = 2000
SEEDS_PER_ITER = GAMES_PER_ITER // 4

key = secrets.randbits(64)

print("=" * 70)
print("WALKING vs Mortal 对战")
print("=" * 70)

# ============================================
# 加载 Challenger (WALKING 模型)
# ============================================
print("\n[1] 加载 Challenger (WALKING)...")
walking_path = "/data/jiacheng/system/cache/temp/icml2026/WALKING/workdir/checkpoints/walking.pth"
state = torch.load(walking_path, weights_only=True, map_location='cpu')
cfg = state['config']

version = cfg['control'].get('version', 1)
conv_channels = cfg['resnet']['conv_channels']
num_blocks = cfg['resnet']['num_blocks']

print(f"    Version: {version}")
print(f"    Conv Channels: {conv_channels}")
print(f"    Num Blocks: {num_blocks}")

walking_brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
walking_dqn = DQN(version=version).eval()
walking_brain.load_state_dict(state['walking'])
walking_dqn.load_state_dict(state['current_dqn'])

engine_challenger = WalkingEngine(
    walking_brain, walking_dqn,
    is_oracle=False,
    version=version,
    device=torch.device('cuda:0'),
    enable_amp=True,
    enable_rule_based_agari_guard=True,
    name='WALKING',
)
print("    ✓ WALKING 加载完成")

# ============================================
# 加载 Champion (Mortal 模型)
# ============================================
print("\n[2] 加载 Champion (Mortal from Akagi)...")
mortal_path = "/data/jiacheng/system/cache/temp/icml2026/Akagi/mjai_bot/mortal/mortal.pth"
state = torch.load(mortal_path, weights_only=False, map_location='cpu')
cfg = state['config']

version = cfg['control'].get('version', 1)
conv_channels = cfg['resnet']['conv_channels']
num_blocks = cfg['resnet']['num_blocks']

print(f"    Version: {version}")
print(f"    Conv Channels: {conv_channels}")
print(f"    Num Blocks: {num_blocks}")

mortal_brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
mortal_dqn = DQN(version=version).eval()

# 注意: Mortal 的权重 key 是 'mortal' 而不是 'walking'
mortal_brain.load_state_dict(state['mortal'])
mortal_dqn.load_state_dict(state['current_dqn'])

engine_champion = WalkingEngine(
    mortal_brain, mortal_dqn,
    is_oracle=False,
    version=version,
    device=torch.device('cuda:1'),
    enable_amp=True,
    enable_rule_based_agari_guard=True,
    name='Mortal',
)
print("    ✓ Mortal 加载完成")

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

# ============================================
# 最终结果
# ============================================
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

# 计算置信区间 (简化版)
std_rank = np.sqrt(np.sum([(r - final_avg_rank)**2 * total[r-1] for r in range(1,5)]) / total_games)
se_rank = std_rank / np.sqrt(total_games)

print(f"\n  排名标准误: ±{se_rank:.4f}")
print(f"  95%置信区间: [{final_avg_rank - 1.96*se_rank:.4f}, {final_avg_rank + 1.96*se_rank:.4f}]")

print("\n" + "-" * 70)
if final_avg_rank < 2.4:
    print("🏆 WALKING 显著优于 Mortal!")
elif final_avg_rank < 2.5:
    print("✓ WALKING 略优于 Mortal")
elif final_avg_rank < 2.6:
    print("= 两者表现相当")
elif final_avg_rank < 2.7:
    print("✗ Mortal 略优于 WALKING")
else:
    print("❌ Mortal 显著优于 WALKING")
print("-" * 70)
PYTHON_BATTLE
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    echo "======================================================================"
    echo "WALKING vs Mortal 对战测试"
    echo "======================================================================"
    echo ""
    
    activate_env
    
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