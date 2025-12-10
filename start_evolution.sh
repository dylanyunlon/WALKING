#!/bin/bash
#
# Walking 进化系统快速启动脚本
# 
# 功能:
#   1. 自动检测项目目录
#   2. 验证环境基本可用
#   3. 调用主进化脚本 evolution_runner.sh
#
# 用法:
#   bash start_evolution.sh                    # 启动进化循环
#   bash start_evolution.sh --once             # 只运行一次
#   bash start_evolution.sh --dry-run          # 试运行
#   bash start_evolution.sh --status           # 查看状态
#   bash start_evolution.sh --check-deps       # 检查依赖
#   bash start_evolution.sh --mode quick       # 指定运行模式
#   bash start_evolution.sh --once --mode full # 组合使用

set -e

# =============================================================================
# 动态路径检测
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_project_dir() {
    # 1. 环境变量
    if [ -n "${WALKING_PROJECT_DIR}" ] && [ -d "${WALKING_PROJECT_DIR}" ]; then
        echo "${WALKING_PROJECT_DIR}"
        return
    fi
    
    # 2. 脚本所在目录
    if [ -f "${SCRIPT_DIR}/walking_vs_mortal.sh" ]; then
        echo "${SCRIPT_DIR}"
        return
    fi
    
    # 3. 脚本所在目录的父目录 (如果脚本在 evolution/ 下)
    if [ -f "${SCRIPT_DIR}/../walking_vs_mortal.sh" ]; then
        echo "$(cd "${SCRIPT_DIR}/.." && pwd)"
        return
    fi
    
    # 4. 向上查找
    local current="${SCRIPT_DIR}"
    for i in {1..5}; do
        if [ -f "${current}/walking_vs_mortal.sh" ]; then
            echo "${current}"
            return
        fi
        current="$(dirname "${current}")"
    done
    
    # 5. 常见路径
    for p in "/root/dylan/icml2026/WALKING" "${HOME}/WALKING" "/workspace/WALKING"; do
        if [ -d "${p}" ]; then
            echo "${p}"
            return
        fi
    done
    
    # 6. 当前目录
    pwd
}

PROJECT_DIR="$(detect_project_dir)"
EVOLUTION_DIR="${PROJECT_DIR}/evolution"
RUNNER_SCRIPT="${EVOLUTION_DIR}/evolution_runner.sh"

# 导出供子脚本使用
export WALKING_PROJECT_DIR="${PROJECT_DIR}"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# 预检查函数
# =============================================================================
preflight_check() {
    local has_error=false
    
    # 检查项目目录
    if [ ! -d "${PROJECT_DIR}" ]; then
        echo -e "${RED}[ERROR]${NC} 项目目录不存在: ${PROJECT_DIR}"
        has_error=true
    fi
    
    # 检查进化目录
    if [ ! -d "${EVOLUTION_DIR}" ]; then
        echo -e "${YELLOW}[WARN]${NC} 进化目录不存在，正在创建: ${EVOLUTION_DIR}"
        mkdir -p "${EVOLUTION_DIR}"
    fi
    
    # 检查主脚本
    if [ ! -f "${RUNNER_SCRIPT}" ]; then
        echo -e "${RED}[ERROR]${NC} 主进化脚本不存在: ${RUNNER_SCRIPT}"
        echo -e "${YELLOW}[HINT]${NC} 请先运行 install_evolution.sh 或手动放置 evolution_runner.sh"
        has_error=true
    fi
    
    # 如果有错误，退出
    if [ "${has_error}" = "true" ]; then
        exit 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    # 运行预检查
    preflight_check
    
    # 切换到进化目录并运行主脚本
    cd "${EVOLUTION_DIR}"
    bash "${RUNNER_SCRIPT}" "$@"
}

main "$@"