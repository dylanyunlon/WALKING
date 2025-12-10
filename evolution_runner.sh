#!/bin/bash
#
# Walking 自进化系统 - 主循环控制器
# 
# 核心思想 (来自"进化思想.md"):
#   程序A运行 → 产生日志 → 发给LLM → LLM返回A' → 覆盖A → 循环
#   success/error 来自真实世界，LLM只是"修复建议器"
#
# Jeff Dean 理论:
#   将重要的调试信息写入文件，下一轮用bash读取
#
# 用法:
#   bash evolution_runner.sh              # 启动进化循环
#   bash evolution_runner.sh --once       # 只运行一次
#   bash evolution_runner.sh --dry-run    # 试运行，不覆盖文件

set -e

# =============================================================================
# 配置变量 - 动态路径检测
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目目录 - 优先使用环境变量，否则自动检测
detect_project_dir() {
    # 1. 环境变量
    if [ -n "${WALKING_PROJECT_DIR}" ] && [ -d "${WALKING_PROJECT_DIR}" ]; then
        echo "${WALKING_PROJECT_DIR}"
        return
    fi
    
    # 2. 脚本所在目录的父目录
    if [ -f "${SCRIPT_DIR}/../walking_vs_mortal.sh" ]; then
        echo "$(cd "${SCRIPT_DIR}/.." && pwd)"
        return
    fi
    
    # 3. 脚本所在目录
    if [ -f "${SCRIPT_DIR}/walking_vs_mortal.sh" ]; then
        echo "${SCRIPT_DIR}"
        return
    fi
    
    # 4. 向上查找特征文件
    local current="${SCRIPT_DIR}"
    for i in {1..5}; do
        if [ -f "${current}/walking_vs_mortal.sh" ] || [ -f "${current}/Cargo.toml" ]; then
            echo "${current}"
            return
        fi
        current="$(dirname "${current}")"
    done
    
    # 5. 常见路径
    local common_paths=(
        "/root/dylan/icml2026/WALKING"
        "${HOME}/WALKING"
        "/workspace/WALKING"
    )
    for p in "${common_paths[@]}"; do
        if [ -d "${p}" ]; then
            echo "${p}"
            return
        fi
    done
    
    # 6. 回退到当前目录
    pwd
}

PROJECT_DIR="$(detect_project_dir)"
WALKING_DIR="${PROJECT_DIR}/walking"
CONDA_ENV="${WALKING_CONDA_ENV:-walking3}"

# 进化系统配置
EVOLUTION_DIR="${PROJECT_DIR}/evolution"
DEBUG_FILE="${EVOLUTION_DIR}/debug_info.json"
HISTORY_DIR="${EVOLUTION_DIR}/history"
LOG_FILE="${EVOLUTION_DIR}/evolution.log"

# 目标脚本 (要被进化的程序A)
TARGET_SCRIPT="${EVOLUTION_TARGET_SCRIPT:-${PROJECT_DIR}/walking_vs_mortal.sh}"

# 目标脚本运行模式配置
# 可选值: quick (10轮), full (100轮), check (仅检查)
# 可被环境变量 EVOLUTION_RUN_MODE 覆盖
TARGET_RUN_MODE="${EVOLUTION_RUN_MODE:-quick}"

# LLM API 配置 - 支持环境变量覆盖
API_BASE_URL="${EVOLUTION_API_URL:-https://balloonet.tech:17432}"
API_USERNAME="${EVOLUTION_API_USER:-newuser}"
API_PASSWORD="${EVOLUTION_API_PASS:-newPass123}"

# 进化间隔 (秒)
EVOLUTION_INTERVAL="${EVOLUTION_INTERVAL:-60}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO $(date '+%H:%M:%S')]${NC} $1" | tee -a ${LOG_FILE}; }
log_warn()  { echo -e "${YELLOW}[WARN $(date '+%H:%M:%S')]${NC} $1" | tee -a ${LOG_FILE}; }
log_error() { echo -e "${RED}[ERROR $(date '+%H:%M:%S')]${NC} $1" | tee -a ${LOG_FILE}; }
log_step()  { echo -e "${BLUE}[STEP $(date '+%H:%M:%S')]${NC} $1" | tee -a ${LOG_FILE}; }
log_evolution() { echo -e "${MAGENTA}[EVOLUTION $(date '+%H:%M:%S')]${NC} $1" | tee -a ${LOG_FILE}; }

# =============================================================================
# 全局依赖状态变量
# =============================================================================
declare -g HAS_JQ=false
declare -g HAS_BC=false
declare -g HAS_CURL=false
declare -g HAS_PYTHON3=false

# 依赖注册表 - 易于扩展
# 格式: "命令名:包名:描述:是否必需"
DEPENDENCIES=(
    "jq:jq:JSON处理工具:required"
    "bc:bc:数学计算工具:required"
    "curl:curl:网络请求工具:optional"
    "python3:python3:Python解释器:required"
)

# =============================================================================
# 依赖检查与自动安装
# =============================================================================
check_command_exists() {
    command -v "$1" &> /dev/null
}

install_package() {
    local package_name="$1"
    local description="$2"
    
    log_info "正在安装 ${package_name} (${description})..."
    
    # 检测包管理器并安装
    if check_command_exists apt-get; then
        # Debian/Ubuntu
        sudo apt-get update -qq && sudo apt-get install -y -qq "${package_name}"
    elif check_command_exists apt; then
        # Debian/Ubuntu (newer)
        sudo apt update -qq && sudo apt install -y -qq "${package_name}"
    elif check_command_exists yum; then
        # CentOS/RHEL
        sudo yum install -y -q "${package_name}"
    elif check_command_exists dnf; then
        # Fedora
        sudo dnf install -y -q "${package_name}"
    elif check_command_exists pacman; then
        # Arch Linux
        sudo pacman -S --noconfirm --quiet "${package_name}"
    elif check_command_exists zypper; then
        # openSUSE
        sudo zypper install -y -q "${package_name}"
    elif check_command_exists apk; then
        # Alpine
        sudo apk add --quiet "${package_name}"
    else
        log_error "无法检测到包管理器，请手动安装 ${package_name}"
        return 1
    fi
    
    return $?
}

check_and_install_dependencies() {
    log_step "检查系统依赖..."
    
    local all_satisfied=true
    local installed_count=0
    local failed_required=false
    
    for dep_entry in "${DEPENDENCIES[@]}"; do
        # 解析依赖条目
        IFS=':' read -r cmd_name pkg_name description requirement <<< "${dep_entry}"
        
        # 构造全局变量名 (例如: jq -> HAS_JQ)
        local var_name="HAS_$(echo "${cmd_name}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
        
        if check_command_exists "${cmd_name}"; then
            # 已安装 - 设置全局变量
            declare -g "${var_name}=true"
            log_info "✓ ${cmd_name} 已安装 (${description})"
        else
            # 未安装 - 尝试自动安装
            log_warn "✗ ${cmd_name} 未安装 (${description})"
            
            if install_package "${pkg_name}" "${description}"; then
                # 安装成功 - 再次检查并设置变量
                if check_command_exists "${cmd_name}"; then
                    declare -g "${var_name}=true"
                    log_info "✓ ${cmd_name} 安装成功!"
                    ((installed_count++))
                else
                    declare -g "${var_name}=false"
                    log_error "✗ ${cmd_name} 安装后仍不可用"
                    if [ "${requirement}" = "required" ]; then
                        failed_required=true
                    fi
                fi
            else
                declare -g "${var_name}=false"
                log_error "✗ ${cmd_name} 安装失败"
                if [ "${requirement}" = "required" ]; then
                    failed_required=true
                fi
                all_satisfied=false
            fi
        fi
    done
    
    # 输出摘要
    echo ""
    log_info "依赖检查完成:"
    log_info "  - JQ:      ${HAS_JQ}"
    log_info "  - BC:      ${HAS_BC}"
    log_info "  - CURL:    ${HAS_CURL}"
    log_info "  - Python3: ${HAS_PYTHON3}"
    
    if [ ${installed_count} -gt 0 ]; then
        log_info "本次自动安装了 ${installed_count} 个依赖"
    fi
    
    if [ "${failed_required}" = "true" ]; then
        log_error "存在必需依赖安装失败，系统无法继续"
        exit 1
    fi
    
    return 0
}

# =============================================================================
# 添加新依赖的辅助函数 (供未来扩展使用)
# =============================================================================
add_dependency() {
    local cmd_name="$1"
    local pkg_name="${2:-$1}"
    local description="${3:-$1 tool}"
    local requirement="${4:-optional}"
    
    DEPENDENCIES+=("${cmd_name}:${pkg_name}:${description}:${requirement}")
}

# =============================================================================
# 初始化
# =============================================================================
init_evolution_system() {
    log_step "Initializing evolution system..."
    
    # 创建必要目录
    mkdir -p ${EVOLUTION_DIR}
    mkdir -p ${HISTORY_DIR}
    
    # 初始化日志
    echo "===== Evolution System Started at $(date) =====" >> ${LOG_FILE}
    
    # ========== 自动检查并安装依赖 ==========
    check_and_install_dependencies
    
    # 初始化调试信息文件
    if [ ! -f "${DEBUG_FILE}" ]; then
        cat > ${DEBUG_FILE} << 'EOF'
{
    "generation": 0,
    "last_run": null,
    "last_status": "init",
    "last_error": null,
    "metrics": {
        "avg_rank": null,
        "avg_pt": null,
        "total_games": 0
    },
    "history": []
}
EOF
        log_info "创建初始调试信息文件"
    fi
    
    # 检查目标脚本
    if [ ! -f "${TARGET_SCRIPT}" ]; then
        log_error "目标脚本不存在: ${TARGET_SCRIPT}"
        exit 1
    fi
    
    # 备份原始脚本
    if [ ! -f "${HISTORY_DIR}/generation_0.sh" ]; then
        cp ${TARGET_SCRIPT} ${HISTORY_DIR}/generation_0.sh
        log_info "备份原始脚本为 generation_0.sh"
    fi
    
    log_info "Evolution system initialization complete"
}

# =============================================================================
# 激活 Conda 环境
# =============================================================================
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
    cd ${PROJECT_DIR}
}

# =============================================================================
# 运行目标脚本并捕获输出
# =============================================================================
run_target_script() {
    log_step "运行目标脚本: ${TARGET_SCRIPT} ${TARGET_RUN_MODE}"
    
    local OUTPUT_FILE="${EVOLUTION_DIR}/last_output.txt"
    local ERROR_FILE="${EVOLUTION_DIR}/last_error.txt"
    local START_TIME=$(date +%s)
    
    # 运行脚本，捕获输出和错误
    # 使用配置的运行模式
    set +e  # 暂时允许错误
    bash ${TARGET_SCRIPT} ${TARGET_RUN_MODE} 2>${ERROR_FILE} | tee ${OUTPUT_FILE}
    local EXIT_CODE=$?
    set -e
    
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    
    # 更新调试信息 (使用已验证的 jq)
    local GENERATION=$(jq -r '.generation' ${DEBUG_FILE})
    local NEW_GENERATION=$((GENERATION + 1))
    
    # 提取关键指标 (从输出中解析)
    local AVG_RANK=$(grep -oP '平均排名: \K[\d.]+' ${OUTPUT_FILE} 2>/dev/null || echo "null")
    local AVG_PT=$(grep -oP '平均得分: \K[+-]?[\d.]+' ${OUTPUT_FILE} 2>/dev/null || echo "null")
    local TOTAL_GAMES=$(grep -oP '共 \K\d+' ${OUTPUT_FILE} 2>/dev/null || echo "0")
    
    # 读取错误信息
    local ERROR_MSG=""
    if [ -s "${ERROR_FILE}" ]; then
        ERROR_MSG=$(cat ${ERROR_FILE} | head -50 | tr '\n' ' ' | sed 's/"/\\"/g')
    fi
    
    # 检查 debug_info.json 中是否有致命错误
    local HAS_FATAL_ERROR=false
    if [ -f "${DEBUG_FILE}" ]; then
        local FATAL_CHECK=$(jq -r '.session_data.errors[]? | select(.fatal == true) | .error' ${DEBUG_FILE} 2>/dev/null | head -1)
        if [ -n "${FATAL_CHECK}" ]; then
            HAS_FATAL_ERROR=true
            if [ -z "${ERROR_MSG}" ]; then
                ERROR_MSG="${FATAL_CHECK}"
            fi
            log_warn "检测到致命错误: ${FATAL_CHECK:0:100}..."
        fi
    fi
    
    # 确定状态 (改进的逻辑)
    local STATUS="success"
    if [ ${EXIT_CODE} -ne 0 ]; then
        STATUS="error"
        log_warn "脚本退出码非零: ${EXIT_CODE}"
    elif [ "${HAS_FATAL_ERROR}" = "true" ]; then
        STATUS="error"
        log_warn "检测到致命错误标记"
    elif [ "${AVG_RANK}" = "null" ] && [ "${TOTAL_GAMES}" = "0" ]; then
        # 没有产生任何对战结果，也视为错误
        STATUS="error"
        log_warn "未产生有效对战结果"
    elif [ "${AVG_RANK}" != "null" ] && [ $(echo "${AVG_RANK} > 2.6" | bc -l) -eq 1 ]; then
        STATUS="needs_improvement"
    fi
    
    # 更新 debug_info.json
    local TIMESTAMP=$(date -Iseconds)
    jq --arg ts "${TIMESTAMP}" \
       --arg status "${STATUS}" \
       --arg error "${ERROR_MSG}" \
       --argjson rank "${AVG_RANK:-null}" \
       --argjson pt "${AVG_PT:-null}" \
       --argjson games "${TOTAL_GAMES:-0}" \
       --argjson gen "${NEW_GENERATION}" \
       --argjson duration "${DURATION}" \
       --argjson exit_code "${EXIT_CODE}" \
       '. + {
           generation: $gen,
           last_run: $ts,
           last_status: $status,
           last_error: (if $error == "" then null else $error end),
           last_exit_code: $exit_code,
           last_duration_seconds: $duration,
           metrics: {
               avg_rank: $rank,
               avg_pt: $pt,
               total_games: $games
           }
       }' ${DEBUG_FILE} > ${DEBUG_FILE}.tmp && mv ${DEBUG_FILE}.tmp ${DEBUG_FILE}
    
    log_info "运行完成 - 状态: ${STATUS}, 排名: ${AVG_RANK}, 得分: ${AVG_PT}"
    
    # 返回状态
    echo ${STATUS}
}

# =============================================================================
# 调用 LLM 获取改进建议
# =============================================================================
call_llm_for_evolution() {
    log_step "调用 LLM 获取进化建议..."
    
    # 读取当前脚本内容
    local CURRENT_SCRIPT=$(cat ${TARGET_SCRIPT})
    
    # 读取调试信息
    local DEBUG_INFO=$(cat ${DEBUG_FILE})
    
    # 读取最近的输出
    local LAST_OUTPUT=""
    if [ -f "${EVOLUTION_DIR}/last_output.txt" ]; then
        LAST_OUTPUT=$(tail -100 ${EVOLUTION_DIR}/last_output.txt)
    fi
    
    # 读取最近的错误
    local LAST_ERROR=""
    if [ -f "${EVOLUTION_DIR}/last_error.txt" ]; then
        LAST_ERROR=$(cat ${EVOLUTION_DIR}/last_error.txt | head -50)
    fi
    
    # 导出环境变量供 Python 使用
    export WALKING_PROJECT_DIR="${PROJECT_DIR}"
    export EVOLUTION_TARGET_SCRIPT="${TARGET_SCRIPT}"
    export EVOLUTION_API_URL="${API_BASE_URL}"
    export EVOLUTION_API_USER="${API_USERNAME}"
    export EVOLUTION_API_PASS="${API_PASSWORD}"
    
    # 调用 Python 客户端
    cd ${EVOLUTION_DIR}
    
    python3 << 'PYTHON_EVOLUTION'
import asyncio
import json
import sys
import os

# 从环境变量获取路径
evolution_dir = os.environ.get('WALKING_PROJECT_DIR', '') + '/evolution'
sys.path.insert(0, evolution_dir)

from evolution_client import EvolutionClient

async def get_evolution():
    async with EvolutionClient() as client:
        # 登录 (会自动从环境变量获取凭据)
        if not await client.login():
            print("FAILED: 登录失败")
            return
        
        # 读取文件
        target_script = os.environ.get('EVOLUTION_TARGET_SCRIPT', client.target_script)
        debug_file = os.path.join(client.evolution_dir, 'debug_info.json')
        
        with open(target_script, "r") as f:
            current_script = f.read()
        
        with open(debug_file, "r") as f:
            debug_info = json.load(f)
        
        last_output = ""
        try:
            with open(os.path.join(str(client.evolution_dir), "last_output.txt"), "r") as f:
                last_output = f.read()[-5000:]
        except:
            pass
        
        last_error = ""
        try:
            with open(os.path.join(str(client.evolution_dir), "last_error.txt"), "r") as f:
                last_error = f.read()[:2000]
        except:
            pass
        
        # 构建进化请求
        result = await client.request_evolution(
            current_script=current_script,
            debug_info=debug_info,
            last_output=last_output,
            last_error=last_error
        )
        
        if result["success"]:
            # 保存新脚本 (主脚本)
            new_script = result["new_script"]
            new_script_path = os.path.join(str(client.evolution_dir), "new_script.sh")
            with open(new_script_path, "w") as f:
                f.write(new_script)
            
            # 保存文件修改列表 (用于多文件支持)
            file_changes = result.get("file_changes", [])
            if file_changes:
                changes_path = os.path.join(str(client.evolution_dir), "file_changes.json")
                with open(changes_path, "w") as f:
                    json.dump(file_changes, f, indent=2)
                print(f"FILE_CHANGES: {len(file_changes)} 个文件")
            
            # 保存 shell 命令 (如果有)
            shell_commands = result.get("shell_commands", [])
            if shell_commands:
                cmd_path = os.path.join(str(client.evolution_dir), "shell_commands.txt")
                with open(cmd_path, "w") as f:
                    f.write("\n".join(shell_commands))
                print(f"SHELL_COMMANDS: {len(shell_commands)} 条")
            
            print("SUCCESS")
            print(f"改进说明: {result.get('explanation', 'N/A')[:200]}")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

asyncio.run(get_evolution())
PYTHON_EVOLUTION
}

# =============================================================================
# 应用进化 (支持多文件修改)
# =============================================================================
apply_evolution() {
    log_step "应用进化..."
    
    local NEW_SCRIPT="${EVOLUTION_DIR}/new_script.sh"
    local FILE_CHANGES="${EVOLUTION_DIR}/file_changes.json"
    local SHELL_COMMANDS="${EVOLUTION_DIR}/shell_commands.txt"
    
    # 获取当前代数
    local GENERATION=$(jq -r '.generation' ${DEBUG_FILE})
    
    # 1. 执行 shell 命令 (如果有)
    if [ -f "${SHELL_COMMANDS}" ]; then
        log_info "执行 LLM 建议的 shell 命令..."
        while IFS= read -r cmd; do
            if [ -n "${cmd}" ]; then
                log_info "执行: ${cmd}"
                set +e
                eval "${cmd}" 2>&1 | tee -a ${LOG_FILE}
                local CMD_EXIT=$?
                set -e
                if [ ${CMD_EXIT} -ne 0 ]; then
                    log_warn "命令执行返回非零: ${CMD_EXIT}"
                fi
            fi
        done < "${SHELL_COMMANDS}"
        rm -f "${SHELL_COMMANDS}"
    fi
    
    # 2. 应用多文件修改 (如果有)
    if [ -f "${FILE_CHANGES}" ]; then
        log_info "应用多文件修改..."
        
        python3 << PYTHON_APPLY
import json
import os
import shutil
from datetime import datetime

with open("${FILE_CHANGES}", "r") as f:
    changes = json.load(f)

history_dir = "${HISTORY_DIR}"
generation = ${GENERATION}

for change in changes:
    path = change["path"]
    content = change["content"]
    
    try:
        # 备份原文件 (如果存在)
        if os.path.exists(path):
            backup_name = os.path.basename(path) + f".gen{generation}.bak"
            backup_path = os.path.join(history_dir, backup_name)
            shutil.copy2(path, backup_path)
            print(f"  📦 备份: {path} → {backup_path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 写入新内容
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 设置执行权限 (shell 脚本)
        if path.endswith(".sh"):
            os.chmod(path, 0o755)
        
        print(f"  ✅ 写入: {path} ({len(content)} 字符)")
        
    except Exception as e:
        print(f"  ❌ 失败: {path} - {e}")

print(f"共处理 {len(changes)} 个文件")
PYTHON_APPLY
        
        rm -f "${FILE_CHANGES}"
        
    # 3. 回退逻辑: 如果没有 file_changes，使用传统方式
    elif [ -f "${NEW_SCRIPT}" ]; then
        # 验证新脚本语法
        if ! bash -n ${NEW_SCRIPT} 2>/dev/null; then
            log_error "新脚本语法错误，拒绝应用"
            return 1
        fi
        
        # 备份当前版本到历史
        cp ${TARGET_SCRIPT} ${HISTORY_DIR}/generation_${GENERATION}.sh
        log_info "备份当前版本为 generation_${GENERATION}.sh"
        
        # 应用新脚本
        cp ${NEW_SCRIPT} ${TARGET_SCRIPT}
        chmod +x ${TARGET_SCRIPT}
    else
        log_error "没有找到可应用的进化内容"
        return 1
    fi
    
    log_evolution "进化成功! Generation ${GENERATION} → $((GENERATION + 1))"
    
    # 记录进化历史
    jq --arg gen "${GENERATION}" \
       --arg ts "$(date -Iseconds)" \
       '.history += [{"generation": ($gen | tonumber), "timestamp": $ts}]' \
       ${DEBUG_FILE} > ${DEBUG_FILE}.tmp && mv ${DEBUG_FILE}.tmp ${DEBUG_FILE}
    
    return 0
}

# =============================================================================
# 回滚到上一代
# =============================================================================
rollback() {
    log_warn "执行回滚..."
    
    local GENERATION=$(jq -r '.generation' ${DEBUG_FILE})
    local PREV_GEN=$((GENERATION - 1))
    
    if [ ${PREV_GEN} -lt 0 ]; then
        log_error "无法回滚：已经是第0代"
        return 1
    fi
    
    local PREV_SCRIPT="${HISTORY_DIR}/generation_${PREV_GEN}.sh"
    
    if [ ! -f "${PREV_SCRIPT}" ]; then
        log_error "上一代脚本不存在: ${PREV_SCRIPT}"
        return 1
    fi
    
    cp ${PREV_SCRIPT} ${TARGET_SCRIPT}
    chmod +x ${TARGET_SCRIPT}
    
    # 更新代数
    jq --argjson gen "${PREV_GEN}" '.generation = $gen' ${DEBUG_FILE} > ${DEBUG_FILE}.tmp \
        && mv ${DEBUG_FILE}.tmp ${DEBUG_FILE}
    
    log_info "已回滚到 generation_${PREV_GEN}"
}

# =============================================================================
# 主进化循环
# =============================================================================
evolution_loop() {
    local RUN_ONCE=${1:-false}
    local DRY_RUN=${2:-false}
    
    log_evolution "=========================================="
    log_evolution "     Walking self-evolution system started"
    log_evolution "=========================================="
    log_info "Target script: ${TARGET_SCRIPT}"
    log_info "Target run mode: ${TARGET_RUN_MODE}"
    log_info "Evolution interval: ${EVOLUTION_INTERVAL} seconds"
    log_info "Run mode: $([ "${RUN_ONCE}" = "true" ] && echo "Single" || echo "Loop")"
    log_info "Trial run: $([ "${DRY_RUN}" = "true" ] && echo "Yes" || echo "No")"
    
    activate_env
    
    while true; do
        echo ""
        log_evolution ">>> Starting new evolution round <<<"
        
        # 1. 运行目标脚本
        local STATUS=$(run_target_script)
        
        # 2. 根据状态决定是否需要进化
        if [ "${STATUS}" = "success" ]; then
            log_info "运行成功且指标良好，暂不需要进化"
            
            # 检查是否可以挑战更难的任务
            local AVG_RANK=$(jq -r '.metrics.avg_rank' ${DEBUG_FILE})
            if [ "${AVG_RANK}" != "null" ] && [ $(echo "${AVG_RANK} < 2.3" | bc -l) -eq 1 ]; then
                log_evolution "表现优异! 考虑增加难度..."
                # 自适应升级：如果当前是 quick 模式且表现优异，升级到 full 模式
                if [ "${TARGET_RUN_MODE}" = "quick" ]; then
                    log_evolution "升级到 full 模式进行更严格测试"
                    TARGET_RUN_MODE="full"
                fi
            fi
            
        elif [ "${STATUS}" = "needs_improvement" ] || [ "${STATUS}" = "error" ]; then
            log_warn "需要改进，调用 LLM..."
            
            # 3. 调用 LLM 获取改进
            call_llm_for_evolution
            
            # 4. 应用进化 (如果不是试运行)
            if [ "${DRY_RUN}" = "false" ]; then
                if apply_evolution; then
                    log_evolution "进化应用成功!"
                else
                    log_error "进化应用失败"
                fi
            else
                log_info "[DRY-RUN] 跳过应用进化"
                if [ -f "${EVOLUTION_DIR}/new_script.sh" ]; then
                    log_info "新脚本预览 (前50行):"
                    head -50 ${EVOLUTION_DIR}/new_script.sh
                fi
            fi
        fi
        
        # 单次运行则退出
        if [ "${RUN_ONCE}" = "true" ]; then
            log_info "Single run completed, exiting."
            break
        fi
        
        # 等待下一轮
        log_info "等待 ${EVOLUTION_INTERVAL} 秒后进行下一轮..."
        sleep ${EVOLUTION_INTERVAL}
    done
}

# =============================================================================
# 显示状态
# =============================================================================
show_status() {
    echo ""
    echo "=========================================="
    echo "        Walking 进化系统状态"
    echo "=========================================="
    
    # 先检查依赖
    echo ""
    echo "依赖状态:"
    for dep_entry in "${DEPENDENCIES[@]}"; do
        IFS=':' read -r cmd_name pkg_name description requirement <<< "${dep_entry}"
        if check_command_exists "${cmd_name}"; then
            echo -e "  ${GREEN}✓${NC} ${cmd_name} (${description})"
        else
            echo -e "  ${RED}✗${NC} ${cmd_name} (${description}) - ${requirement}"
        fi
    done
    
    if [ -f "${DEBUG_FILE}" ]; then
        echo ""
        echo "调试信息:"
        jq '.' ${DEBUG_FILE}
    fi
    
    echo ""
    echo "历史版本:"
    ls -la ${HISTORY_DIR}/*.sh 2>/dev/null || echo "  (无历史版本)"
    
    echo ""
    echo "最近日志:"
    tail -20 ${LOG_FILE} 2>/dev/null || echo "  (无日志)"
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    # 解析参数
    local CMD=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                TARGET_RUN_MODE="$2"
                shift 2
                ;;
            --mode=*)
                TARGET_RUN_MODE="${1#*=}"
                shift
                ;;
            *)
                CMD="$1"
                shift
                ;;
        esac
    done
    
    case "${CMD:-}" in
        --once)
            init_evolution_system
            evolution_loop true false
            ;;
        --dry-run)
            init_evolution_system
            evolution_loop true true
            ;;
        --status)
            show_status
            ;;
        --rollback)
            init_evolution_system
            activate_env
            rollback
            ;;
        --check-deps)
            # 只检查依赖
            check_and_install_dependencies
            ;;
        --help|-h)
            echo ""
            echo "Walking 自进化系统"
            echo ""
            echo "用法: bash evolution_runner.sh [选项]"
            echo ""
            echo "选项:"
            echo "  (无参数)           启动进化循环 (持续运行)"
            echo "  --once             只运行一次进化"
            echo "  --dry-run          试运行 (不覆盖文件)"
            echo "  --status           显示当前状态"
            echo "  --rollback         回滚到上一代"
            echo "  --check-deps       只检查并安装依赖"
            echo "  --mode <MODE>      设置目标脚本运行模式 (quick/full/check)"
            echo "  --help             显示帮助"
            echo ""
            echo "环境变量:"
            echo "  EVOLUTION_RUN_MODE  目标脚本运行模式 (默认: quick)"
            echo ""
            echo "示例:"
            echo "  bash evolution_runner.sh --once --mode quick"
            echo "  EVOLUTION_RUN_MODE=full bash evolution_runner.sh"
            echo ""
            ;;
        *)
            init_evolution_system
            evolution_loop false false
            ;;
    esac
}

main "$@"