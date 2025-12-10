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
# 配置变量
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/root/dylan/icml2026/WALKING"
WALKING_DIR="${PROJECT_DIR}/walking"
CONDA_ENV="walking3"

# 进化系统配置
EVOLUTION_DIR="${PROJECT_DIR}/evolution"
DEBUG_FILE="${EVOLUTION_DIR}/debug_info.json"
HISTORY_DIR="${EVOLUTION_DIR}/history"
LOG_FILE="${EVOLUTION_DIR}/evolution.log"

# 目标脚本 (要被进化的程序A)
TARGET_SCRIPT="${PROJECT_DIR}/walking_vs_mortal.sh"

# LLM API 配置
API_BASE_URL="https://balloonet.tech:17432"
API_USERNAME="newuser"
API_PASSWORD="newPass123"

# 进化间隔 (秒)
EVOLUTION_INTERVAL=60

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
# 初始化
# =============================================================================
init_evolution_system() {
    log_step "初始化进化系统..."
    
    # 创建必要目录
    mkdir -p ${EVOLUTION_DIR}
    mkdir -p ${HISTORY_DIR}
    
    # 初始化日志
    echo "===== Evolution System Started at $(date) =====" >> ${LOG_FILE}
    
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
    
    log_info "进化系统初始化完成"
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
    log_step "运行目标脚本: ${TARGET_SCRIPT}"
    
    local OUTPUT_FILE="${EVOLUTION_DIR}/last_output.txt"
    local ERROR_FILE="${EVOLUTION_DIR}/last_error.txt"
    local START_TIME=$(date +%s)
    
    # 运行脚本，捕获输出和错误
    set +e  # 暂时允许错误
    bash ${TARGET_SCRIPT} quick 2>${ERROR_FILE} | tee ${OUTPUT_FILE}
    local EXIT_CODE=$?
    set -e
    
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    
    # 更新调试信息
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
    
    # 确定状态
    local STATUS="success"
    if [ ${EXIT_CODE} -ne 0 ]; then
        STATUS="error"
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
    
    # 调用 Python 客户端
    cd ${EVOLUTION_DIR}
    
    python3 << PYTHON_EVOLUTION
import asyncio
import json
import sys
sys.path.insert(0, '${EVOLUTION_DIR}')

from evolution_client import EvolutionClient

async def get_evolution():
    async with EvolutionClient() as client:
        # 登录
        await client.login("${API_USERNAME}", "${API_PASSWORD}")
        
        # 读取文件
        with open("${TARGET_SCRIPT}", "r") as f:
            current_script = f.read()
        
        with open("${DEBUG_FILE}", "r") as f:
            debug_info = json.load(f)
        
        last_output = ""
        try:
            with open("${EVOLUTION_DIR}/last_output.txt", "r") as f:
                last_output = f.read()[-5000:]  # 最后5000字符
        except:
            pass
        
        last_error = ""
        try:
            with open("${EVOLUTION_DIR}/last_error.txt", "r") as f:
                last_error = f.read()[:2000]  # 前2000字符
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
            # 保存新脚本
            new_script = result["new_script"]
            with open("${EVOLUTION_DIR}/new_script.sh", "w") as f:
                f.write(new_script)
            print("SUCCESS")
            print(f"改进说明: {result.get('explanation', 'N/A')[:200]}")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

asyncio.run(get_evolution())
PYTHON_EVOLUTION
}

# =============================================================================
# 应用进化 (覆盖目标脚本)
# =============================================================================
apply_evolution() {
    log_step "应用进化..."
    
    local NEW_SCRIPT="${EVOLUTION_DIR}/new_script.sh"
    
    if [ ! -f "${NEW_SCRIPT}" ]; then
        log_error "新脚本不存在"
        return 1
    fi
    
    # 验证新脚本语法
    if ! bash -n ${NEW_SCRIPT} 2>/dev/null; then
        log_error "新脚本语法错误，拒绝应用"
        return 1
    fi
    
    # 获取当前代数
    local GENERATION=$(jq -r '.generation' ${DEBUG_FILE})
    
    # 备份当前版本到历史
    cp ${TARGET_SCRIPT} ${HISTORY_DIR}/generation_${GENERATION}.sh
    log_info "备份当前版本为 generation_${GENERATION}.sh"
    
    # 应用新脚本
    cp ${NEW_SCRIPT} ${TARGET_SCRIPT}
    chmod +x ${TARGET_SCRIPT}
    
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
    log_evolution "     Walking 自进化系统启动"
    log_evolution "=========================================="
    log_info "目标脚本: ${TARGET_SCRIPT}"
    log_info "进化间隔: ${EVOLUTION_INTERVAL}秒"
    log_info "运行模式: $([ "${RUN_ONCE}" = "true" ] && echo "单次" || echo "循环")"
    log_info "试运行: $([ "${DRY_RUN}" = "true" ] && echo "是" || echo "否")"
    
    activate_env
    
    while true; do
        echo ""
        log_evolution ">>> 开始新一轮进化 <<<"
        
        # 1. 运行目标脚本
        local STATUS=$(run_target_script)
        
        # 2. 根据状态决定是否需要进化
        if [ "${STATUS}" = "success" ]; then
            log_info "运行成功且指标良好，暂不需要进化"
            
            # 检查是否可以挑战更难的任务
            local AVG_RANK=$(jq -r '.metrics.avg_rank' ${DEBUG_FILE})
            if [ "${AVG_RANK}" != "null" ] && [ $(echo "${AVG_RANK} < 2.3" | bc -l) -eq 1 ]; then
                log_evolution "表现优异! 考虑增加难度..."
                # 这里可以触发"增加难度"的进化
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
            log_info "单次运行完成，退出"
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
    case "${1:-}" in
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
        --help|-h)
            echo ""
            echo "Walking 自进化系统"
            echo ""
            echo "用法: bash evolution_runner.sh [选项]"
            echo ""
            echo "选项:"
            echo "  (无参数)    启动进化循环 (持续运行)"
            echo "  --once      只运行一次进化"
            echo "  --dry-run   试运行 (不覆盖文件)"
            echo "  --status    显示当前状态"
            echo "  --rollback  回滚到上一代"
            echo "  --help      显示帮助"
            echo ""
            ;;
        *)
            init_evolution_system
            evolution_loop false false
            ;;
    esac
}

main "$@"
