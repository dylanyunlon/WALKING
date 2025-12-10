#!/bin/bash
#
# Walking 自进化系统 - 安装部署脚本
#
# 用法:
#   bash install_evolution.sh

set -e

# =============================================================================
# 配置
# =============================================================================
PROJECT_DIR="/root/dylan/icml2026/WALKING"
EVOLUTION_DIR="${PROJECT_DIR}/evolution"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "     Walking 自进化系统安装"
echo "======================================================================"
echo ""
echo "源目录: ${SOURCE_DIR}"
echo "目标目录: ${EVOLUTION_DIR}"
echo ""

# =============================================================================
# 创建目录
# =============================================================================
echo "[1/5] 创建目录结构..."
mkdir -p ${EVOLUTION_DIR}
mkdir -p ${EVOLUTION_DIR}/history

# =============================================================================
# 复制核心文件
# =============================================================================
echo "[2/5] 复制核心文件..."

# 主循环控制器
cp ${SOURCE_DIR}/evolution_runner.sh ${EVOLUTION_DIR}/
chmod +x ${EVOLUTION_DIR}/evolution_runner.sh

# 进化客户端
cp ${SOURCE_DIR}/evolution_client.py ${EVOLUTION_DIR}/

# 调试写入器
cp ${SOURCE_DIR}/debug_writer.py ${EVOLUTION_DIR}/

echo "    ✓ evolution_runner.sh"
echo "    ✓ evolution_client.py"
echo "    ✓ debug_writer.py"

# =============================================================================
# 备份原始脚本
# =============================================================================
echo "[3/5] 备份原始脚本..."

ORIGINAL_SCRIPT="${PROJECT_DIR}/walking_vs_mortal.sh"
if [ -f "${ORIGINAL_SCRIPT}" ]; then
    cp ${ORIGINAL_SCRIPT} ${EVOLUTION_DIR}/history/original_backup.sh
    cp ${ORIGINAL_SCRIPT} ${EVOLUTION_DIR}/history/generation_0.sh
    echo "    ✓ 备份 walking_vs_mortal.sh"
fi

# =============================================================================
# 安装进化版脚本 (可选)
# =============================================================================
echo "[4/5] 安装进化版脚本..."

EVOLVED_SCRIPT="${SOURCE_DIR}/walking_vs_mortal_evolved.sh"
if [ -f "${EVOLVED_SCRIPT}" ]; then
    read -p "是否用进化版脚本替换原始脚本? (y/N): " REPLACE
    if [ "${REPLACE}" = "y" ] || [ "${REPLACE}" = "Y" ]; then
        cp ${EVOLVED_SCRIPT} ${ORIGINAL_SCRIPT}
        chmod +x ${ORIGINAL_SCRIPT}
        echo "    ✓ 已替换 walking_vs_mortal.sh"
    else
        echo "    - 跳过替换 (保留原始脚本)"
    fi
fi

# =============================================================================
# 初始化调试信息
# =============================================================================
echo "[5/5] 初始化调试信息..."

DEBUG_FILE="${EVOLUTION_DIR}/debug_info.json"
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
    echo "    ✓ 创建 debug_info.json"
fi

# =============================================================================
# 创建便捷脚本
# =============================================================================
echo ""
echo "创建便捷启动脚本..."

# 创建启动脚本
cat > ${PROJECT_DIR}/start_evolution.sh << 'STARTER'
#!/bin/bash
# Walking 进化系统快速启动
cd /root/dylan/icml2026/WALKING/evolution
bash evolution_runner.sh "$@"
STARTER
chmod +x ${PROJECT_DIR}/start_evolution.sh

echo "    ✓ 创建 start_evolution.sh"

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "======================================================================"
echo "     安装完成!"
echo "======================================================================"
echo ""
echo "文件结构:"
echo "  ${EVOLUTION_DIR}/"
echo "  ├── evolution_runner.sh    # 主循环控制器"
echo "  ├── evolution_client.py    # LLM API 客户端"
echo "  ├── debug_writer.py        # 调试信息写入"
echo "  ├── debug_info.json        # 调试信息文件"
echo "  └── history/               # 历史版本存档"
echo ""
echo "使用方法:"
echo ""
echo "  # 启动进化循环 (持续运行)"
echo "  bash ${PROJECT_DIR}/start_evolution.sh"
echo ""
echo "  # 只运行一次"
echo "  bash ${PROJECT_DIR}/start_evolution.sh --once"
echo ""
echo "  # 试运行 (不覆盖文件)"
echo "  bash ${PROJECT_DIR}/start_evolution.sh --dry-run"
echo ""
echo "  # 查看状态"
echo "  bash ${PROJECT_DIR}/start_evolution.sh --status"
echo ""
echo "  # 回滚到上一代"
echo "  bash ${PROJECT_DIR}/start_evolution.sh --rollback"
echo ""
