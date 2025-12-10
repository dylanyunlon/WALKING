# Walking 自进化系统架构说明

## 核心思想



```
程序A运行 → 产生调试信息 → 发给LLM → LLM返回A' → 覆盖A → 循环
```

**关键点：**
1. **真实世界的 success/error 是最终裁判** - LLM 只是"修复建议器"
2. **训练和推理不分离** - 每次运行都在改写自己
3. **调试信息写入文件** - 下一轮用 bash 读取

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Walking 自进化系统                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────┐                                            │
│   │  evolution_runner.sh │  ← 主循环控制器 (永久运行)                  │
│   └──────────┬──────────┘                                            │
│              │                                                       │
│              ▼                                                       │
│   ┌─────────────────────┐     ┌────────────────┐                     │
│   │walking_vs_mortal.sh │ ──→│ debug_info.json │  ← Jeff Dean思想    │
│   │  (可被覆盖的程序A)   │     │ (运行状态/错误) │                      │
│   └─────────────────────┘     └───────┬────────┘                     │
│                                       │                              │
│                                       ▼                              │
│                           ┌───────────────────────┐                  │
│                           │  evolution_client.py   │                  │
│                           │  调用 LLM API          │                  │
│                           │  balloonet.tech:17432  │                  │
│                           └───────────┬───────────┘                  │
│                                       │                              │
│                                       ▼                              │
│                           ┌───────────────────────┐                  │
│                           │  LLM 返回新的 sh 代码  │                  │
│                           │  提取 bash 命令        │                  │
│                           └───────────┬───────────┘                  │
│                                       │                              │
│                                       ▼                              │
│                           ┌───────────────────────┐                  │
│                           │  覆盖 walking_vs_mortal.sh               │
│                           │  程序A → 程序A'        │                  │
│                           └───────────────────────┘                  │
│                                       │                              │
│                                       ▼                              │
│                               🔄 下一轮循环                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
/root/dylan/icml2026/WALKING/
├── evolution/                      # 进化系统目录
│   ├── evolution_runner.sh         # 主循环控制器
│   ├── evolution_client.py         # LLM API 客户端
│   ├── debug_writer.py             # 调试信息写入模块
│   ├── debug_info.json             # 调试信息文件 (Jeff Dean)
│   ├── last_output.txt             # 上次运行输出
│   ├── last_error.txt              # 上次运行错误
│   ├── new_script.sh               # LLM 生成的新脚本 (临时)
│   ├── evolution.log               # 进化日志
│   └── history/                    # 历史版本存档
│       ├── generation_0.sh         # 原始脚本
│       ├── generation_1.sh         # 第1代
│       └── ...
├── walking_vs_mortal.sh            # 目标脚本 (会被进化覆盖)
└── start_evolution.sh              # 便捷启动脚本
```

## 核心组件

### 1. evolution_runner.sh (主循环控制器)

```bash
# 工作流程:
while true; do
    # 1. 运行目标脚本，捕获输出
    STATUS=$(run_target_script)
    
    # 2. 根据状态决定是否进化
    if [ "${STATUS}" != "success" ]; then
        # 3. 调用 LLM 获取改进
        call_llm_for_evolution
        
        # 4. 应用进化 (覆盖目标脚本)
        apply_evolution
    fi
    
    # 5. 等待下一轮
    sleep ${EVOLUTION_INTERVAL}
done
```

### 2. evolution_client.py (LLM 客户端)

基于 `enhanced_client_example.py`，调用 `/api/chat/v2/message` 接口：

```python
async def request_evolution(self, current_script, debug_info, ...):
    """
    发送进化请求:
    - 当前脚本内容
    - 调试信息 (JSON)
    - 运行日志
    
    返回:
    - 新的脚本内容 (从 LLM 响应中提取 bash 代码块)
    """
```

### 3. debug_writer.py (调试信息写入)

Jeff Dean 思想的实现：

```python
class DebugWriter:
    def log_metric(name, value):  # 记录指标
    def log_error(error):         # 记录错误
    def record_battle_result(...): # 记录对战结果
    def save():                   # 写入文件
```

### 4. debug_info.json (调试信息文件)

```json
{
    "generation": 5,
    "last_run": "2025-01-15T10:30:00",
    "last_status": "needs_improvement",
    "metrics": {
        "avg_rank": 2.55,
        "avg_pt": -5.2,
        "total_games": 20000
    },
    "history": [...]
}
```

## 进化决策逻辑

```
状态判断:
- avg_rank < 2.3  → excellent (不需要进化，考虑增加难度)
- avg_rank < 2.5  → success (不需要进化)
- avg_rank < 2.7  → needs_improvement (需要进化)
- avg_rank >= 2.7 → poor (需要进化)
- 有错误          → error (需要进化修复)
```

## LLM 提示词模板

```markdown
# Walking 麻将 AI 自进化系统

## 当前状态
- Generation: {N}
- 平均排名: {avg_rank}
- 上次状态: {status}

## 当前脚本
```bash
{current_script}
```

## 上次运行日志
{last_output}

## 任务
分析问题，生成改进后的脚本...
```

## 使用方法

```bash
# 安装
bash install_evolution.sh

# 启动进化循环 (持续运行)
bash start_evolution.sh

# 只运行一次
bash start_evolution.sh --once

# 试运行 (不覆盖文件)
bash start_evolution.sh --dry-run

# 查看状态
bash start_evolution.sh --status

# 回滚到上一代
bash start_evolution.sh --rollback
```

## 安全机制

1. **语法检查**: 新脚本必须通过 `bash -n` 检查
2. **历史备份**: 每次进化前备份当前版本
3. **回滚支持**: 可以随时回滚到任意历史版本
4. **日志记录**: 所有操作都记录到日志

## 类比：考试与答疑

来自"进化思想.md"：

> 就像考试一样，学生1在考场上两个小时完全不知道答案，但是考试结束就有老师为他改卷子并答疑解惑。如果过程中学生1提出了又快又好的例子，那么不用这个老师表扬，更牛的科学家都会为他鼓掌。

- **学生 (程序A)**: 在考场 (真实环境) 运行，产生答卷 (日志)
- **老师 (LLM)**: 看答卷，告诉学生哪里错了，怎么改
- **科学家 (真实世界的 success)**: 如果学生直接答对了，老师闭嘴

**LLM 不是奖励函数的定义者，是修复建议器。真正的奖励信号是真实世界的返回。**
