#!/usr/bin/env python3
"""
Walking 调试信息写入模块

Jeff Dean 理论实现:
- 将重要的调试信息写入文件
- 下一轮用 bash 读取
- 支持从 Python 训练/对战代码中直接调用

改进历史:
- v1: 基础版本
- v2: 增强错误追踪，记录文件绝对路径和堆栈信息

用法:
    from debug_writer import DebugWriter
    
    writer = DebugWriter()
    writer.log_metric("avg_rank", 2.45)
    writer.log_error("CUDA out of memory", file_path="/path/to/engine.py")
    writer.save()
"""

import json
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


class DebugWriter:
    """调试信息写入器"""
    
    def __init__(
        self,
        debug_file: str = None,
        evolution_dir: str = None
    ):
        """
        初始化调试写入器
        
        Args:
            debug_file: 调试信息文件路径
            evolution_dir: 进化目录
        """
        # 默认路径 - 支持动态检测
        if evolution_dir is None:
            evolution_dir = os.environ.get(
                "EVOLUTION_DIR",
                self._detect_evolution_dir()
            )
        
        self.evolution_dir = Path(evolution_dir)
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        
        # 项目目录
        self.project_dir = self.evolution_dir.parent
        
        if debug_file is None:
            debug_file = self.evolution_dir / "debug_info.json"
        
        self.debug_file = Path(debug_file)
        
        # 加载现有数据或创建新的
        self.data = self._load_or_create()
        
        # 当前会话的临时数据
        self._session_metrics: Dict[str, Any] = {}
        self._session_errors: List[Dict] = []
        self._session_logs: List[Dict] = []
    
    def _detect_evolution_dir(self) -> str:
        """自动检测进化目录"""
        # 1. 环境变量
        env_project = os.environ.get("WALKING_PROJECT_DIR")
        if env_project:
            return os.path.join(env_project, "evolution")
        
        # 2. 脚本所在目录
        script_dir = Path(__file__).resolve().parent
        if script_dir.name == "evolution":
            return str(script_dir)
        if (script_dir / "evolution").exists():
            return str(script_dir / "evolution")
        if (script_dir.parent / "evolution").exists():
            return str(script_dir.parent / "evolution")
        
        # 3. 常见路径
        common = Path("/root/dylan/icml2026/WALKING/evolution")
        if common.exists():
            return str(common)
        
        # 4. 默认
        return "/root/dylan/icml2026/WALKING/evolution"
    
    def _load_or_create(self) -> Dict[str, Any]:
        """加载现有数据或创建新的"""
        if self.debug_file.exists():
            try:
                with open(self.debug_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # 创建默认结构
        return {
            "generation": 0,
            "last_run": None,
            "last_status": "init",
            "last_error": None,
            "last_exit_code": None,
            "last_duration_seconds": None,
            "metrics": {
                "avg_rank": None,
                "avg_pt": None,
                "total_games": 0
            },
            "history": [],
            "session_data": {}
        }
    
    def log_metric(self, name: str, value: Any):
        """
        记录一个指标
        
        Args:
            name: 指标名称 (如 "avg_rank", "loss", "accuracy")
            value: 指标值
        """
        self._session_metrics[name] = value
        
        # 同时更新主数据结构中的 metrics
        if name in ["avg_rank", "avg_pt", "total_games"]:
            self.data["metrics"][name] = value
    
    def log_error(
        self, 
        error: str, 
        fatal: bool = False,
        file_path: str = None,
        line_number: int = None,
        exception: Exception = None
    ):
        """
        记录错误 (增强版 - 包含文件路径信息)
        
        Args:
            error: 错误信息
            fatal: 是否是致命错误
            file_path: 错误发生的文件绝对路径
            line_number: 错误发生的行号
            exception: 异常对象 (用于自动提取堆栈信息)
        """
        timestamp = datetime.now().isoformat()
        
        error_entry = {
            "time": timestamp,
            "error": str(error),
            "fatal": fatal
        }
        
        # 添加文件路径信息
        if file_path:
            # 确保是绝对路径
            abs_path = str(Path(file_path).resolve()) if os.path.exists(file_path) else file_path
            error_entry["file_path"] = abs_path
            error_entry["file_exists"] = os.path.exists(file_path)
        
        if line_number:
            error_entry["line_number"] = line_number
        
        # 如果有异常对象，提取更多信息
        if exception:
            # 获取完整的堆栈跟踪
            tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
            error_entry["traceback"] = ''.join(tb_lines[-10:])  # 保留最后10行
            
            # 从堆栈中提取相关文件
            related_files = self._extract_files_from_traceback(tb_lines)
            if related_files:
                error_entry["related_files"] = related_files
        
        # 如果没有提供文件路径，尝试从错误消息中提取
        if not file_path:
            extracted = self._extract_file_from_error(error)
            if extracted:
                error_entry["extracted_file"] = extracted
        
        self._session_errors.append(error_entry)
        
        if fatal:
            self.data["last_error"] = error[:500]  # 限制长度
            self.data["last_status"] = "error"
    
    def _extract_files_from_traceback(self, tb_lines: List[str]) -> List[Dict[str, Any]]:
        """从堆栈跟踪中提取文件信息"""
        files = []
        seen = set()
        
        # 匹配 Python 堆栈格式: File "/path/to/file.py", line 123
        pattern = r'File "([^"]+)", line (\d+)'
        
        for line in tb_lines:
            match = re.search(pattern, line)
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))
                
                # 跳过标准库文件
                if '/usr/lib/' in file_path or '/site-packages/' in file_path:
                    continue
                
                # 去重
                if file_path in seen:
                    continue
                seen.add(file_path)
                
                files.append({
                    "path": file_path,
                    "line": line_num,
                    "exists": os.path.exists(file_path)
                })
        
        return files[:5]  # 最多返回5个文件
    
    def _extract_file_from_error(self, error: str) -> Optional[Dict[str, Any]]:
        """从错误消息中提取文件路径"""
        patterns = [
            # Python 风格: File "/path/to/file.py", line 123
            r'File "([^"]+\.py)", line (\d+)',
            # 带行号的路径: /path/to/file.py:123
            r'(/[^\s:]+\.(?:py|sh)):(\d+)',
            # 一般绝对路径
            r'(/[^\s:]+\.(?:py|sh|pth|toml|json))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                groups = match.groups()
                file_path = groups[0]
                line_num = int(groups[1]) if len(groups) > 1 and groups[1] else None
                
                result = {
                    "path": file_path,
                    "exists": os.path.exists(file_path)
                }
                
                if os.path.exists(file_path):
                    result["absolute_path"] = str(Path(file_path).resolve())
                
                if line_num:
                    result["line"] = line_num
                
                return result
        
        return None
    
    def log(self, message: str, level: str = "info"):
        """
        记录日志
        
        Args:
            message: 日志消息
            level: 日志级别 (info, warn, error, debug)
        """
        timestamp = datetime.now().isoformat()
        self._session_logs.append({
            "time": timestamp,
            "level": level,
            "message": message
        })
    
    def set_status(self, status: str):
        """
        设置运行状态
        
        Args:
            status: 状态 (success, error, needs_improvement, running)
        """
        self.data["last_status"] = status
    
    def set_exit_code(self, code: int):
        """设置退出码"""
        self.data["last_exit_code"] = code
    
    def increment_generation(self):
        """增加代数"""
        self.data["generation"] = self.data.get("generation", 0) + 1
    
    def record_battle_result(
        self,
        rankings: List[int],
        avg_rank: float,
        avg_pt: float,
        total_games: int,
        duration_seconds: float = None
    ):
        """
        记录对战结果 (专门为 walking_vs_mortal 设计)
        
        Args:
            rankings: 排名分布 [1位数, 2位数, 3位数, 4位数]
            avg_rank: 平均排名
            avg_pt: 平均得分
            total_games: 总局数
            duration_seconds: 运行时长
        """
        self.data["metrics"] = {
            "avg_rank": avg_rank,
            "avg_pt": avg_pt,
            "total_games": total_games,
            "rankings": rankings
        }
        
        if duration_seconds:
            self.data["last_duration_seconds"] = duration_seconds
        
        # 根据排名判断状态
        if avg_rank < 2.3:
            self.data["last_status"] = "excellent"
        elif avg_rank < 2.5:
            self.data["last_status"] = "success"
        elif avg_rank < 2.7:
            self.data["last_status"] = "needs_improvement"
        else:
            self.data["last_status"] = "poor"
        
        self.log(f"Battle result: rank={avg_rank:.4f}, pt={avg_pt:.2f}")
    
    def save(self):
        """保存调试信息到文件"""
        # 更新时间戳
        self.data["last_run"] = datetime.now().isoformat()
        
        # 合并会话数据
        self.data["session_data"] = {
            "metrics": self._session_metrics,
            "errors": self._session_errors[-10:],  # 保留最近10条错误
            "logs": self._session_logs[-50:]  # 保留最近50条日志
        }
        
        # 写入文件
        try:
            with open(self.debug_file, 'w') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            print(f"📝 调试信息已保存到: {self.debug_file}")
        except IOError as e:
            print(f"❌ 保存调试信息失败: {e}", file=sys.stderr)
    
    def save_output(self, output: str, filename: str = "last_output.txt"):
        """保存输出到文件"""
        output_file = self.evolution_dir / filename
        with open(output_file, 'w') as f:
            f.write(output)
    
    def save_error(self, error: str, filename: str = "last_error.txt"):
        """保存错误到文件"""
        error_file = self.evolution_dir / filename
        with open(error_file, 'w') as f:
            f.write(error)
    
    def get_summary(self) -> str:
        """获取调试信息摘要"""
        metrics = self.data.get("metrics", {})
        
        # 错误摘要
        error_summary = ""
        session_data = self.data.get("session_data", {})
        errors = session_data.get("errors", [])
        if errors:
            latest_error = errors[-1]
            file_info = latest_error.get('file_path') or \
                        latest_error.get('extracted_file', {}).get('path') or \
                        'N/A'
            error_summary = f"""
Latest Error:
  Time: {latest_error.get('time', 'N/A')}
  Message: {latest_error.get('error', 'N/A')[:100]}
  File: {file_info}
  Fatal: {latest_error.get('fatal', False)}
"""
            # 显示相关文件
            related = latest_error.get('related_files', [])
            if related:
                error_summary += "  Related Files:\n"
                for rf in related[:3]:
                    error_summary += f"    - {rf.get('path')}:{rf.get('line', '?')}\n"
        
        return f"""
Generation: {self.data.get('generation', 0)}
Status: {self.data.get('last_status', 'unknown')}
Last Run: {self.data.get('last_run', 'N/A')}
Avg Rank: {metrics.get('avg_rank', 'N/A')}
Avg Pt: {metrics.get('avg_pt', 'N/A')}
Total Games: {metrics.get('total_games', 0)}
{error_summary}
"""


class EvolutionContext:
    """
    进化上下文管理器 - 用于包装训练/对战代码
    
    用法:
        with EvolutionContext() as ctx:
            # 运行对战
            result = run_battle()
            ctx.record_battle_result(result)
            
            if error:
                ctx.log_error(error, file_path=__file__)
    """
    
    def __init__(self, evolution_dir: str = None):
        self.writer = DebugWriter(evolution_dir=evolution_dir)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.writer.set_status("running")
        self.writer.log("Evolution context started")
        return self.writer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 计算运行时长
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.writer.data["last_duration_seconds"] = duration
        
        # 如果有异常，记录错误（包含完整异常信息）
        if exc_type is not None:
            self.writer.log_error(
                f"{exc_type.__name__}: {exc_val}", 
                fatal=True,
                exception=exc_val
            )
            self.writer.set_exit_code(1)
        else:
            self.writer.set_exit_code(0)
        
        # 保存
        self.writer.save()
        
        return False  # 不抑制异常


# =============================================================================
# 便捷函数 - 可以直接在 bash 中调用
# =============================================================================
def cli():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Walking 调试信息写入工具")
    parser.add_argument("command", choices=["log", "metric", "error", "status", "show"])
    parser.add_argument("--name", "-n", help="指标/日志名称")
    parser.add_argument("--value", "-v", help="指标值")
    parser.add_argument("--message", "-m", help="消息内容")
    parser.add_argument("--file", "-f", help="相关文件路径")
    parser.add_argument("--line", "-l", type=int, help="行号")
    parser.add_argument("--fatal", action="store_true", help="标记为致命错误")
    parser.add_argument("--level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    writer = DebugWriter()
    
    if args.command == "log":
        writer.log(args.message or "", args.level)
        writer.save()
        
    elif args.command == "metric":
        if args.name and args.value:
            # 尝试转换为数字
            try:
                value = float(args.value)
            except ValueError:
                value = args.value
            writer.log_metric(args.name, value)
            writer.save()
        else:
            print("需要 --name 和 --value 参数")
            
    elif args.command == "error":
        writer.log_error(
            args.message or "Unknown error",
            fatal=args.fatal,
            file_path=args.file,
            line_number=args.line
        )
        writer.save()
        
    elif args.command == "status":
        writer.set_status(args.value or "unknown")
        writer.save()
        
    elif args.command == "show":
        print(writer.get_summary())


if __name__ == "__main__":
    cli()