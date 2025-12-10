#!/usr/bin/env python3
"""
Walking 进化客户端 - 调用 LLM API 获取代码改进

核心思想:
- 将当前脚本 + 调试信息 + 运行日志 + 项目结构 发送给 LLM
- LLM 返回改进后的脚本或多个文件修改
- 提取代码块并应用修改

改进历史:
- v1: 基础版本，基于 enhanced_client_example.py
- v2: 添加项目结构信息 (tree)，包含错误文件路径
- v3: 动态路径检测，支持多文件修改，移除硬编码路径
"""

import asyncio
import json
import ssl
import re
import os
import subprocess
import aiohttp
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path


class EvolutionClient:
    """进化客户端 - 调用 LLM 获取代码改进"""
    
    def __init__(self, base_url: str = None, project_dir: str = None):
        """
        初始化进化客户端
        
        Args:
            base_url: API 基础 URL (默认从环境变量或配置获取)
            project_dir: 项目目录 (默认自动检测)
        """
        # API 配置 - 优先使用环境变量
        self.base_url = base_url or os.environ.get(
            "EVOLUTION_API_URL", 
            "https://balloonet.tech:17432"
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.token: Optional[str] = None
        self.v2_endpoint = "/api/chat/v2"
        
        # 项目路径 - 动态检测
        self.project_dir = Path(project_dir) if project_dir else self._detect_project_dir()
        self.evolution_dir = self.project_dir / "evolution"
        
        # 目标脚本 - 可配置
        self.target_script = os.environ.get(
            "EVOLUTION_TARGET_SCRIPT",
            str(self.project_dir / "walking_vs_mortal.sh")
        )
    
    def _detect_project_dir(self) -> Path:
        """
        自动检测项目目录
        
        检测顺序:
        1. WALKING_PROJECT_DIR 环境变量
        2. 当前脚本所在目录的父目录
        3. 当前工作目录向上查找包含特征文件的目录
        4. 默认路径
        """
        # 1. 环境变量
        env_dir = os.environ.get("WALKING_PROJECT_DIR")
        if env_dir and os.path.isdir(env_dir):
            return Path(env_dir)
        
        # 2. 脚本所在目录
        script_dir = Path(__file__).resolve().parent
        if (script_dir.parent / "walking_vs_mortal.sh").exists():
            return script_dir.parent
        if (script_dir / "walking_vs_mortal.sh").exists():
            return script_dir
        
        # 3. 向上查找特征文件
        feature_files = ["walking_vs_mortal.sh", "Cargo.toml", "walking"]
        current = Path.cwd()
        for _ in range(5):  # 最多向上5层
            for feature in feature_files:
                if (current / feature).exists():
                    return current
            if current.parent == current:
                break
            current = current.parent
        
        # 4. 尝试常见路径
        common_paths = [
            Path("/root/dylan/icml2026/WALKING"),
            Path.home() / "WALKING",
            Path("/workspace/WALKING"),
        ]
        for p in common_paths:
            if p.exists():
                return p
        
        # 5. 最终回退到当前目录
        return Path.cwd()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,
            limit_per_host=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=300,
            connect=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Walking-Evolution-Client/3.0'}
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    def get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    async def login(self, username: str = None, password: str = None) -> bool:
        """
        登录获取 token
        
        Args:
            username: 用户名 (默认从环境变量获取)
            password: 密码 (默认从环境变量获取)
        """
        username = username or os.environ.get("EVOLUTION_API_USER", "newuser")
        password = password or os.environ.get("EVOLUTION_API_PASS", "newPass123")
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.token = data.get("access_token") or data.get("token")
                    print(f"✅ 登录成功 (项目目录: {self.project_dir})")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 登录失败: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ 登录异常: {e}")
            return False
    
    # =========================================================================
    # 项目结构收集 (使用 tree)
    # =========================================================================
    
    def get_project_tree(self, max_depth: int = 2, focus_dirs: List[str] = None) -> str:
        """
        获取项目结构树
        
        Args:
            max_depth: 最大深度
            focus_dirs: 重点关注的目录列表
        """
        result_parts = []
        
        # 检查 tree 命令是否可用
        tree_available = subprocess.run(
            ["which", "tree"], capture_output=True
        ).returncode == 0
        
        if not tree_available:
            # 回退到 find + 格式化
            return self._get_project_tree_fallback(max_depth, focus_dirs)
        
        # 1. 项目根目录浅层结构
        try:
            root_tree = subprocess.run(
                ["tree", "-L", str(max_depth), "--dirsfirst", "-I", 
                 "__pycache__|*.pyc|.git|node_modules|*.egg-info|target"],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=10
            )
            if root_tree.returncode == 0:
                result_parts.append(f"## 项目根目录: `{self.project_dir}`")
                result_parts.append("```")
                result_parts.append(root_tree.stdout[:2000])
                result_parts.append("```")
        except Exception as e:
            result_parts.append(f"## 项目结构 (获取失败: {e})")
        
        # 2. 重点目录详细结构
        if focus_dirs:
            for focus_dir in focus_dirs:
                focus_path = self.project_dir / focus_dir
                if focus_path.exists() and focus_path.is_dir():
                    try:
                        focus_tree = subprocess.run(
                            ["tree", "-L", "3", "--dirsfirst", "-I",
                             "__pycache__|*.pyc|.git|target"],
                            cwd=str(focus_path),
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if focus_tree.returncode == 0 and focus_tree.stdout.strip():
                            result_parts.append(f"\n## 重点目录: `{focus_path}`")
                            result_parts.append("```")
                            result_parts.append(focus_tree.stdout[:1500])
                            result_parts.append("```")
                    except Exception:
                        pass
        
        return "\n".join(result_parts)
    
    def _get_project_tree_fallback(self, max_depth: int, focus_dirs: List[str]) -> str:
        """当 tree 不可用时的回退方案"""
        result_parts = [f"## 项目目录: `{self.project_dir}`"]
        result_parts.append("```")
        
        try:
            for root, dirs, files in os.walk(self.project_dir):
                # 计算深度
                depth = root.replace(str(self.project_dir), '').count(os.sep)
                if depth >= max_depth:
                    dirs[:] = []  # 不再递归
                    continue
                
                # 过滤隐藏目录
                dirs[:] = [d for d in dirs if not d.startswith('.') 
                          and d not in ['__pycache__', 'node_modules', 'target']]
                
                indent = '  ' * depth
                result_parts.append(f"{indent}{os.path.basename(root)}/")
                
                for file in files[:10]:  # 每个目录最多显示10个文件
                    if not file.startswith('.'):
                        result_parts.append(f"{indent}  {file}")
        except Exception as e:
            result_parts.append(f"(遍历失败: {e})")
        
        result_parts.append("```")
        return "\n".join(result_parts[:100])  # 限制行数
    
    def get_file_content(self, file_path: str, max_lines: int = 100) -> str:
        """
        获取文件内容
        
        Args:
            file_path: 文件路径 (支持相对路径和绝对路径)
            max_lines: 最大行数
        """
        # 解析路径
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_dir / path
        
        if not path.exists():
            return f"(文件不存在: {file_path})"
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if len(lines) > max_lines:
                head = lines[:max_lines // 2]
                tail = lines[-(max_lines // 2):]
                content = ''.join(head) + f"\n... (省略 {len(lines) - max_lines} 行) ...\n" + ''.join(tail)
            else:
                content = ''.join(lines)
            
            return content
        except Exception as e:
            return f"(读取失败: {e})"
    
    def detect_error_source_files(self, error_msg: str, last_output: str) -> List[str]:
        """从错误信息中检测相关源文件"""
        relevant_files = []
        combined_text = f"{error_msg}\n{last_output}"
        
        # 匹配各种文件路径模式
        patterns = [
            r'File ["\']?([^"\':\s]+\.(py|sh))["\']?',  # Python/Shell 文件
            r'(/[^\s:]+\.(py|sh|toml|json))',  # 绝对路径
            r'at ([^\s:]+\.(py|sh)):',  # 错误位置
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if os.path.exists(match):
                    relevant_files.append(match)
        
        # 去重并限制数量
        seen = set()
        unique_files = []
        for f in relevant_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
                if len(unique_files) >= 5:
                    break
        
        return unique_files
    
    def detect_focus_directories(self, error_msg: str, last_output: str) -> List[str]:
        """根据错误信息检测需要关注的目录"""
        focus_dirs = []
        combined_text = f"{error_msg}\n{last_output}".lower()
        
        # 目录关键词映射
        dir_keywords = {
            "walking": ["walking", "engine", "model", "brain"],
            "evolution": ["evolution", "evolve", "generation"],
            "libriichi": ["libriichi", "arena", "mjai"],
            "workdir": ["checkpoint", "workdir", "pth", "model"],
        }
        
        for dir_name, keywords in dir_keywords.items():
            if any(kw in combined_text for kw in keywords):
                if (self.project_dir / dir_name).exists():
                    focus_dirs.append(dir_name)
        
        return focus_dirs[:3] if focus_dirs else ["evolution"]
    
    # =========================================================================
    # 主要进化请求方法
    # =========================================================================
    
    async def request_evolution(
        self,
        current_script: str,
        debug_info: Dict[str, Any],
        last_output: str = "",
        last_error: str = ""
    ) -> Dict[str, Any]:
        """
        请求 LLM 进行代码进化
        
        Returns:
            {
                "success": bool,
                "new_script": str,       # 主脚本内容 (如果是单文件修改)
                "file_changes": list,    # 多文件修改列表
                "explanation": str,
                "error": str
            }
        """
        
        # 检测相关文件和目录
        error_source_files = self.detect_error_source_files(last_error, last_output)
        focus_dirs = self.detect_focus_directories(last_error, last_output)
        
        # 构建提示词
        prompt = self._build_evolution_prompt(
            current_script=current_script,
            debug_info=debug_info,
            last_output=last_output,
            last_error=last_error,
            error_source_files=error_source_files,
            focus_dirs=focus_dirs
        )
        
        payload = {
            "content": prompt,
            "model": "claude-sonnet-4-20250514-all",
            "extract_code": True,
            "auto_execute": False,
            "conversation_id": None
        }
        
        try:
            print(f"📤 发送进化请求...")
            print(f"   项目目录: {self.project_dir}")
            print(f"   目标脚本: {self.target_script}")
            print(f"   相关文件: {error_source_files}")
            print(f"   关注目录: {focus_dirs}")
            
            async with self.session.post(
                f"{self.base_url}{self.v2_endpoint}/message",
                json=payload,
                headers=self.get_headers()
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_evolution_response(result)
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"API 错误 ({response.status}): {error_text}"
                    }
                    
        except asyncio.TimeoutError:
            return {"success": False, "error": "请求超时"}
        except Exception as e:
            return {"success": False, "error": f"请求异常: {str(e)}"}
    
    def _build_evolution_prompt(
        self,
        current_script: str,
        debug_info: Dict[str, Any],
        last_output: str,
        last_error: str,
        error_source_files: List[str] = None,
        focus_dirs: List[str] = None
    ) -> str:
        """构建进化提示词"""
        
        metrics = debug_info.get("metrics", {})
        avg_rank = metrics.get("avg_rank", "N/A")
        avg_pt = metrics.get("avg_pt", "N/A")
        generation = debug_info.get("generation", 0)
        last_status = debug_info.get("last_status", "unknown")
        
        # Session 错误
        session_errors = []
        session_data = debug_info.get("session_data", {})
        if session_data.get("errors"):
            for err in session_data["errors"][-3:]:
                session_errors.append(f"- [{err.get('time', 'N/A')}] {err.get('error', 'Unknown')[:300]}")
        
        # 项目结构
        project_tree = self.get_project_tree(max_depth=2, focus_dirs=focus_dirs or [])
        
        # 错误相关文件
        error_files_content = ""
        if error_source_files:
            error_files_content = "\n## 错误相关文件\n"
            for file_path in error_source_files:
                lang = "python" if file_path.endswith('.py') else "bash"
                error_files_content += f"\n### `{file_path}`\n"
                error_files_content += f"```{lang}\n"
                error_files_content += self.get_file_content(file_path, max_lines=80)
                error_files_content += "\n```\n"
        
        prompt = f"""# Walking 麻将 AI 自进化系统

## 系统概述
这是一个自进化系统。你拥有完整的 shell 权限，可以修改项目中的任何文件。
目标是让 WALKING 麻将 AI 在与 Mortal AI 的对战中取得更好的成绩。

## 环境信息
- **项目目录**: `{self.project_dir}`
- **目标脚本**: `{self.target_script}`
- **进化目录**: `{self.evolution_dir}`

## 当前状态
- **代数 (Generation)**: {generation}
- **上次状态**: {last_status}
- **平均排名**: {avg_rank} (目标: < 2.5)
- **平均得分**: {avg_pt} pt

## Session 错误
{chr(10).join(session_errors) if session_errors else "(无)"}

## 项目结构
{project_tree}
{error_files_content}

## 当前主脚本
**路径**: `{self.target_script}`
```bash
{current_script}
```

## 运行输出 (最后部分)
```
{last_output[-3000:] if last_output else "(无)"}
```

## 运行错误
```
{last_error[:2000] if last_error else "(无)"}
```

## 你的任务

分析问题并修复。你可以：

1. **修改主脚本** (`{self.target_script}`)
2. **修改其他文件** (如 Python 文件、配置文件等)
3. **创建新文件**
4. **执行 shell 命令** (如安装依赖)

### 输出格式

对于每个需要修改的文件，使用以下格式：

```file:{self.target_script}
文件内容
```

例如修改主脚本：

```file:{self.target_script}
#!/bin/bash
# Generation: {generation + 1}
# 改进: 修复了 XXX 问题
...完整脚本内容...
```

例如修改其他文件：

```file:{self.project_dir}/walking/some_fix.py
# Python 文件内容
...
```

如果需要执行 shell 命令：

```shell
pip install some-package
```

### 重要提示
- 至少输出主脚本的完整内容
- 使用 ```file:路径``` 格式指明文件路径
- 确保所有代码可以直接运行
- 在注释中说明改进内容
"""
        return prompt
    
    def _parse_evolution_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析 LLM 返回的进化响应"""
        
        if not response.get("success"):
            return {
                "success": False,
                "error": response.get("error", "Unknown error")
            }
        
        data = response.get("data", {})
        content = data.get("content", "")
        
        # 1. 尝试解析多文件格式: ```file:/path/to/file
        file_changes = []
        file_pattern = r'```file:([^\n]+)\n(.*?)```'
        file_matches = re.findall(file_pattern, content, re.DOTALL)
        
        for file_path, file_content in file_matches:
            file_path = file_path.strip()
            file_content = file_content.strip()
            file_changes.append({
                "path": file_path,
                "content": file_content,
                "type": "modify"
            })
        
        # 2. 解析 shell 命令
        shell_commands = []
        shell_pattern = r'```shell\n(.*?)```'
        shell_matches = re.findall(shell_pattern, content, re.DOTALL)
        for cmd in shell_matches:
            shell_commands.append(cmd.strip())
        
        # 3. 如果没有找到多文件格式，回退到单文件模式
        main_script = None
        if not file_changes:
            bash_patterns = [
                r'```bash\n(.*?)```',
                r'```shell\n(#!/bin/bash.*?)```',
                r'```\n(#!/bin/bash.*?)```',
            ]
            for pattern in bash_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    main_script = max(matches, key=len).strip()
                    if not main_script.startswith("#!/bin/bash"):
                        main_script = "#!/bin/bash\n" + main_script
                    file_changes.append({
                        "path": self.target_script,
                        "content": main_script,
                        "type": "modify"
                    })
                    break
        
        if file_changes:
            # 找到主脚本
            for fc in file_changes:
                if fc["path"] == self.target_script or "walking_vs_mortal" in fc["path"]:
                    main_script = fc["content"]
                    break
            
            return {
                "success": True,
                "new_script": main_script or file_changes[0]["content"],
                "file_changes": file_changes,
                "shell_commands": shell_commands,
                "explanation": self._extract_explanation(content)
            }
        else:
            return {
                "success": False,
                "error": "无法从响应中提取代码",
                "raw_content": content[:1500]
            }
    
    def _extract_explanation(self, content: str) -> str:
        """提取改进说明"""
        lines = content.split('\n')
        explanation_lines = []
        
        keywords = ['改进', '修改', '优化', '修复', 'fix', 'improve', 'change']
        in_explanation = False
        
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                in_explanation = True
            if in_explanation:
                if line.startswith('```'):
                    break
                explanation_lines.append(line)
                if len(explanation_lines) > 10:
                    break
        
        return '\n'.join(explanation_lines) if explanation_lines else "(无说明)"
    
    # =========================================================================
    # 文件修改应用
    # =========================================================================
    
    def apply_file_changes(self, file_changes: List[Dict], dry_run: bool = False) -> List[Dict]:
        """
        应用文件修改
        
        Args:
            file_changes: 文件修改列表
            dry_run: 是否试运行 (不实际写入)
        
        Returns:
            应用结果列表
        """
        results = []
        
        for change in file_changes:
            path = change["path"]
            content = change["content"]
            
            result = {"path": path, "success": False, "message": ""}
            
            try:
                if dry_run:
                    result["success"] = True
                    result["message"] = f"[DRY-RUN] 将写入 {len(content)} 字符"
                else:
                    # 创建目录
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # 写入文件
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # 如果是 shell 脚本，添加执行权限
                    if path.endswith('.sh'):
                        os.chmod(path, 0o755)
                    
                    result["success"] = True
                    result["message"] = f"成功写入 {len(content)} 字符"
                    
            except Exception as e:
                result["message"] = f"写入失败: {e}"
            
            results.append(result)
            print(f"  {'✅' if result['success'] else '❌'} {path}: {result['message']}")
        
        return results


# =============================================================================
# 命令行接口
# =============================================================================
async def main():
    """命令行测试"""
    print("🧬 Walking 进化客户端 v3")
    print("=" * 50)
    
    async with EvolutionClient() as client:
        print(f"项目目录: {client.project_dir}")
        print(f"目标脚本: {client.target_script}")
        
        # 登录
        if not await client.login():
            print("登录失败")
            return
        
        # 测试项目结构
        print("\n📁 项目结构:")
        tree = client.get_project_tree(max_depth=2, focus_dirs=["evolution"])
        print(tree[:1000])
        
        # 测试进化
        print("\n🧬 测试进化请求...")
        test_script = "#!/bin/bash\necho 'test'"
        test_debug = {
            "generation": 0,
            "last_status": "error",
            "metrics": {},
            "session_data": {"errors": [{"error": "test error", "fatal": True}]}
        }
        
        result = await client.request_evolution(
            current_script=test_script,
            debug_info=test_debug,
            last_error="CUDA error: invalid device ordinal"
        )
        
        print(f"\n结果: {'✅ 成功' if result.get('success') else '❌ 失败'}")
        if result.get("file_changes"):
            print(f"文件修改: {len(result['file_changes'])} 个")
            for fc in result["file_changes"]:
                print(f"  - {fc['path']}")


if __name__ == "__main__":
    asyncio.run(main())