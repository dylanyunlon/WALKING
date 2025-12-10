#!/usr/bin/env python3
"""
Walking 进化客户端 - 调用 LLM API 获取代码改进

核心思想:
- 将当前脚本 + 调试信息 + 运行日志 发送给 LLM
- LLM 返回改进后的脚本
- 提取 bash 代码块并返回

基于 enhanced_client_example.py 的 API 调用方式
"""

import asyncio
import json
import ssl
import re
import aiohttp
from typing import Optional, Dict, Any
from datetime import datetime


class EvolutionClient:
    """进化客户端 - 调用 LLM 获取代码改进"""
    
    def __init__(self, base_url: str = "https://balloonet.tech:17432"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.token: Optional[str] = None
        self.v2_endpoint = "/api/chat/v2"
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 创建宽松的 SSL 上下文
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
            total=300,  # 5分钟超时，因为代码生成可能较慢
            connect=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Walking-Evolution-Client/1.0'}
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
    
    async def login(self, username: str, password: str) -> bool:
        """登录获取 token"""
        try:
            async with self.session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.token = data.get("access_token") or data.get("token")
                    print(f"✅ 登录成功")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 登录失败: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ 登录异常: {e}")
            return False
    
    async def request_evolution(
        self,
        current_script: str,
        debug_info: Dict[str, Any],
        last_output: str = "",
        last_error: str = ""
    ) -> Dict[str, Any]:
        """
        请求 LLM 进行代码进化
        
        Args:
            current_script: 当前脚本内容
            debug_info: 调试信息 (JSON)
            last_output: 上次运行的输出
            last_error: 上次运行的错误
            
        Returns:
            {
                "success": bool,
                "new_script": str,  # 新的脚本内容
                "explanation": str,  # 改进说明
                "error": str  # 错误信息 (如果失败)
            }
        """
        
        # 构建进化提示词
        prompt = self._build_evolution_prompt(
            current_script=current_script,
            debug_info=debug_info,
            last_output=last_output,
            last_error=last_error
        )
        
        payload = {
            "content": prompt,
            "model": "claude-sonnet-4-20250514-all",
            "extract_code": True,
            "auto_execute": False,
            "conversation_id": None
        }
        
        try:
            print(f"📤 发送进化请求到 LLM...")
            
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
        last_error: str
    ) -> str:
        """构建进化提示词"""
        
        # 提取关键指标
        metrics = debug_info.get("metrics", {})
        avg_rank = metrics.get("avg_rank", "N/A")
        avg_pt = metrics.get("avg_pt", "N/A")
        generation = debug_info.get("generation", 0)
        last_status = debug_info.get("last_status", "unknown")
        
        prompt = f"""# Walking 麻将 AI 自进化系统

## 当前状态
- **代数 (Generation)**: {generation}
- **上次状态**: {last_status}
- **平均排名**: {avg_rank} (目标: < 2.5，越低越好)
- **平均得分**: {avg_pt} pt

## 当前脚本 (walking_vs_mortal.sh)
```bash
{current_script}
```

## 上次运行输出 (最后部分)
```
{last_output[-3000:] if last_output else "(无输出)"}
```

## 上次运行错误
```
{last_error[:2000] if last_error else "(无错误)"}
```

## 你的任务

作为 Walking 麻将 AI 的"进化引擎"，请分析上述信息，生成改进后的脚本。

### 改进方向
1. **如果有错误**: 修复错误，确保脚本可以正常运行
2. **如果排名 > 2.5**: 分析可能的问题，调整参数或策略
3. **如果排名 < 2.3**: 考虑增加测试轮数或挑战更强对手

### 输出格式
请直接输出完整的改进后的 bash 脚本，用 ```bash 和 ``` 包裹。

**重要**: 
- 只输出一个完整的 bash 脚本
- 保持脚本的基本结构不变
- 在脚本注释中说明你做了什么改进
- 确保脚本可以直接运行

```bash
#!/bin/bash
# 在这里输出改进后的完整脚本
# 第 {generation + 1} 代 - 改进说明: ...
...
```
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
        
        # 提取 bash 代码块
        bash_pattern = r'```bash\n(.*?)```'
        matches = re.findall(bash_pattern, content, re.DOTALL)
        
        if not matches:
            # 尝试其他格式
            bash_pattern = r'```shell\n(.*?)```'
            matches = re.findall(bash_pattern, content, re.DOTALL)
        
        if not matches:
            # 尝试不带语言标识的代码块
            bash_pattern = r'```\n(#!/bin/bash.*?)```'
            matches = re.findall(bash_pattern, content, re.DOTALL)
        
        if matches:
            # 取最长的代码块 (通常是完整脚本)
            new_script = max(matches, key=len).strip()
            
            # 验证脚本以 #!/bin/bash 开头
            if not new_script.startswith("#!/bin/bash"):
                new_script = "#!/bin/bash\n" + new_script
            
            return {
                "success": True,
                "new_script": new_script,
                "explanation": self._extract_explanation(content)
            }
        else:
            return {
                "success": False,
                "error": "无法从响应中提取 bash 脚本",
                "raw_content": content[:1000]
            }
    
    def _extract_explanation(self, content: str) -> str:
        """提取改进说明"""
        # 尝试找到说明部分
        lines = content.split('\n')
        explanation_lines = []
        
        in_explanation = False
        for line in lines:
            if '改进' in line or '修改' in line or '优化' in line:
                in_explanation = True
            if in_explanation:
                if line.startswith('```'):
                    break
                explanation_lines.append(line)
                if len(explanation_lines) > 10:
                    break
        
        return '\n'.join(explanation_lines) if explanation_lines else "无说明"


# =============================================================================
# 命令行接口
# =============================================================================
async def main():
    """命令行测试"""
    import sys
    
    print("🧬 Walking 进化客户端测试")
    print("=" * 50)
    
    async with EvolutionClient() as client:
        # 登录
        if not await client.login("newuser", "newPass123"):
            print("登录失败，退出")
            return
        
        # 测试进化请求
        test_script = """#!/bin/bash
echo "Hello World"
# 这是一个测试脚本
"""
        
        test_debug = {
            "generation": 0,
            "last_status": "test",
            "metrics": {"avg_rank": 2.7, "avg_pt": -10}
        }
        
        result = await client.request_evolution(
            current_script=test_script,
            debug_info=test_debug,
            last_output="测试输出",
            last_error=""
        )
        
        print("\n结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    asyncio.run(main())
