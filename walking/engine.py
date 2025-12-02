"""
1. 张量缓冲区预分配和重用
2. 批量推理优化
3. 减少数据复制
4. 支持动态批处理大小
"""

import json
import traceback
from typing import List, Tuple, Optional, Any

import torch
import numpy as np
from torch import Tensor
from torch.distributions import Normal, Categorical


class WalkingEngine:
    """
    Walking AI 推理引擎
    
    优化版本支持:
    - 张量缓冲区重用
    - 高效批量推理
    - 可配置的探索策略
    """
    
    def __init__(
        self,
        brain,
        dqn,
        is_oracle: bool,
        version: int,
        device: Optional[torch.device] = None,
        stochastic_latent: bool = False,
        enable_amp: bool = False,
        enable_quick_eval: bool = True,
        enable_rule_based_agari_guard: bool = False,
        name: str = 'NoName',
        boltzmann_epsilon: float = 0,
        boltzmann_temp: float = 1,
        top_p: float = 1,
        # [新增] 缓冲区配置
        max_batch_size: int = 256,
        prealloc_buffers: bool = True,
    ):
        self.engine_type = 'walking'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p
        
        # [优化] 缓冲区预分配
        self.max_batch_size = max_batch_size
        self.prealloc_buffers = prealloc_buffers
        self._init_buffers()

    def _init_buffers(self):
        """初始化预分配的张量缓冲区"""
        if not self.prealloc_buffers:
            self._obs_buffer = None
            self._masks_buffer = None
            self._invisible_obs_buffer = None
            self._current_batch_size = 0
            return
        
        # 获取观察空间形状
        from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE
        
        obs_shape_val = obs_shape(self.version)
        
        # 预分配缓冲区
        self._obs_buffer = torch.empty(
            (self.max_batch_size, *obs_shape_val),
            dtype=torch.float32,
            device=self.device
        )
        self._masks_buffer = torch.empty(
            (self.max_batch_size, ACTION_SPACE),
            dtype=torch.bool,
            device=self.device
        )
        
        if self.is_oracle:
            oracle_shape = oracle_obs_shape(self.version)
            self._invisible_obs_buffer = torch.empty(
                (self.max_batch_size, *oracle_shape),
                dtype=torch.float32,
                device=self.device
            )
        else:
            self._invisible_obs_buffer = None
        
        self._current_batch_size = 0

    def _ensure_buffer_size(self, batch_size: int):
        """确保缓冲区足够大"""
        if batch_size <= self._current_batch_size:
            return
        
        if batch_size > self.max_batch_size:
            # 动态扩展缓冲区
            self.max_batch_size = batch_size * 2
            self._init_buffers()
        
        self._current_batch_size = batch_size

    def react_batch(
        self,
        obs: List[np.ndarray],
        masks: List[np.ndarray],
        invisible_obs: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[int], List[List[float]], List[List[bool]], List[bool]]:
        """
        批量决策接口
        
        Args:
            obs: 观察列表
            masks: 动作掩码列表
            invisible_obs: 隐藏信息列表 (oracle 模式)
            
        Returns:
            actions: 选择的动作
            q_values: Q值分布
            masks_out: 动作掩码
            is_greedy: 是否贪婪选择
        """
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch_optimized(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch_optimized(
        self,
        obs: List[np.ndarray],
        masks: List[np.ndarray],
        invisible_obs: Optional[List[np.ndarray]]
    ) -> Tuple[List[int], List[List[float]], List[List[bool]], List[bool]]:
        """
        [优化] 使用预分配缓冲区的批量推理
        """
        batch_size = len(obs)
        
        if self.prealloc_buffers:
            # 使用预分配缓冲区
            self._ensure_buffer_size(batch_size)
            
            # [优化] 批量复制数据到缓冲区
            obs_tensor = self._fill_obs_buffer(obs, batch_size)
            masks_tensor = self._fill_masks_buffer(masks, batch_size)
            
            if invisible_obs is not None:
                invisible_obs_tensor = self._fill_invisible_obs_buffer(invisible_obs, batch_size)
            else:
                invisible_obs_tensor = None
        else:
            # 原始方式: 每次创建新张量
            obs_tensor = torch.as_tensor(
                np.stack(obs, axis=0), 
                device=self.device
            )
            masks_tensor = torch.as_tensor(
                np.stack(masks, axis=0), 
                device=self.device
            )
            if invisible_obs is not None:
                invisible_obs_tensor = torch.as_tensor(
                    np.stack(invisible_obs, axis=0), 
                    device=self.device
                )
            else:
                invisible_obs_tensor = None

        # 模型推理
        match self.version:
            case 1:
                mu, logsig = self.brain(obs_tensor, invisible_obs_tensor)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks_tensor)
            case 2 | 3 | 4:
                phi = self.brain(obs_tensor)
                q_out = self.dqn(phi, masks_tensor)

        # 动作选择
        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full(
                (batch_size,), 
                1 - self.boltzmann_epsilon, 
                device=self.device
            ).bernoulli().to(torch.bool)
            
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks_tensor, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return (
            actions.tolist(),
            q_out.tolist(),
            masks_tensor.tolist(),
            is_greedy.tolist()
        )

    def _fill_obs_buffer(self, obs: List[np.ndarray], batch_size: int) -> Tensor:
        """[优化] 高效填充观察缓冲区"""
        buffer = self._obs_buffer[:batch_size]
        
        # 批量转换并复制
        stacked = np.stack(obs, axis=0)
        buffer.copy_(torch.from_numpy(stacked))
        
        return buffer

    def _fill_masks_buffer(self, masks: List[np.ndarray], batch_size: int) -> Tensor:
        """[优化] 高效填充掩码缓冲区"""
        buffer = self._masks_buffer[:batch_size]
        
        stacked = np.stack(masks, axis=0)
        buffer.copy_(torch.from_numpy(stacked))
        
        return buffer

    def _fill_invisible_obs_buffer(
        self, 
        invisible_obs: List[np.ndarray], 
        batch_size: int
    ) -> Tensor:
        """[优化] 高效填充隐藏观察缓冲区"""
        buffer = self._invisible_obs_buffer[:batch_size]
        
        stacked = np.stack(invisible_obs, axis=0)
        buffer.copy_(torch.from_numpy(stacked))
        
        return buffer

    def get_state_dict(self) -> dict:
        """获取引擎状态"""
        return {
            'brain': self.brain.state_dict(),
            'dqn': self.dqn.state_dict(),
            'version': self.version,
            'is_oracle': self.is_oracle,
            'name': self.name,
        }

    def load_state_dict(self, state: dict):
        """加载引擎状态"""
        self.brain.load_state_dict(state['brain'])
        self.dqn.load_state_dict(state['dqn'])


def sample_top_p(logits: Tensor, p: float) -> Tensor:
    """
    Top-p (nucleus) 采样
    
    Args:
        logits: 动作 logits
        p: 累积概率阈值
        
    Returns:
        采样的动作
    """
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    
    return sampled


class ExampleMjaiLogEngine:
    """
    示例 MJAI 日志引擎
    
    用于从日志回放动作
    """
    
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids: Optional[List[int]] = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states: List[Any]) -> List[str]:
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        
        return res

    def start_game(self, game_idx: int):
        pass
    
    def end_kyoku(self, game_idx: int):
        pass
    
    def end_game(self, game_idx: int, scores: List[int]):
        pass


class EfficientBatchInference:
    """
    [新增] 高效批量推理包装器
    
    支持:
    - 自动批次聚合
    - 异步推理队列
    - 延迟执行
    """
    
    def __init__(
        self,
        engine: WalkingEngine,
        max_batch_size: int = 64,
        max_wait_ms: float = 10.0,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self._pending_requests = []
        self._request_lock = None  # threading.Lock() if needed

    def submit(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        invisible_obs: Optional[np.ndarray] = None
    ) -> 'InferenceRequest':
        """
        提交推理请求
        
        Returns:
            InferenceRequest 对象，可用于获取结果
        """
        request = InferenceRequest()
        self._pending_requests.append({
            'obs': obs,
            'masks': masks,
            'invisible_obs': invisible_obs,
            'request': request,
        })
        
        # 检查是否需要执行批处理
        if len(self._pending_requests) >= self.max_batch_size:
            self._execute_batch()
        
        return request

    def _execute_batch(self):
        """执行批量推理"""
        if not self._pending_requests:
            return
        
        obs_list = [r['obs'] for r in self._pending_requests]
        masks_list = [r['masks'] for r in self._pending_requests]
        invisible_obs_list = [r['invisible_obs'] for r in self._pending_requests]
        
        # 过滤 None
        if all(io is None for io in invisible_obs_list):
            invisible_obs_list = None
        
        # 执行推理
        actions, q_values, _, is_greedy = self.engine.react_batch(
            obs_list, masks_list, invisible_obs_list
        )
        
        # 分发结果
        for i, req_data in enumerate(self._pending_requests):
            req_data['request'].set_result(
                action=actions[i],
                q_values=q_values[i],
                is_greedy=is_greedy[i]
            )
        
        self._pending_requests.clear()

    def flush(self):
        """强制执行所有待处理的请求"""
        self._execute_batch()


class InferenceRequest:
    """推理请求句柄"""
    
    def __init__(self):
        self._result = None
        self._ready = False

    def set_result(self, action: int, q_values: List[float], is_greedy: bool):
        self._result = {
            'action': action,
            'q_values': q_values,
            'is_greedy': is_greedy,
        }
        self._ready = True

    def wait(self, timeout: float = None) -> dict:
        """等待结果 (同步模式)"""
        # 在当前简单实现中，结果应该已经设置
        if not self._ready:
            raise RuntimeError("Result not ready")
        return self._result

    @property
    def ready(self) -> bool:
        return self._ready
