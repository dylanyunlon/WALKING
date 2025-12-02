"""
1. 批量计算支持
2. 缓存机制
3. 减少重复计算
4. GPU 加速 (可选)
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
from functools import lru_cache


class RewardCalculator:
    """
    奖励计算器
    
    基于 GRP (Game Result Prediction) 模型计算预期收益变化
    
    优化版本支持:
    - 批量计算多个游戏的奖励
    - 可选的结果缓存
    - GPU 加速计算
    """
    
    def __init__(
        self,
        grp=None,
        pts: Optional[List[float]] = None,
        uniform_init: bool = False,
        device: Optional[torch.device] = None,
        enable_cache: bool = False,
        cache_size: int = 1000,
    ):
        self.device = device or torch.device('cpu')
        self.grp = grp.to(self.device).eval() if grp else None
        self.uniform_init = uniform_init
        self.enable_cache = enable_cache

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)
        
        # 缓存配置
        if enable_cache:
            self._cache = {}
            self._cache_size = cache_size
        else:
            self._cache = None

    def calc_grp(self, grp_feature: np.ndarray) -> torch.Tensor:
        """
        计算 GRP 矩阵
        
        Args:
            grp_feature: GRP 特征序列, shape (seq_len, feature_dim)
            
        Returns:
            排名概率矩阵, shape (seq_len, 4, 4)
        """
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(
        self,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray
    ) -> torch.Tensor:
        """
        计算指定玩家的排名概率序列
        
        Args:
            player_id: 玩家 ID (0-3)
            grp_feature: GRP 特征序列
            rank_by_player: 最终排名 (by player)
            
        Returns:
            排名概率序列, shape (seq_len + 1, 4)
        """
        matrix = self.calc_grp(grp_feature)

        final_ranking = torch.zeros((1, 4), device=self.device, dtype=matrix.dtype)
        final_ranking[0, rank_by_player[player_id]] = 1.0
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        
        if self.uniform_init:
            rank_prob[0, :] = 0.25
        
        return rank_prob

    def calc_delta_pt(
        self,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray
    ) -> np.ndarray:
        """
        计算预期点数变化序列
        
        Args:
            player_id: 玩家 ID
            grp_feature: GRP 特征序列
            rank_by_player: 最终排名
            
        Returns:
            每个 kyoku 的预期点数变化
        """
        # 尝试缓存
        if self.enable_cache:
            cache_key = self._make_cache_key(player_id, grp_feature, rank_by_player)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        result = reward.cpu().numpy()
        
        # 存入缓存
        if self.enable_cache:
            if len(self._cache) >= self._cache_size:
                # 简单的 LRU: 清空一半
                keys_to_remove = list(self._cache.keys())[:self._cache_size // 2]
                for k in keys_to_remove:
                    del self._cache[k]
            self._cache[cache_key] = result
        
        return result

    def calc_delta_points(
        self,
        player_id: int,
        grp_feature: np.ndarray,
        final_scores: np.ndarray
    ) -> np.ndarray:
        """
        计算实际点数变化序列
        
        Args:
            player_id: 玩家 ID
            grp_feature: GRP 特征序列
            final_scores: 最终分数
            
        Returns:
            每个 kyoku 的实际点数变化
        """
        seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
        delta_points = seq[1:] - seq[:-1]
        return delta_points

    def _make_cache_key(
        self,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray
    ) -> tuple:
        """生成缓存键"""
        return (
            player_id,
            grp_feature.tobytes(),
            rank_by_player.tobytes(),
        )


class BatchRewardCalculator:
    """
    [新增] 批量奖励计算器
    
    针对多个游戏批量计算奖励，最大化 GPU 利用率
    """
    
    def __init__(
        self,
        grp,
        pts: Optional[List[float]] = None,
        uniform_init: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.uniform_init = uniform_init
        
        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    @torch.inference_mode()
    def calc_batch_delta_pt(
        self,
        player_ids: List[int],
        grp_features: List[np.ndarray],
        rank_by_players: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        批量计算多个游戏的预期点数变化
        
        Args:
            player_ids: 每个游戏的玩家 ID 列表
            grp_features: 每个游戏的 GRP 特征序列列表
            rank_by_players: 每个游戏的最终排名列表
            
        Returns:
            每个游戏的 kyoku 奖励列表
        """
        num_games = len(player_ids)
        if num_games == 0:
            return []
        
        # 构建所有序列
        all_seqs = []
        boundaries = [0]  # 每个游戏的起始位置
        
        for grp_feature in grp_features:
            seq_len = len(grp_feature)
            for idx in range(seq_len):
                tensor = torch.as_tensor(
                    grp_feature[:idx + 1],
                    dtype=torch.float64,
                    device=self.device
                )
                all_seqs.append(tensor)
            boundaries.append(len(all_seqs))
        
        if not all_seqs:
            return [np.array([]) for _ in range(num_games)]
        
        # 批量推理
        all_logits = self.grp(all_seqs)
        all_matrices = self.grp.calc_matrix(all_logits)
        
        # 分割结果并计算每个游戏的奖励
        results = []
        for i in range(num_games):
            start, end = boundaries[i], boundaries[i + 1]
            
            if start == end:
                # 空游戏
                results.append(np.array([]))
                continue
            
            matrices = all_matrices[start:end]
            player_id = player_ids[i]
            rank_by_player = rank_by_players[i]
            
            # 计算排名概率序列
            final_ranking = torch.zeros((1, 4), device=self.device, dtype=matrices.dtype)
            final_ranking[0, rank_by_player[player_id]] = 1.0
            rank_prob = torch.cat((matrices[:, player_id], final_ranking))
            
            if self.uniform_init:
                rank_prob[0, :] = 0.25
            
            # 计算期望收益变化
            exp_pts = rank_prob @ self.pts
            reward = exp_pts[1:] - exp_pts[:-1]
            
            results.append(reward.cpu().numpy())
        
        return results

    @torch.inference_mode()
    def calc_batch_grp_matrices(
        self,
        grp_features: List[np.ndarray]
    ) -> List[torch.Tensor]:
        """
        批量计算 GRP 矩阵
        
        Args:
            grp_features: GRP 特征序列列表
            
        Returns:
            每个游戏的 GRP 矩阵序列
        """
        # 构建所有序列
        all_seqs = []
        boundaries = [0]
        
        for grp_feature in grp_features:
            for idx in range(len(grp_feature)):
                tensor = torch.as_tensor(
                    grp_feature[:idx + 1],
                    dtype=torch.float64,
                    device=self.device
                )
                all_seqs.append(tensor)
            boundaries.append(len(all_seqs))
        
        if not all_seqs:
            return []
        
        # 批量推理
        all_logits = self.grp(all_seqs)
        all_matrices = self.grp.calc_matrix(all_logits)
        
        # 分割结果
        results = []
        for i in range(len(grp_features)):
            start, end = boundaries[i], boundaries[i + 1]
            results.append(all_matrices[start:end])
        
        return results


class CachedRewardCalculator:
    """
    [新增] 带 LRU 缓存的奖励计算器
    
    适用于需要重复计算相同状态的场景
    """
    
    def __init__(
        self,
        grp,
        pts: Optional[List[float]] = None,
        cache_size: int = 10000,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.cache_size = cache_size
        
        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)
        
        # 使用 LRU 缓存
        self._calc_with_cache = lru_cache(maxsize=cache_size)(self._calc_impl)

    def _make_key(
        self,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray
    ) -> Tuple:
        """创建可哈希的缓存键"""
        return (
            player_id,
            tuple(map(tuple, grp_feature)),
            tuple(rank_by_player),
        )

    def _calc_impl(self, key: Tuple) -> Tuple[float, ...]:
        """实际计算逻辑 (结果会被缓存)"""
        player_id, grp_feature_tuple, rank_tuple = key
        
        grp_feature = np.array(grp_feature_tuple)
        rank_by_player = np.array(rank_tuple)
        
        # 构建序列
        seq = [
            torch.as_tensor(grp_feature[:idx+1], device=self.device)
            for idx in range(len(grp_feature))
        ]
        
        with torch.inference_mode():
            logits = self.grp(seq)
            matrix = self.grp.calc_matrix(logits)
        
        # 计算最终排名概率
        final_ranking = torch.zeros((1, 4), device=self.device, dtype=matrix.dtype)
        final_ranking[0, rank_by_player[player_id]] = 1.0
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        
        # 计算奖励
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        
        return tuple(reward.cpu().numpy().tolist())

    def calc_delta_pt(
        self,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray
    ) -> np.ndarray:
        """计算预期点数变化 (带缓存)"""
        key = self._make_key(player_id, grp_feature, rank_by_player)
        result_tuple = self._calc_with_cache(key)
        return np.array(result_tuple)

    def clear_cache(self):
        """清空缓存"""
        self._calc_with_cache.cache_clear()

    @property
    def cache_info(self):
        """获取缓存统计信息"""
        return self._calc_with_cache.cache_info()


def create_reward_calculator(
    grp,
    pts: Optional[List[float]] = None,
    mode: str = 'standard',
    **kwargs
) -> RewardCalculator:
    """
    工厂函数: 创建奖励计算器
    
    Args:
        grp: GRP 模型
        pts: 排名点数
        mode: 计算模式
            - 'standard': 标准计算器
            - 'batch': 批量计算器
            - 'cached': 带缓存的计算器
        **kwargs: 额外参数
        
    Returns:
        奖励计算器实例
    """
    match mode:
        case 'standard':
            return RewardCalculator(grp, pts, **kwargs)
        case 'batch':
            return BatchRewardCalculator(grp, pts, **kwargs)
        case 'cached':
            return CachedRewardCalculator(grp, pts, **kwargs)
        case _:
            raise ValueError(f"Unknown mode: {mode}")
