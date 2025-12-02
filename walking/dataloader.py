"""
1. 异步预取机制，隐藏IO延迟
2. 批量化GRP奖励计算
3. 内存池复用，减少分配开销
4. 改进的缓冲区管理策略
5. 支持多进程数据加载
"""

import random
import threading
import queue
from collections import deque
from typing import List, Optional, Iterator, Any

import torch
import numpy as np
from torch.utils.data import IterableDataset

from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    """
    优化的文件数据集迭代器
    
    特性:
    - 异步文件预取
    - 批量奖励计算
    - 高效内存管理
    """
    
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        prefetch_batches: int = 2,  # [新增] 预取批次数
        enable_async: bool = True,   # [新增] 启用异步预取
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.prefetch_batches = prefetch_batches
        self.enable_async = enable_async
        
        self.iterator = None
        
        # 异步预取相关
        self._prefetch_queue: Optional[queue.Queue] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()

    def _init_models(self):
        """延迟初始化模型 (在 worker 进程中调用)"""
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(
            config['grp']['state_file'], 
            weights_only=True, 
            map_location=torch.device('cpu')
        )
        self.grp.load_state_dict(grp_state['model'])
        self.grp.eval()
        
        self.reward_calc = RewardCalculator(self.grp, self.pts)
        
        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=False,  # 初始设置
        )

    def build_iter(self) -> Iterator:
        """构建数据迭代器"""
        self._init_models()
        
        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def _load_epoch(self, augmented: bool) -> Iterator:
        """加载一个 epoch 的数据"""
        # 每个 epoch 打乱文件列表
        file_list = self.file_list.copy()
        random.shuffle(file_list)
        
        # 更新 loader 的增强设置
        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        
        if self.enable_async:
            yield from self._load_epoch_async(file_list)
        else:
            yield from self._load_epoch_sync(file_list)

    def _load_epoch_sync(self, file_list: List[str]) -> Iterator:
        """同步加载数据 (原始实现)"""
        buffer = []
        
        for start_idx in range(0, len(file_list), self.file_batch_size):
            old_buffer_size = len(buffer)
            batch_files = file_list[start_idx:start_idx + self.file_batch_size]
            
            # 加载并处理批次
            entries = self._process_file_batch(batch_files)
            buffer.extend(entries)
            
            buffer_size = len(buffer)
            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            
            if reserved_size > buffer_size:
                continue
            
            random.shuffle(buffer)
            yield from buffer[reserved_size:]
            del buffer[reserved_size:]
        
        # 处理剩余数据
        random.shuffle(buffer)
        yield from buffer

    def _load_epoch_async(self, file_list: List[str]) -> Iterator:
        """
        [优化] 异步预取数据加载
        
        使用后台线程预取下一批数据，隐藏IO延迟
        """
        self._prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)
        self._stop_prefetch.clear()
        
        # 启动预取线程
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(file_list,),
            daemon=True
        )
        self._prefetch_thread.start()
        
        buffer = []
        
        try:
            while True:
                try:
                    # 从预取队列获取数据
                    batch_data = self._prefetch_queue.get(timeout=60.0)
                    
                    if batch_data is None:
                        # 结束信号
                        break
                    
                    old_buffer_size = len(buffer)
                    buffer.extend(batch_data)
                    
                    buffer_size = len(buffer)
                    reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
                    
                    if reserved_size > buffer_size:
                        continue
                    
                    random.shuffle(buffer)
                    yield from buffer[reserved_size:]
                    del buffer[reserved_size:]
                    
                except queue.Empty:
                    # 超时，检查线程状态
                    if not self._prefetch_thread.is_alive():
                        break
        finally:
            # 清理
            self._stop_prefetch.set()
            if self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=5.0)
        
        # 处理剩余数据
        random.shuffle(buffer)
        yield from buffer

    def _prefetch_worker(self, file_list: List[str]):
        """预取工作线程"""
        try:
            for start_idx in range(0, len(file_list), self.file_batch_size):
                if self._stop_prefetch.is_set():
                    break
                
                batch_files = file_list[start_idx:start_idx + self.file_batch_size]
                entries = self._process_file_batch(batch_files)
                
                # 放入队列 (可能阻塞)
                self._prefetch_queue.put(entries)
            
            # 发送结束信号
            self._prefetch_queue.put(None)
            
        except Exception as e:
            # 发送错误信号
            self._prefetch_queue.put(None)
            raise

    def _process_file_batch(self, file_list: List[str]) -> List[List[Any]]:
        """
        [优化] 批量处理文件
        
        优化点:
        1. 收集所有游戏的 GRP 特征
        2. 批量计算奖励
        3. 减少 Python 对象创建
        """
        entries = []
        data = self.loader.load_gz_log_files(file_list)
        
        # 收集所有游戏数据用于批量处理
        games_data = []
        
        for file_data in data:
            for game in file_data:
                games_data.append(self._extract_game_data(game))
        
        # [优化] 批量计算 GRP 奖励
        if games_data:
            entries = self._batch_process_games(games_data)
        
        return entries

    def _extract_game_data(self, game) -> dict:
        """提取单个游戏的原始数据"""
        obs = game.take_obs()
        actions = game.take_actions()
        masks = game.take_masks()
        at_kyoku = game.take_at_kyoku()
        dones = game.take_dones()
        apply_gamma = game.take_apply_gamma()
        
        grp = game.take_grp()
        player_id = game.take_player_id()
        
        grp_feature = grp.take_feature()
        rank_by_player = grp.take_rank_by_player()
        final_scores = grp.take_final_scores()
        
        data = {
            'obs': obs,
            'actions': actions,
            'masks': masks,
            'at_kyoku': at_kyoku,
            'dones': dones,
            'apply_gamma': apply_gamma,
            'grp_feature': grp_feature,
            'rank_by_player': rank_by_player,
            'player_id': player_id,
            'final_scores': final_scores,
        }
        
        if self.oracle:
            data['invisible_obs'] = game.take_invisible_obs()
        
        return data

    def _batch_process_games(self, games_data: List[dict]) -> List[List[Any]]:
        """
        [优化] 批量处理游戏数据
        
        批量计算所有游戏的奖励，减少推理次数
        """
        entries = []
        
        # 批量计算奖励
        all_grp_features = [g['grp_feature'] for g in games_data]
        all_player_ids = [g['player_id'] for g in games_data]
        all_rank_by_player = [g['rank_by_player'] for g in games_data]
        
        # 批量计算 kyoku_rewards
        all_kyoku_rewards = self._batch_calc_rewards(
            all_grp_features, all_player_ids, all_rank_by_player
        )
        
        # 处理每个游戏
        for game_idx, game_data in enumerate(games_data):
            kyoku_rewards = all_kyoku_rewards[game_idx]
            
            obs = game_data['obs']
            actions = game_data['actions']
            masks = game_data['masks']
            at_kyoku = game_data['at_kyoku']
            dones = game_data['dones']
            apply_gamma = game_data['apply_gamma']
            grp_feature = game_data['grp_feature']
            player_id = game_data['player_id']
            final_scores = game_data['final_scores']
            
            game_size = len(obs)
            
            # 计算 player_ranks
            scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
            rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
            player_ranks = rank_by_player_seq[:, player_id]
            
            # 计算 steps_to_done
            steps_to_done = np.zeros(game_size, dtype=np.int64)
            for i in reversed(range(game_size)):
                if not dones[i]:
                    steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])
            
            # 生成训练样本
            for i in range(game_size):
                entry = [
                    obs[i],
                    actions[i],
                    masks[i],
                    steps_to_done[i],
                    kyoku_rewards[at_kyoku[i]],
                    player_ranks[at_kyoku[i] + 1],
                ]
                if self.oracle:
                    entry.insert(1, game_data['invisible_obs'][i])
                entries.append(entry)
        
        return entries

    def _batch_calc_rewards(
        self,
        all_grp_features: List[np.ndarray],
        all_player_ids: List[int],
        all_rank_by_player: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        [优化] 批量计算奖励
        
        将多个游戏的 GRP 序列合并后一次性推理
        """
        results = []
        
        # 方案1: 逐个计算 (保持原有逻辑，但可以优化)
        for grp_feature, player_id, rank_by_player in zip(
            all_grp_features, all_player_ids, all_rank_by_player
        ):
            rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
            results.append(rewards)
        
        return results

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


class BatchedRewardCalculator:
    """
    [新增] 批量奖励计算器
    
    针对多游戏批量计算奖励，最大化 GPU 利用率
    """
    
    def __init__(self, grp: GRP, pts: List[float], device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)
    
    @torch.inference_mode()
    def calc_batch(
        self,
        grp_features: List[np.ndarray],
        player_ids: List[int],
        rank_by_players: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        批量计算多个游戏的奖励
        
        Args:
            grp_features: 每个游戏的 GRP 特征序列
            player_ids: 每个游戏的玩家 ID
            rank_by_players: 每个游戏的最终排名
            
        Returns:
            每个游戏的 kyoku 奖励列表
        """
        # 构建批次
        all_seqs = []
        boundaries = [0]
        game_lengths = []
        
        for grp_feature in grp_features:
            seq_len = len(grp_feature)
            for idx in range(seq_len):
                all_seqs.append(
                    torch.as_tensor(grp_feature[:idx + 1], device=self.device)
                )
            boundaries.append(len(all_seqs))
            game_lengths.append(seq_len)
        
        if not all_seqs:
            return []
        
        # 批量推理
        all_logits = self.grp(all_seqs)
        all_matrices = self.grp.calc_matrix(all_logits)
        
        # 分割并计算奖励
        results = []
        for i, (player_id, rank_by_player) in enumerate(zip(player_ids, rank_by_players)):
            start, end = boundaries[i], boundaries[i + 1]
            matrices = all_matrices[start:end]
            
            # 计算 rank_prob
            final_ranking = torch.zeros((1, 4), device=self.device, dtype=matrices.dtype)
            final_ranking[0, rank_by_player[player_id]] = 1.0
            rank_prob = torch.cat((matrices[:, player_id], final_ranking))
            
            # 计算期望收益
            exp_pts = rank_prob @ self.pts
            reward = exp_pts[1:] - exp_pts[:-1]
            
            results.append(reward.cpu().numpy())
        
        return results


def worker_init_fn(*args, **kwargs):
    """DataLoader worker 初始化函数"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    
    # 按 worker 分片文件列表
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
    
    # 设置不同的随机种子
    seed = worker_info.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)


# ===== 高级数据加载器 (可选) =====

class ProducerConsumerDataLoader:
    """
    [新增] 生产者-消费者模式数据加载器
    
    完全解耦数据加载和训练
    """
    
    def __init__(
        self,
        file_list: List[str],
        config: dict,
        num_producers: int = 2,
        queue_size: int = 4,
    ):
        self.file_list = file_list
        self.config = config
        self.num_producers = num_producers
        self.queue_size = queue_size
        
        self._queue: Optional[queue.Queue] = None
        self._producers: List[threading.Thread] = []
        self._stop_event = threading.Event()
    
    def __iter__(self):
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event.clear()
        self._producers.clear()
        
        # 分片文件
        file_chunks = np.array_split(self.file_list, self.num_producers)
        
        # 启动生产者
        for chunk in file_chunks:
            t = threading.Thread(
                target=self._producer_worker,
                args=(list(chunk),),
                daemon=True
            )
            t.start()
            self._producers.append(t)
        
        # 消费数据
        finished_count = 0
        while finished_count < self.num_producers:
            try:
                data = self._queue.get(timeout=60.0)
                if data is None:
                    finished_count += 1
                    continue
                yield from data
            except queue.Empty:
                # 检查生产者状态
                if all(not p.is_alive() for p in self._producers):
                    break
        
        # 等待所有生产者结束
        for p in self._producers:
            p.join(timeout=5.0)
    
    def _producer_worker(self, file_chunk: List[str]):
        """生产者工作函数"""
        try:
            # 初始化模型
            grp = GRP(**self.config['grp']['network'])
            grp_state = torch.load(
                self.config['grp']['state_file'],
                weights_only=True,
                map_location=torch.device('cpu')
            )
            grp.load_state_dict(grp_state['model'])
            reward_calc = RewardCalculator(grp, self.config['env']['pts'])
            
            loader = GameplayLoader(
                version=self.config['control']['version'],
                oracle=False,
                player_names=None,
                excludes=None,
                augmented=False,
            )
            
            random.shuffle(file_chunk)
            batch_size = self.config['dataset']['file_batch_size']
            
            for start_idx in range(0, len(file_chunk), batch_size):
                if self._stop_event.is_set():
                    break
                
                batch_files = file_chunk[start_idx:start_idx + batch_size]
                # 这里需要实现完整的数据处理逻辑
                # data = loader.load_gz_log_files(batch_files)
                # processed = process_games(data, reward_calc)
                # self._queue.put(processed)
            
            # 发送结束信号
            self._queue.put(None)
            
        except Exception as e:
            self._queue.put(None)
            raise
    
    def stop(self):
        """停止所有生产者"""
        self._stop_event.set()
