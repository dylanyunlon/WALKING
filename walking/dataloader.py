import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config
import logging
from collections import defaultdict

class FileDatasetsIter(IterableDataset):
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
        self.iterator = None

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        
        # 详细的统计信息
        stats = {
            'total_games': 0,
            'skipped_games': 0,
            'skip_reasons': defaultdict(int),
            'successful_entries': 0,
            'kyoku_rewards_stats': [],
            'at_kyoku_stats': [],
            'game_size_stats': []
        }
        
        for file_idx, file in enumerate(data):
            for game_idx, game in enumerate(file):
                stats['total_games'] += 1
                try:
                    # per move
                    obs = game.take_obs()
                    if self.oracle:
                        invisible_obs = game.take_invisible_obs()
                    actions = game.take_actions()
                    masks = game.take_masks()
                    at_kyoku = game.take_at_kyoku()
                    dones = game.take_dones()
                    apply_gamma = game.take_apply_gamma()

                    # per game
                    grp = game.take_grp()
                    player_id = game.take_player_id()

                    game_size = len(obs)
                    stats['game_size_stats'].append(game_size)
                    
                    # 检查基本数据完整性
                    if game_size == 0:
                        stats['skipped_games'] += 1
                        stats['skip_reasons']['empty_game_size'] += 1
                        continue
                        
                    if len(at_kyoku) == 0:
                        stats['skipped_games'] += 1
                        stats['skip_reasons']['empty_at_kyoku'] += 1
                        continue

                    grp_feature = grp.take_feature()
                    rank_by_player = grp.take_rank_by_player()
                    kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                    
                    stats['kyoku_rewards_stats'].append(len(kyoku_rewards))
                    stats['at_kyoku_stats'].append(max(at_kyoku) if at_kyoku else -1)
                    
                    # 防御性检查：如果kyoku_rewards长度不足，跳过这个游戏
                    max_kyoku_idx = at_kyoku[-1]
                    required_length = max_kyoku_idx + 1
                    
                    if len(kyoku_rewards) < required_length:
                        stats['skipped_games'] += 1
                        stats['skip_reasons']['insufficient_kyoku_rewards'] += 1
                        # 详细记录第一个这样的案例
                        if stats['skip_reasons']['insufficient_kyoku_rewards'] <= 3:
                            logging.warning(f"DETAILED: Game {game_idx} in file {file_idx}: "
                                          f"kyoku_rewards={len(kyoku_rewards)}, "
                                          f"max_kyoku_idx={max_kyoku_idx}, "
                                          f"required={required_length}, "
                                          f"at_kyoku range=[{min(at_kyoku)}, {max(at_kyoku)}], "
                                          f"game_size={game_size}")
                        continue

                    final_scores = grp.take_final_scores()
                    scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                    rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                    player_ranks = rank_by_player_seq[:, player_id]
                    
                    # 检查 player_ranks 长度
                    if len(player_ranks) <= max_kyoku_idx + 1:
                        stats['skipped_games'] += 1
                        stats['skip_reasons']['insufficient_player_ranks'] += 1
                        # 详细记录第一个这样的案例
                        if stats['skip_reasons']['insufficient_player_ranks'] <= 3:
                            logging.warning(f"DETAILED: Game {game_idx} in file {file_idx}: "
                                          f"player_ranks={len(player_ranks)}, "
                                          f"required={max_kyoku_idx + 2}, "
                                          f"grp_feature.shape={grp_feature.shape if grp_feature is not None else None}")
                        continue

                    steps_to_done = np.zeros(game_size, dtype=np.int64)
                    for i in reversed(range(game_size)):
                        if not dones[i]:
                            steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                    entries_added = 0
                    for i in range(game_size):
                        kyoku_idx = at_kyoku[i]
                        # 额外的边界检查
                        if kyoku_idx >= len(kyoku_rewards):
                            stats['skip_reasons']['kyoku_index_oob'] += 1
                            break
                        if kyoku_idx + 1 >= len(player_ranks):
                            stats['skip_reasons']['player_rank_index_oob'] += 1
                            break
                            
                        entry = [
                            obs[i],
                            actions[i],
                            masks[i],
                            steps_to_done[i],
                            kyoku_rewards[kyoku_idx],
                            player_ranks[kyoku_idx + 1],
                        ]
                        if self.oracle:
                            entry.insert(1, invisible_obs[i])
                        self.buffer.append(entry)
                        entries_added += 1
                    
                    if entries_added > 0:
                        stats['successful_entries'] += entries_added
                        
                except Exception as e:
                    stats['skipped_games'] += 1
                    stats['skip_reasons']['exception'] += 1
                    # 详细记录前几个异常
                    if stats['skip_reasons']['exception'] <= 3:
                        logging.error(f"DETAILED: Exception in game {game_idx} in file {file_idx}: {e}")
                    continue
        
        # 详细的统计报告
        skip_rate = stats['skipped_games'] / stats['total_games'] * 100 if stats['total_games'] > 0 else 0
        
        logging.info(f"=== BATCH PROCESSING REPORT ===")
        logging.info(f"Total games processed: {stats['total_games']}")
        logging.info(f"Successfully processed: {stats['total_games'] - stats['skipped_games']}")
        logging.info(f"Skipped games: {stats['skipped_games']} ({skip_rate:.1f}%)")
        logging.info(f"Total training entries generated: {stats['successful_entries']}")
        
        if stats['skip_reasons']:
            logging.info(f"Skip reasons breakdown:")
            for reason, count in stats['skip_reasons'].items():
                logging.info(f"  - {reason}: {count}")
        
        if stats['kyoku_rewards_stats']:
            kyoku_rewards_lengths = stats['kyoku_rewards_stats']
            logging.info(f"Kyoku rewards length stats: min={min(kyoku_rewards_lengths)}, "
                        f"max={max(kyoku_rewards_lengths)}, "
                        f"avg={np.mean(kyoku_rewards_lengths):.1f}")
        
        if stats['at_kyoku_stats']:
            max_kyoku_indices = stats['at_kyoku_stats']
            logging.info(f"Max kyoku index stats: min={min(max_kyoku_indices)}, "
                        f"max={max(max_kyoku_indices)}, "
                        f"avg={np.mean(max_kyoku_indices):.1f}")
        
        if stats['game_size_stats']:
            game_sizes = stats['game_size_stats']
            logging.info(f"Game size stats: min={min(game_sizes)}, "
                        f"max={max(game_sizes)}, "
                        f"avg={np.mean(game_sizes):.1f}")

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]