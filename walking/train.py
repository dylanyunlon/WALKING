"""
Jeff Dean 系统设计原则与深度学习最佳实践

主要优化:
1. 修复 scheduler 调用时机 (仅在优化步骤后调用)
2. 预缓存参数列表避免重复迭代
3. 优化内存管理和批次处理
4. 改进日志和监控
5. 支持动态梯度累积
"""

def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libriichi.consts import obs_shape
    from config import config

    # ===== 配置加载 =====
    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    
    # 配置验证
    assert save_every % opt_step_every == 0, "save_every must be divisible by opt_step_every"
    assert test_every % save_every == 0, "test_every must be divisible by save_every"

    # ===== 设备设置 =====
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']
    
    # 设置确定性 (可选，用于调试)
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(42)

    # ===== 超参数 =====
    pts = config['env']['pts']
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    # ===== 模型初始化 =====
    walking = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (walking, dqn, aux_net)
    
    if enable_compile:
        logging.info('Compiling models with torch.compile()...')
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'walking params: {parameter_count(walking):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    walking.freeze_bn(config['freeze_bn']['walking'])

    # ===== 优化器设置 (带权重衰减分离) =====
    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    
    # [优化] 预缓存所有参数列表，避免每次梯度裁剪时重新迭代
    all_params = list(chain.from_iterable(g['params'] for g in optimizer.param_groups))
    logging.info(f'Total trainable parameters: {sum(p.numel() for p in all_params):,}')
    
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    # ===== 状态恢复 =====
    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        walking.load_state_dict(state['walking'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']
        logging.info(f'Resuming from step {steps:,}')

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(walking, dqn, is_idle=True)
        logging.info('param has been submitted')

    # ===== TensorBoard 设置 =====
    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    
    # [优化] 预分配统计张量
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0
    
    # [优化] 累积步数计数器，用于正确的 scheduler 调用
    accumulation_counter = 0

    def train_epoch():
        nonlocal steps, idx, accumulation_counter

        # ===== 数据准备 =====
        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file', desc='Filtering'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        # ===== DataLoader 设置 =====
        if num_workers > 1:
            random.shuffle(file_list)
            
        file_data = FileDatasetsIter(
            version=version,
            file_list=file_list,
            pts=pts,
            file_batch_size=file_batch_size,
            reserve_ratio=reserve_ratio,
            player_names=player_names,
            num_epochs=num_epochs,
            enable_augmentation=enable_augmentation,
            augmented_first=augmented_first,
        )
        
        data_loader = iter(DataLoader(
            dataset=file_data,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # [优化] 持久化 worker 减少启动开销
            persistent_workers=num_workers > 0,
        ))

        # ===== 批次累积缓冲区 =====
        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            """单批次训练步骤"""
            nonlocal steps, idx, accumulation_counter, pb

            # ===== 数据传输到设备 =====
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
            masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device, non_blocking=True)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
            
            # 验证掩码
            assert masks[range(batch_size), actions].all(), "Invalid action: not in valid mask"

            # ===== 目标值计算 =====
            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            # ===== 前向传播 =====
            with torch.autocast(device.type, enabled=enable_amp):
                phi = walking(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                
                # DQN 损失
                dqn_loss = 0.5 * mse(q, q_target_mc)
                
                # CQL 损失 (仅离线模式)
                cql_loss = 0
                if not online:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                # 辅助任务损失
                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)

                # 总损失
                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                ))

            # ===== 反向传播 (带梯度累积缩放) =====
            scaler.scale(loss / opt_step_every).backward()

            # ===== 统计更新 =====
            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            accumulation_counter += 1

            # ===== 优化器步骤 (每 opt_step_every 步) =====
            if accumulation_counter >= opt_step_every:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    # [优化] 使用预缓存的参数列表
                    clip_grad_norm_(all_params, max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # [修复] scheduler 仅在优化器步骤后调用
                scheduler.step()
                
                accumulation_counter = 0

            pb.update(1)

            # ===== 参数提交 (在线模式) =====
            if online and steps % submit_every == 0:
                submit_param(walking, dqn, is_idle=False)
                logging.info('Param submitted')

            # ===== 检查点保存 =====
            if steps % save_every == 0:
                pb.close()

                # 下采样用于 TensorBoard
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                # 记录指标
                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                
                # [新增] 额外监控指标
                if device.type == 'cuda':
                    writer.add_scalar('system/gpu_memory_gb', 
                                    torch.cuda.memory_allocated(device) / 1e9, steps)
                    writer.add_scalar('system/gpu_memory_cached_gb', 
                                    torch.cuda.memory_reserved(device) / 1e9, steps)
                
                writer.flush()

                # 重置统计
                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'Total steps: {steps:,} (next test in ~{before_next_test_play:,} steps)')

                # 保存检查点
                state = {
                    'walking': walking.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(walking, dqn, is_idle=False)
                    logging.info('Param submitted')

                # ===== 测试对局 =====
                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, walking, dqn, device)
                    walking.train()
                    dqn.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135])
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'Test play avg rank: {stat.avg_rank:.6}')
                    logging.info(f'Test play avg pt: {avg_pt:.6}')
                    
                    # 记录测试指标
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            f'New best model! '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}'
                        )
                        shutil.copy(state_file, best_state_file)
                    
                    if online:
                        sys.exit(0)
                
                pb = tqdm(total=save_every, desc='TRAIN')

        # ===== 主训练循环 =====
        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                # 累积不完整批次
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        # ===== 处理剩余批次 =====
        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
                start = end
                end += batch_size
        pb.close()

        if online:
            submit_param(walking, dqn, is_idle=True)
            logging.info('Final param submitted for epoch')

    # ===== 训练主循环 =====
    while True:
        train_epoch()
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if not online:
            break


def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    is_sub_proc_key = 'WALKING_IS_SUB_PROC'
    online = config['control']['online']
    
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    
    while True:
        child = Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
