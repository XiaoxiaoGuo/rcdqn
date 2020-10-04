import torch
import argparse
import math


def get_epsilon_by_frame(config):
    return lambda frame_idx: (config.epsilon_final + (config.epsilon_start -
                              config.epsilon_final) * math.exp(-1. *
                              frame_idx / config.epsilon_decay))


def get_rl_args():
    parser = argparse.ArgumentParser(description='DRL for IF Games')

    # env configuration
    parser.add_argument('--rom_path', default='./roms/jericho-game-suite/{}')
    parser.add_argument('--env_id', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--env_seed', default=12, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--observe_act', action='store_true')
    parser.add_argument('--resume_folder', default='',
                        help='if not empty, resume from checkpoint')
    # model parameters
    parser.add_argument('--encoder_type', default='rnn',
                        choices=['rnn'])
    parser.add_argument('--norm', default='layer', help='layer/batch')
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--max_obs_seq_len', default=400, type=int)
    parser.add_argument('--max_template_num', default=64, type=int)
    parser.add_argument('--max_template_len', default=8, type=int)
    parser.add_argument('--glove_dim', default=100, type=int)
    parser.add_argument('--glove_file', default='./agents/glove/dict.pt',
                        type=str)
    parser.add_argument('--keep_prob', default=1.0,
                        type=float, help="0 <= keep_prob <= 1")

    # - observation history control
    parser.add_argument('--use_history', action='store_true')

    # algorithm
    parser.add_argument('--algo', default='rcdqn', help='rcdqn',
                        choices=['rcdqn'])
    # - multi-step returns
    parser.add_argument('--n_steps', default=1, type=int)
    # - memory
    parser.add_argument('--replay_buffer_type', default='priority',
                        choices=['priority', 'standard'],
                        help='standard/priority')
    parser.add_argument('--replay_buffer_size', default=20000, type=int)
    parser.add_argument('--rho', default=.5, type=float)
    # - experience replay
    parser.add_argument('--epsilon_start', default=1.0, type=float)
    parser.add_argument('--epsilon_final', default=0.05, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--exploit_type', default='greedy',
                        choices=['greedy', 'softmax'])
    # - learning variables
    parser.add_argument('--gamma', default=0.98, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--target_net_update_freq', default=300, type=int)
    parser.add_argument('--training_dump_freq', default=100, type=int)
    parser.add_argument('--normalize_rewards', default='clip',
                        choices=['clip', 'scale', 'original'],
                        help='clip/scale/original')

    # learning control variables
    parser.add_argument('--learn_start', default=64, type=int)
    # roughly 24 hours
    parser.add_argument('--max_steps', default=100000, type=int)

    # experiment logs: log_dir/env_id/DATE/all files
    parser.add_argument('--log_dir', default='logs/')
    parser.add_argument('--experiment_monitor_freq', default=100, type=int)
    parser.add_argument('--recent_history_length', default=100, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--grad_clip', default=10.0, type=float)
    parser.add_argument('--history_window', default=2, type=int)

    args = parser.parse_args()

    # cpu or gpu
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    setattr(args, 'cuda', cuda)
    setattr(args, 'device', device)

    # compute epsilon by frame
    epsilon_by_frame = get_epsilon_by_frame(args)
    setattr(args, 'epsilon_by_frame', epsilon_by_frame)

    return args


def save_rl_config(file_name, config):
    epsilon_by_frame = config.epsilon_by_frame
    config.epsilon_by_frame = None
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(config, f)
    config.epsilon_by_frame = epsilon_by_frame
    return


def load_rl_config(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        config = pickle.load(f)
    config.epsilon_by_frame = get_epsilon_by_frame(config)
    cuda = not config.no_cuda and torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    setattr(config, 'cuda', cuda)
    setattr(config, 'device', device)
    return config


def setup_experiment_folder(config):
    from datetime import datetime
    import os
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")

    try:
        os.makedirs(config.log_dir)
    except OSError:
        print("{} folder exists.".format(config.log_dir))

    try:
        os.makedirs(os.path.join(config.log_dir, config.env_id))
    except OSError:
        print("{} folder exists.".format(
            os.path.join(config.log_dir, config.env_id)))

    try:
        os.makedirs(os.path.join(config.log_dir, config.env_id, date_time))
    except OSError:
        print("{} folder exists.".format(
            os.path.join(config.log_dir, config.env_id, date_time)))

    folder_path = os.path.join(config.log_dir, config.env_id, date_time)
    save_rl_config(os.path.join(folder_path, 'config.pt'), config)
    return folder_path
