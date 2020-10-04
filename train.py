import os
from jericho.util import clean
from tqdm import tqdm
import time
from datetime import timedelta
from timeit import default_timer as timer
import agents
import utils


def train():
    # initialize environments and set up logging folders
    config = utils.get_rl_args()
    rom = config.rom_path.format(config.env_id)
    env = utils.make_env(rom, 0, max_episode_steps=config.env_step_limit)

    frame_idx = 0
    resume_flag = False
    if bool(config.resume_folder):
        folder_path = os.path.join(
            config.log_dir, config.env_id, config.resume_folder)
        if os.path.exists(folder_path):
            print('## Resume training from ', folder_path)

            last_frame_idx = 0
            with open(os.path.join(folder_path, 'loss.csv'), 'r') as f:
                for line in f:
                    if line[0] == '#':
                        continue
                    last_frame_idx = int(line.split(',')[0])
            print('## ## last_frame_idx', last_frame_idx)
            frame_idx = last_frame_idx
            if os.path.exists(os.path.join(folder_path, 'model.pt')):
                resume_flag = True
        else:
            print('## Initialize training from ', folder_path)
            try:
                os.makedirs(folder_path)
            except OSError:
                print('Creating {} folder failed.'.format(folder_path))
    else:
        folder_path = utils.setup_experiment_folder(config)

    model = agents.get_agent(config=config, env=env, log_dir=folder_path)
    monitor = utils.ExperimentMonitor(config, folder_path)

    if resume_flag:
        print('## load checkpoint from ', folder_path)
        model.load_checkpoint(folder_path)
        monitor.add_separator()

    dataset = utils.wrap_experience_replay(
        model.replay_buffer, config,
        size_limit=config.experiment_monitor_freq * config.batch_size)

    episode_logger = {'reward': 0, 'init_time': 0, 'num': 0}
    greedy = (config.exploit_type == 'greedy')

    # some logging functions
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(folder_path, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')
                f_log.flush()

    def dump_trajectory_action(action_text, actions, action_id, frame_idx):
        if (episode_logger['num'] % config.training_dump_freq ==
                config.training_dump_freq - 1):
            frame_num = frame_idx - episode_logger['init_time']
            logging('[Episode {} step {}] Act: {}=({})\n'.format(
                episode_logger['num'], frame_num, action_text,
                actions[action_id]))
        return

    def dump_trajectory_state(obs_text, frame_idx):
        if (episode_logger['num'] % config.training_dump_freq ==
                config.training_dump_freq - 1):
            st = obs_text.split('|')
            logging('[Episode {} step {}] Obs: \nl={}\ni={}\no={}\n'.format(
                episode_logger['num'], frame_idx - episode_logger['init_time'],
                clean(st[0]), clean(st[1]), clean(st[2])))
        return

    def dump_rewards(reward, frame_idx):
        if (episode_logger['num'] % config.training_dump_freq ==
                config.training_dump_freq - 1):
            logging('[Episode {} step {}] Reward:{}, CumR:{}'.format(
                episode_logger['num'], frame_idx - episode_logger['init_time'],
                reward, episode_logger['reward'],
            ))
        return

    # history observation
    obs_history = utils.ObservationHistory(config.history_window)

    # interact with the environment
    def actor_step(obs_ids, action_tuple, frame_idx):
        # compute current action
        template_ids, obj1_pos, obj2_pos, actions = action_tuple
        epsilon = config.epsilon_by_frame(frame_idx)

        action_id, action_text, prob = model.get_action(
            obs_ids, action_tuple, epsilon, greedy=greedy)

        dump_trajectory_action(action_text, actions, action_id, frame_idx)

        # interact with the environment
        next_obs_text, reward, done, next_info = env.step(
            action_text, parallel=True)
        episode_logger['reward'] += reward
        #
        done = done or len(next_info['valid_act']) == 0

        # history part
        past_obs = ""
        if config.use_history:
            active_entity = obs_history.extract_entity(next_info['valid_act'])
            past_obs = obs_history.retrieve_obs(active_entity)
            obs_history.update_history(active_entity, next_obs_text)

        dump_rewards(reward, frame_idx)
        dump_trajectory_state(next_obs_text, frame_idx+1)
        next_obs_ids = model.encode_observation(past_obs + next_obs_text)
        next_action_tuple = model.encode_action(
            next_info['valid_act'], next_info['objs'], next_obs_ids)

        next_template_ids = next_action_tuple[0]
        next_obj1_pos = next_action_tuple[1]
        next_obj2_pos = next_action_tuple[2]

        # update experience replay
        model.update_experience_replay(
            s=obs_ids, aset=(template_ids, obj1_pos, obj2_pos),
            a=action_id, r=reward, done=done, ns=next_obs_ids,
            na=(next_template_ids, next_obj1_pos, next_obj2_pos))
        # tracking behavior trajectories
        monitor.add_ard(frame_idx, actions[action_id], reward, done, prob)

        if done or env.env.emulator_halted():
            score = next_info['score']
            model.reset_hx()
            next_obs_text, next_info = env.reset(parallel=True)
            past_obs = ""
            if config.use_history:
                obs_history.reset()
                active_entity = obs_history.extract_entity(info['valid_act'])
                past_obs = obs_history.retrieve_obs(active_entity)
                obs_history.update_history(active_entity, obs_text)

            next_obs_ids = model.encode_observation(past_obs + next_obs_text)
            next_action_tuple = model.encode_action(
                next_info['valid_act'], next_info['objs'], next_obs_ids)

            monitor.add_episode_reward(
                episode_logger['reward'], score, frame_idx)
            episode_logger['reward'] = 0
            episode_logger['init_time'] = frame_idx
            episode_logger['num'] += 1
            dump_trajectory_state(next_obs_text, frame_idx)

        return next_obs_ids, next_action_tuple

    logging(str(config))

    obs_text, info = env.reset(parallel=True)

    past_obs = ""
    if config.use_history:
        active_entity = obs_history.extract_entity(info['valid_act'])
        past_obs = obs_history.retrieve_obs(active_entity)
        obs_history.update_history(active_entity, obs_text)

    dump_trajectory_state(obs_text, frame_idx)
    obs_ids = model.encode_observation(past_obs + obs_text)
    action_tuple = model.encode_action(info['valid_act'], info['objs'], obs_ids)

    start = timer()
    model.reset_time_log()
    act_time = 0

    # pre-fill exp replay for |learn_start| steps
    if frame_idx < config.learn_start:
        for time_step in tqdm(range(config.learn_start), desc='non-train step'):
            obs_ids, action_tuple = actor_step(
                obs_ids, action_tuple, frame_idx=frame_idx)
            frame_idx += 1
    loop_length = config.experiment_monitor_freq * config.update_freq
    loop_start = frame_idx // loop_length
    loop_max = int(config.max_steps / loop_length) + 1
    for loop_idx in range(loop_start, loop_max):
        time_start = loop_idx * loop_length
        time_end = time_start + loop_length
        for batch_vars in tqdm(
                dataset, desc='training step {}-{}'.format(time_start, time_end)):
            # one step update

            td_loss, aux_loss = model.learn_step(batch_vars)
            norm = model.get_trainable_parameter_norm()
            monitor.add_loss(frame_idx, td_loss, aux_loss, norm)
            # interact with environment and write data
            act_ep_time = int(round(time.time() * 1000))
            for _ in range(config.update_freq):
                obs_ids, action_tuple = actor_step(obs_ids, action_tuple, frame_idx)
                frame_idx += 1
            act_time += int(round(time.time() * 1000)) - act_ep_time

        model.save_networks()
        model.save_optimizer()
        model.save_replay()

        e_r, score = monitor.get_episode_reward_record()
        action_record = monitor.get_action_record()
        td_avg, td_max, td_min = monitor.get_td_record()
        norm_avg, norm_max, norm_min = monitor.get_norm_record()
        exp_avg, exp_max, exp_min = monitor.get_exploration_record()
        # aux_avg, aux_max, aux_min = monitor.get_aux_record()
        logging('step {}, time {}, episode {}, R (avg/max/min) '
                '{:.1f}/{:.1f}/{:.1f}::{:.1f}/{:.1f}/{:.1f}, '
                'epx (p/n) {:.0f}/{:.0f} \n'
                'tpl (max/avg/num) {:.2f}/{:.2f}/{}, '
                'obj (max/avg/num) {:.2f}/{:.2f}/{}, '
                'td (avg/max) {:.3f}/{:.3f}, norm (avg) {:.5f}, '
                'eps {:.3f}/{:.3f}:{:.3f}:{:.3f}'
                .format(frame_idx, timedelta(seconds=int(timer() - start)),
                        episode_logger['num'],
                        e_r[0], e_r[1], e_r[2],
                        score[0], score[1], score[2],
                        len(model.replay_buffer.priority_buffer),
                        len(model.replay_buffer.buffer),
                        action_record['template'][0],
                        action_record['template'][1],
                        action_record['template'][2],
                        action_record['obj'][0],
                        action_record['obj'][1],
                        action_record['obj'][2],
                        td_avg, td_max, norm_avg,
                        config.epsilon_by_frame(frame_idx),
                        exp_avg, exp_max, exp_min))
        model.print_time_log()
        model.reset_time_log()
        print('- act time:{}'.format(timedelta(milliseconds=act_time)))
        act_time = 0

    model.save_checkpoint()
    env.close()


if __name__ == "__main__":
    train()
