import csv
import os
from statistics import mean
from collections import Counter


class ExperimentMonitor:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir
        self.log_freq = config.experiment_monitor_freq
        self.recent_history_length = config.recent_history_length

        self.episode_reward_history = []
        self.score_history = []

        # td mean, max, min
        self.loss_history = []
        self.last_td_record = (0, 0, 0)
        self.last_aux_record = (0, 0, 0)
        self.last_norm_record = (0, 0, 0)
        self.last_exp_record = (0, 0, 0)

        # action, reward, termination
        self.ard_history = []
        self.last_action_record = {'template': (0, 0, 0), 'obj': (0, 0, 0)}
        self.episode_reward_length = 10 # 10 ~ 100
        self.ep_count = 0
        return

    def add_separator(self):
        with open(os.path.join(self.log_dir, 'loss.csv'), 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(('#', '#', '#', '#'))
        with open(os.path.join(
                self.log_dir, 'action_reward_term.csv'), 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(('#', '#', '#', '#'))

    # experiment monitoring
    def add_loss(self, tstep, td, aux, norm):
        self.loss_history.append((tstep, td, aux, norm))
        if tstep % self.log_freq == 0:
            with open(os.path.join(self.log_dir, 'loss.csv'), 'a+') as f:
                writer = csv.writer(f)
                for row in self.loss_history:
                    writer.writerow(row)
            _, td, aux, norm = zip(*self.loss_history)
            self.last_td_record = (mean(td), max(td), min(td))
            self.last_aux_record = (mean(aux), max(aux), min(aux))
            self.last_norm_record = (mean(norm), max(norm), min(norm))
            self.loss_history.clear()

    def add_episode_reward(self, reward, score, step):
        self.episode_reward_history.append(reward)
        self.score_history.append(score)
        with open(os.path.join(self.log_dir, 'ep_logs.txt'), 'a+') as f:
            f.write('{} | {} | {} | {}\n'.format(
                self.ep_count, reward, score, step))
            self.ep_count += 1

    def get_episode_reward_record(self):
        if len(self.episode_reward_history) >= self.episode_reward_length:
            latest_rewards = self.episode_reward_history[-self.episode_reward_length:]
            self.episode_reward_history = latest_rewards
        if len(self.score_history) >= self.episode_reward_length:
            latest_scores = self.score_history[-self.episode_reward_length:]
            self.score_history = latest_scores
        return (mean(self.episode_reward_history),
                max(self.episode_reward_history),
                min(self.episode_reward_history)), (mean(self.score_history),
                                                    max(self.score_history),
                                                    min(self.score_history))

    def get_action_record(self):
        return self.last_action_record

    def get_td_record(self):
        return self.last_td_record

    def get_aux_record(self):
        return self.last_aux_record

    def get_norm_record(self):
        return self.last_norm_record

    def get_exploration_record(self):
        return self.last_exp_record

    def add_ard(self, tstep, action, reward, done, prob):
        self.ard_history.append((tstep, action, reward, done, prob))
        if tstep % self.log_freq == 0:
            with open(os.path.join(
                    self.log_dir, 'action_reward_term.csv'), 'a+') as f:
                writer = csv.writer(f)
                for row in self.ard_history:
                    writer.writerow(row)
            _, actions, _, _, prob = zip(*self.ard_history)
            # compute template max, min etc.
            # templates = [-1] * len(actions)
            # objs = [-1] * (len(actions) * 2)
            self.last_exp_record = (mean(prob), max(prob), min(prob))

            templates = []
            objs = []
            for action in actions:
                # print(action)
                templates.append(action.template_id)
                objs.extend(action.obj_ids)

            self.last_action_record = {'template': (0, 0, 0), 'obj': (0, 0, 0)}

            template_num = len(templates)
            obj_num = len(objs)

            if template_num > 0:
                templates = Counter(templates)
                self.last_action_record['template'] = (
                    templates.most_common(1)[0][1] / template_num,
                    len(templates) / template_num, len(templates))
            if obj_num > 0:
                objs = Counter(objs)
                self.last_action_record['obj'] = (
                    objs.most_common(1)[0][1] / obj_num,
                    len(objs) / obj_num, len(objs))

            self.ard_history.clear()
