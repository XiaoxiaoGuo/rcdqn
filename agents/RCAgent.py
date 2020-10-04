import time
from datetime import timedelta
from collections import deque
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from agents import BaseAgent
from agents import replay
from agents.nn import networks
from agents.glove import glove_utils
import utils
import utils.EnvTextEncoder as env_encoder


class ReadingComprehensionDQN(BaseAgent.BaseAgent):
    """ Reading comprehension model based DQN agent for Interactive Fiction
    games.

    """
    def __init__(self, args=None, env=None, log_dir='/tmp/gym'):
        super(ReadingComprehensionDQN, self).__init__(log_dir=log_dir)
        self.args = args
        self.device = args.device

        # replay buffer config
        self.replay_buffer_type = args.replay_buffer_type
        self.rho = args.rho

        # learning hyper parameters
        self.gamma = args.gamma
        self.lr = args.lr
        self.target_net_update_freq = args.target_net_update_freq
        self.batch_size = args.batch_size
        self.update_freq = args.update_freq
        self.grad_clip = args.grad_clip
        self.normalize_rewards = args.normalize_rewards
        self.ignore_prev_act = not args.observe_act

        # mode training or evaluation
        self.train_mode = (args.mode == 'train')

        # n-step feedback
        self.nsteps = args.n_steps
        self.nstep_buffer = deque()

        # environment and related parameter settings
        self.env = env
        self.max_obs_seq_len = args.max_obs_seq_len
        self.max_template_num = args.max_template_num
        self.max_template_len = args.max_template_len

        # set up text encoder
        word2id, word_emb = glove_utils.get_dict_emb(args.glove_file)
        self.text_encoder = env_encoder.EnvTextEncoder(env, word2id)

        # nn model
        self.encoder_type = args.encoder_type
        self.init_networks(word_emb)

        # memory
        self.rho = args.rho
        self.declare_memory(args.replay_buffer_size)

        # learning related
        self.bce_loss = nn.BCELoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad,
                   self.model.parameters()),
            lr=self.lr
        )

        self.update_count = 0

        # time records
        self.learn_time = 0
        self.write_time = 0

        # template-level exploration counter
        self.template_count = defaultdict(int)

        return

    def reset_time_log(self):
        self.learn_time = 0
        self.write_time = 0
        return

    def print_time_log(self):
        print('- learn time:{}\n- write time:{}'.format(
            timedelta(milliseconds=self.learn_time),
            timedelta(milliseconds=self.write_time)))

    def declare_memory(self, replay_buffer_size):
        # LSTM-DQN style sampling
        if self.replay_buffer_type == 'priority':
            self.replay_buffer = replay.PriorityReplayBuffer(
                int(replay_buffer_size))
        elif self.replay_buffer_type == 'standard':
            self.replay_buffer = replay.ReplayBuffer(
                int(replay_buffer_size))
        else:
            print('unsupported replay buffer type: {}'.format(
                self.replay_buffer_type)
            )
            exit(1)
        return

    def init_networks(self, state_emb):
        if self.encoder_type == 'rnn':
            self.model = networks.BiDAF(self.args, state_emb)
            self.target_model = networks.BiDAF(self.args, state_emb)
        elif self.encoder_type == 'conv':
            self.model = networks.ConvTransformerBiDAF(self.args, state_emb)
            self.target_model = networks.ConvTransformerBiDAF(
                self.args, state_emb)

        self.target_model.load_state_dict(self.model.state_dict())

        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.train_mode:
            self.model.train()
            self.target_model.train()
        else:
            self.model.eval()
            self.target_model.eval()
        return

    def get_template_size(self):
        return self.text_encoder.get_template_size()

    def get_act_vocab_size(self):
        return self.text_encoder.get_act_vocab_size()

    # obs --> state_ids
    def encode_observation(self, observation):
        return self.text_encoder.encode_observation(
            observation, ignore_action=self.ignore_prev_act
        )

    # info --> action_ids
    def encode_action(self, actions, objects, state_ids):
        return self.text_encoder.encode_actions(actions, objects, state_ids)

    # s/0, a/1, r/2, s_/3, A_/4, done/5
    def append_to_replay(self, s, a, r, ns, na, done):
        self.nstep_buffer.append((s, a, r, ns, na, done))
        if done == 1:
            self.finish_nstep()
        else:
            if len(self.nstep_buffer) < self.nsteps:
                return

            cum_r = sum([self.nstep_buffer[i][2] * (self.gamma ** i)
                         for i in range(self.nsteps)])
            state, action, _, _, _, done = self.nstep_buffer.popleft()
            self.replay_buffer.push(state=state,
                                    action=action,
                                    reward=cum_r,
                                    next_state=ns,
                                    next_action=na,
                                    done=done)

    # only be called when a terminal state is appended
    def finish_nstep(self):
        if len(self.nstep_buffer) == 0:
            return
        # the last terminal state
        ns = self.nstep_buffer[-1][3]
        na_set = self.nstep_buffer[-1][4]
        while len(self.nstep_buffer) > 0:
            r_sum = sum([self.nstep_buffer[i][2] * (self.gamma ** i)
                        for i in range(len(self.nstep_buffer))])
            state, action, _, _, _, _ = self.nstep_buffer.popleft()
            # with threading.Lock():
            self.replay_buffer.push(state=state,
                                    action=action,
                                    reward=r_sum,
                                    next_state=ns,
                                    next_action=na_set,
                                    done=1)

    def update_experience_replay(self, s, aset, a, r, ns, na, done):
        if not self.train_mode:
            return None
        ep_start = int(round(time.time() * 1000))
        # A, a --> aa
        template_ids, obj1_pos, obj2_pos = aset
        aa = ([template_ids[a]], [obj1_pos[a]], [obj2_pos[a]])
        if self.normalize_rewards == 'clip':
            r = np.sign(r)
        elif self.normalize_rewards == 'scale':
            r = r / 10.

        self.append_to_replay(
            s=s, a=aa, r=r, ns=ns, na=na, done=1 if done else 0)
        self.write_time += int(round(time.time() * 1000)) - ep_start
        return

    def get_action(self, state_ids, action_tuple, eps,
                   greedy=True, temperature=10):
        with torch.no_grad():
            template_ids, obj1_pos, obj2_pos, actions = action_tuple
            if np.random.random() >= eps:

                state_ids, state_len = utils.wrap_observation_tensor(
                    [state_ids], device=self.device,
                    maxlen=self.max_obs_seq_len)

                template_ids, obj1, obj2 = utils.wrap_action_tensor_average(
                    ([template_ids], [obj1_pos], [obj2_pos]),
                    device=self.device,
                    state_len_limit=state_ids.size(1),
                    act_num_limit=self.max_template_num,
                    act_len_limit=self.max_template_len
                )

                q_values = self.model(
                    state_ids, template_ids, state_len, obj1, obj2).view(-1)
                dist = F.softmax(q_values / temperature, -1)
                if greedy:
                    _, sampled_act = dist.max(dim=0)
                    prob = 1
                else:
                    sampled_act = dist.multinomial(num_samples=1)[0]
                    prob = dist[sampled_act.item()].item()
                sampled_act = sampled_act.item()
            else:
                prob = eps / len(actions)
                tm_count = [self.template_count[actions[i].template_id] + 1.
                            for i in range(len(actions))]
                tm_sum = sum(tm_count) * 2
                tm_w = [np.log(tm_sum) / tm_count[i]
                        for i in range(len(actions))]
                tm_w = torch.tensor(tm_w).view(1, -1)
                dist = F.softmax(tm_w, -1)
                sampled_act = dist.multinomial(num_samples=1)[0]
            self.template_count[actions[sampled_act].template_id] += 1
        return sampled_act, actions[sampled_act].action, prob

    def compute_loss(self, batch_vars):
        state = batch_vars.state.to(self.device, non_blocking=True)
        state_len = batch_vars.state_length.to(self.device, non_blocking=True)
        next_state = batch_vars.next_state.to(self.device, non_blocking=True)
        next_state_len = batch_vars.next_state_length.to(
            self.device, non_blocking=True)

        template = batch_vars.template.to(self.device, non_blocking=True)
        obj1 = batch_vars.obj1.to(self.device, non_blocking=True)
        obj2 = batch_vars.obj2.to(self.device, non_blocking=True)

        next_template = batch_vars.next_template.to(
            self.device, non_blocking=True)
        next_obj1 = batch_vars.next_obj1.to(self.device, non_blocking=True)
        next_obj2 = batch_vars.next_obj2.to(self.device, non_blocking=True)

        reward = batch_vars.reward.to(self.device, non_blocking=True)
        termination_mask = batch_vars.termination_mask.to(
            self.device, non_blocking=True)
        next_template_mask = batch_vars.next_template_mask.to(
            self.device, non_blocking=True)

        # estimate
        self.model.sample_noise()

        q_values = self.model(state, template, state_len, obj1, obj2)

        with torch.no_grad():
            # with done impact
            next_q_values = self.get_max_next_state_action_values(
                next_state, next_template, next_state_len, next_obj1,
                next_obj2, next_template_mask)
            next_q_target = (next_q_values * (1.0 - termination_mask) *
                             (self.gamma ** self.nsteps) + reward)

        td_loss = F.smooth_l1_loss(q_values, next_q_target, reduction='mean')

        return td_loss, None

    #
    def learn_step(self, batch_vars):
        ep_start = int(round(time.time() * 1000))
        td_loss, _ = self.compute_loss(batch_vars)
        # Optimize the model
        loss = td_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.grad_clip)
        self.optimizer.step()
        self.update_target_model()
        self.learn_time += int(round(time.time() * 1000)) - ep_start
        return td_loss.item(), 0

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            with torch.no_grad():
                self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action_values(
            self, next_state, next_template, next_state_len, next_obj1,
            next_obj2, next_template_mask):
        with torch.no_grad():
            next_q = self.target_model(
                next_state, next_template, next_state_len, next_obj1, next_obj2)

            next_q = next_q + next_template_mask
            return next_q.max(dim=1)[0].view(-1, 1)

    def reset_hx(self):
        pass

    def count_trainable_parameters(self):
        with torch.no_grad():
            count = sum(
                [p.numel() for p in self.model.parameters() if p.requires_grad])
        return count

    def get_trainable_parameter_norm(self):
        with torch.no_grad():
            norm = sum(
                [p.norm(1).item()
                 for p in self.model.parameters() if p.requires_grad])
        return norm
