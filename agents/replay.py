from collections import deque
import random


class ReplayBuffer(object):
    """ The vanilla replay buffer as in DQN.

    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_action, done):
        self.buffer.append(
            (state, action, reward, next_state, next_action, done))

    def sample(self, batch_size):
        state, action, reward, next_state, next_action, done = zip(
            *random.sample(self.buffer, batch_size))
        return (list(state), list(action), list(reward), list(next_state),
                list(next_action), list(done))

    def sample_from_priority_buffer(self, batch_size):
        return self.sample(batch_size)

    def sample_from_non_priority_buffer(self, batch_size):
        return self.sample(batch_size)

    def __len__(self):
        return len(self.buffer)

    def get_item(self, index):
        return self.buffer[index]


class PriorityReplayBuffer(object):
    """ The priority replay buffer for DQN style training.

    The priority replay buffer keeps a separate buffer for recent
    well-performed trajectories.

    """
    def __init__(self, capacity, decay_rate=0.95):
        self.buffer = deque(maxlen=int(capacity/2))
        self.priority_buffer = deque(maxlen=int(capacity/2))
        self.decay_rate = decay_rate

        self.score = 0
        self.cum_score = 0
        self.cum_count = 0
        self.trajectory_buffer = deque(maxlen=500)

    def push(self, state, action, reward, next_state, next_action, done):
        self.trajectory_buffer.append(
            (state, action, reward, next_state, next_action, done))

        self.score += reward

        if done == 1:
            self.cum_count = self.decay_rate * self.cum_count + 1
            self.cum_score = self.decay_rate * self.cum_score + self.score
            if self.score > (self.cum_score / self.cum_count):
                self.priority_buffer.extend(self.trajectory_buffer)

            self.trajectory_buffer.clear()
            self.score = 0

        self.buffer.append(
            (state, action, reward, next_state, next_action, done))

    def is_priority_buffer_empty(self):
        return len(self.priority_buffer) == 0

    def sample_from_priority_buffer(self, batch_size):
        state, action, reward, next_state, next_action, done = zip(
            *random.sample(self.priority_buffer, batch_size))
        return (list(state), list(action), list(reward), list(next_state),
                list(next_action), list(done))

    def sample_from_non_priority_buffer(self, batch_size):
        state, action, reward, next_state, next_action, done = zip(
            *random.sample(self.buffer, batch_size))
        return (list(state), list(action), list(reward), list(next_state),
                list(next_action), list(done))

    def sample(self, batch_size, rho):
        pbatch = int(batch_size * rho)
        batch = int(batch_size * (1 - rho))
        if pbatch > len(self.priority_buffer):
            pbatch = len(self.priority_buffer)
            batch = batch_size - len(self.priority_buffer)

        state, action, reward, next_state, next_action, done = zip(
            *random.sample(self.buffer, batch)
        ) if batch > 0 else [], [], [], [], [], []

        pstate, paction, preward, pnext_state, pnext_action, pdone = zip(
            *random.sample(self.priority_buffer, pbatch)
        ) if pbatch > 0 else [], [], [], [], [], []

        return (list(pstate) + list(state), list(paction) + list(action),
                list(preward) + list(reward),
                list(pnext_state) + list(next_state),
                list(pnext_action) + list(next_action),
                list(pdone) + list(done))

    def __len__(self):
        return len(self.buffer)
