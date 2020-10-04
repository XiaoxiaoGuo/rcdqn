import pickle
import os.path
import torch


class BaseAgent(object):
    """" The base class of reinforcement learning agents.

    The base class provides the common model saving/loading utility functions
    for DQN, DDQN, PPO, A2C, etc.
    """

    def __init__(self, log_dir):
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.log_dir = log_dir
        self.rewards = []
        self.device = None
        self.replay_buffer = None

    # save / load interfaces
    def save_networks(self):
        torch.save(self.model.state_dict(),
                   os.path.join(self.log_dir, 'model.pt'))
        torch.save(self.target_model.state_dict(),
                   os.path.join(self.log_dir, 'target_model.pt'))

    def load_networks(self, log_dir=''):
        fname_model = os.path.join(log_dir, 'model.pt')
        fname_tmodel = os.path.join(log_dir, 'target_model.pt')
        if os.path.isfile(fname_model):
            print('## load model from ', fname_model)
            self.model.load_state_dict(
                torch.load(fname_model, map_location=self.device))
        if os.path.isfile(fname_tmodel):
            print('## load target model from ', fname_tmodel)
            self.target_model.load_state_dict(
                torch.load(fname_tmodel, map_location=self.device))

    def save_optimizer(self):
        torch.save(self.optimizer.state_dict(),
                   os.path.join(self.log_dir, 'optim.pt'))

    def load_optimizer(self, log_dir):
        fname_optim = os.path.join(log_dir, 'optim.pt')
        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(
                torch.load(fname_optim, map_location=self.device))

    # computationally heavy
    def save_replay(self):
        pickle.dump(self.replay_buffer,
                    open(os.path.join(self.log_dir, 'exp_replay.pt'), 'wb'))

    def load_replay(self, log_dir):
        fname = os.path.join(log_dir, 'exp_replay.pt')
        if os.path.isfile(fname):
            print('## load replay memory from ', fname)
            self.replay_buffer = pickle.load(open(fname, 'rb'))

    def load_checkpoint(self, log_dir):
        self.load_networks(log_dir)
        self.load_optimizer(log_dir)
        self.load_replay(log_dir)
        return

    def save_checkpoint(self):
        self.save_networks()
        self.save_optimizer()
        self.save_replay()
        return
