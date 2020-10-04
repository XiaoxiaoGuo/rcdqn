import torch
import torch.utils.data as data
import numpy as np


# B x vL --> B x L
def pad_sequences(sequences, maxlen, dtype='int32', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    maxlen = min(np.max(lengths), maxlen)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[:maxlen]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        # post padding
        x[idx, :len(trunc)] = trunc
    return x, lengths


# B x vK x vL --> B x K x L
def pad_group_sequences(sequences, maxgroup, maxlen, dtype='int32', value=0.):
    group_sizes = [len(g) for g in sequences]
    lengths = [len(s) for g in sequences for s in g]
    nb_samples = len(sequences)
    maxlen = min(np.max(lengths), maxlen)

    maxgroup = min(np.max(group_sizes), maxgroup)

    x = (np.ones((nb_samples, maxgroup, maxlen)) * value).astype(dtype)
    for idx, t in enumerate(sequences):
        if len(t) == 0:
            continue  # empty list was found
        for g_idx, g in enumerate(t):
            if len(g) == 0:
                continue
            if g_idx >= maxgroup:
                break
            trunc = g[:maxlen]
            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            # post padding
            x[idx, g_idx, :len(trunc)] = trunc
    return x


# B x 1 --> B x K x L x 1
def pad_object_position(object_position, maxgroup,
                        maxlen, dtype='float', value=0.):
    nb_samples = len(object_position)

    x = (np.ones((nb_samples, maxgroup, maxlen, 1)) * value).astype(dtype)
    for idx, t in enumerate(object_position):
        if len(t) == 0:
            continue  # empty list was found
        for g_idx, g in enumerate(t):
            if g_idx >= maxgroup:
                break
            x[idx, g_idx, g if g < maxlen else maxlen - 1, 0] = 1
    return x


# B x 1 --> B x K x L x 1
def pad_object_position_average(object_position, maxgroup,
                                maxlen, dtype='float', value=0.):
    nb_samples = len(object_position)

    x = (np.ones((nb_samples, maxgroup, maxlen, 1)) * value).astype(dtype)
    for idx, t in enumerate(object_position):
        if len(t) == 0:
            continue  # empty list was found
        for g_idx, g in enumerate(t):
            if g_idx >= maxgroup:
                break
            for gg in g:
                x[idx, g_idx, gg if gg < maxlen
                    else maxlen - 1, 0] = 1.0 / len(g)
    return x


#
def wrap_observation_tensor(state_ids, device, maxlen):
    state_ids, state_lengths = pad_sequences(sequences=state_ids,
                                             maxlen=maxlen)

    batch_state = (torch.from_numpy(state_ids)
                   .to(device=device, dtype=torch.long))

    batch_state_length = (torch.from_numpy(np.asarray(state_lengths))
                          .to(device=device, dtype=torch.long))

    batch_state_length.clamp_max_(max=batch_state.size(1))

    return batch_state, batch_state_length


# padding actions,
# template_ids, B x vK x vL --> B x K x Lq
# obj, B x vK --> B x K x Lc x 1
def wrap_action_tensor(acts, device, state_len_limit,
                       act_num_limit, act_len_limit):
    template_ids, obj1_pos, obj2_pos = acts
    template_ids = pad_group_sequences(sequences=template_ids,
                                       maxgroup=act_num_limit,
                                       maxlen=act_len_limit)
    batch_template = torch.from_numpy(template_ids). \
        to(device=device, dtype=torch.long)

    obj1 = pad_object_position(object_position=obj1_pos,
                               maxgroup=batch_template.size(1),
                               maxlen=state_len_limit)
    obj2 = pad_object_position(object_position=obj2_pos,
                               maxgroup=batch_template.size(1),
                               maxlen=state_len_limit)

    batch_obj1 = torch.from_numpy(obj1).to(device=device, dtype=torch.float)
    batch_obj2 = torch.from_numpy(obj2).to(device=device, dtype=torch.float)

    return batch_template, batch_obj1, batch_obj2


def wrap_action_tensor_average(acts, device, state_len_limit,
                               act_num_limit, act_len_limit):

    template_ids, obj1_pos, obj2_pos = acts
    template_ids = pad_group_sequences(sequences=template_ids,
                                       maxgroup=act_num_limit,
                                       maxlen=act_len_limit)
    batch_template = (torch.from_numpy(template_ids)
                      .to(device=device, dtype=torch.long))

    obj1 = pad_object_position_average(object_position=obj1_pos,
                                       maxgroup=batch_template.size(1),
                                       maxlen=state_len_limit)
    obj2 = pad_object_position_average(object_position=obj2_pos,
                                       maxgroup=batch_template.size(1),
                                       maxlen=state_len_limit)

    batch_obj1 = torch.from_numpy(obj1).to(device=device, dtype=torch.float)
    batch_obj2 = torch.from_numpy(obj2).to(device=device, dtype=torch.float)

    return batch_template, batch_obj1, batch_obj2


# a batch wrapper to handle memory in multi-process
class ImitationExperienceBatch:
    def __init__(self, batch_state, batch_state_length, batch_template,
                 batch_obj1, batch_obj2, batch_act_id, batch_template_mask):
        self.state = batch_state
        self.state_length = batch_state_length
        self.template = batch_template
        self.obj1 = batch_obj1
        self.obj2 = batch_obj2
        self.act_id = batch_act_id
        self.template_mask = batch_template_mask

    # custom memory pinning method
    def pin_memory(self):
        self.state = self.state.pin_memory()
        self.state_length = self.state_length.pin_memory()
        self.template = self.template.pin_memory()
        self.obj1 = self.obj1.pin_memory()
        self.obj2 = self.obj2.pin_memory()
        self.act_id = self.act_id.pin_memory()
        self.template_mask = self.template_mask.pin_memory()
        return self


# a dataset wrapper to support pytorch's built-in multi-process data fetching
class ImitationExperienceReplayBufferDataset(data.Dataset):
    def __init__(self, replay_buffer,
                 state_length_limit=300,
                 action_num_limit=30,
                 action_length_limit=10):

        self.replay_buffer = replay_buffer
        # priority sampling parameters

        # padding limits
        self.action_num_limit = action_num_limit
        self.action_length_limit = action_length_limit
        self.state_length_limit = state_length_limit

        # control the experiment training epoch data size
        self.size_limit = len(self.replay_buffer)

        # data loader is always on CPU
        self.device = 'cpu'

    # tricky: index is not necessary less than size_limit
    def __getitem__(self, index):
        return self.replay_buffer[index]

    # return the training epoch data set, not the real experience replay size
    def __len__(self):
        return self.size_limit
        # return 32

    # only responsible for padding and batching
    # sorting is in model part
    def collate_fn(self, b_data):
        state_ids, acts, act_id = zip(*b_data)
        # padding state
        batch_state, batch_state_length = wrap_observation_tensor(
            state_ids, device=self.device, maxlen=self.state_length_limit)

        t, o1, o2 = zip(*acts)

        batch_template, batch_obj1, batch_obj2 = wrap_action_tensor_average(
            (t, o1, o2), device=self.device,
            act_num_limit=self.action_num_limit,
            act_len_limit=self.action_length_limit,
            state_len_limit=batch_state.size(1))

        template_pad_size = batch_template.size(1)
        template_nums = [
            len(nt) if len(nt) <= template_pad_size else template_pad_size
            for nt in t]
        template_mask = (torch.tensor(
            [[0.] * nt_num + [-1e30] * int(template_pad_size - nt_num)
             for nt_num in template_nums])
                         .to(device=self.device, dtype=torch.float))
        act_id = [act if act < batch_template.size(1) else -1 for act in act_id]
        batch_act_id = (torch.from_numpy(np.asarray(act_id))
                        .to(device=self.device, dtype=torch.long))

        return ImitationExperienceBatch(
            batch_state, batch_state_length, batch_template, batch_obj1,
            batch_obj2, batch_act_id, template_mask)


def wrap_imitation_experience_replay(exp_buffer, config, shuffle,
                                     num_workers=None):
    if not bool(num_workers):
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        num_workers = int(num_workers - 2)

    kwargs = ({'num_workers': num_workers, 'pin_memory': True} if config.cuda
              else {'num_workers': num_workers})
    # print(kwargs)

    exp_dataset = ImitationExperienceReplayBufferDataset(exp_buffer)
    exp_buff_loader = torch.utils.data.DataLoader(
        exp_dataset, batch_size=config.batch_size,
        shuffle=shuffle, collate_fn=exp_dataset.collate_fn,
        drop_last=False, **kwargs)
    return exp_buff_loader


# ---------------------------------------
# ---------- RL setting -----------------
# ---------------------------------------
class ExperienceBatch:
    def __init__(self, batch_state, batch_next_state,
                 batch_state_length, batch_next_state_length,
                 batch_template, batch_next_template,
                 batch_obj1, batch_obj2, batch_next_obj1,  batch_next_obj2,
                 batch_reward, termination_mask, next_template_mask):
        self.state = batch_state
        self.next_state = batch_next_state
        self.state_length = batch_state_length
        self.next_state_length = batch_next_state_length
        self.template = batch_template
        self.next_template = batch_next_template
        self.obj1 = batch_obj1
        self.obj2 = batch_obj2
        self.next_obj1 = batch_next_obj1
        self.next_obj2 = batch_next_obj2
        self.reward = batch_reward
        self.termination_mask = termination_mask
        self.next_template_mask = next_template_mask

    # custom memory pinning method
    def pin_memory(self):
        self.state = self.state.pin_memory()
        self.next_state = self.next_state.pin_memory()
        self.state_length = self.state_length.pin_memory()
        self.next_state_length = self.next_state_length.pin_memory()
        self.template = self.template.pin_memory()
        self.next_template = self.next_template.pin_memory()
        self.obj1 = self.obj1.pin_memory()
        self.obj2 = self.obj2.pin_memory()
        self.next_obj1 = self.next_obj1.pin_memory()
        self.next_obj2 = self.next_obj2.pin_memory()
        self.reward = self.reward.pin_memory()
        self.termination_mask = self.termination_mask.pin_memory()
        self.next_template_mask = self.next_template_mask.pin_memory()
        # self.act_id = self.act_id.pin_memory()
        return self


# a dataset wrapper to support pytorch's built-in multi-process data fetching
class ExperienceReplayBufferDataset(data.Dataset):
    def __init__(self, replay_buffer,
                 size_limit=None,
                 use_priority=True, priority_ratio=0.,
                 random_sampling=True,
                 state_length_limit=300,
                 action_num_limit=32,
                 action_length_limit=10,
                 batch_size=32):

        self.replay_buffer = replay_buffer
        # self.template_obj_counts = template_obj_counts
        # priority sampling parameters
        self.use_priority = use_priority
        self.priority_bias = batch_size * priority_ratio
        self.batch_size = batch_size

        # padding limits
        self.action_num_limit = action_num_limit
        self.action_length_limit = action_length_limit
        self.state_length_limit = state_length_limit

        # control the experiment training epoch data size

        self.size_limit = (len(self.replay_buffer) if not bool(size_limit)
                           else size_limit)
        self.random_sampling = random_sampling

        # data loader is always on CPU
        self.device = 'cpu'

    # tricky: index is not necessary less than size_limit
    def __getitem__(self, index):
        sample_from_priority = (
                self.use_priority and
                not self.replay_buffer.is_priority_buffer_empty())
        sample_from_priority = (
                sample_from_priority and
                index % self.batch_size <= self.priority_bias)

        if self.random_sampling:
            if sample_from_priority:
                sample = self.replay_buffer.sample_from_priority_buffer(1)
            else:
                sample = self.replay_buffer.sample_from_non_priority_buffer(1)
        else:
            sample = self.replay_buffer.get_item(index)
        s, a, R, ns, nA, done = sample
        return s[0], a[0], R[0], ns[0], nA[0], done[0]

    # return the training epoch data set, not the real experience replay size
    def __len__(self):
        return self.size_limit

    # only responsible for padding and batching
    # sorting is in model part
    def collate_fn(self, b_data):
        state_ids, acts, reward, next_state_ids, next_acts, done = zip(*b_data)

        # padding state
        batch_state, batch_state_length = wrap_observation_tensor(
            state_ids, device=self.device, maxlen=self.state_length_limit)
        batch_next_state, batch_next_state_length = wrap_observation_tensor(
            next_state_ids, device=self.device, maxlen=self.state_length_limit)

        # padding actions,
        # template_ids, B x vK x vL --> B x K x Lq
        # obj, B x vK --> B x K x L x 1
        template_ids, obj1_pos, obj2_pos = zip(*acts)

        batch_template, batch_obj1, batch_obj2 = wrap_action_tensor_average(
            (template_ids, obj1_pos, obj2_pos), device=self.device,
            state_len_limit=batch_state.size(1),
            act_num_limit=self.action_num_limit,
            act_len_limit=self.action_length_limit)

        next_template_ids, next_obj1_pos, next_obj2_pos = zip(*next_acts)

        batch_next_template, batch_next_obj1, batch_next_obj2 = (
            wrap_action_tensor_average(
                (next_template_ids, next_obj1_pos, next_obj2_pos),
                device=self.device, state_len_limit=batch_next_state.size(1),
                act_num_limit=self.action_num_limit,
                act_len_limit=self.action_length_limit))
        template_pad_size = batch_next_template.size(1)
        next_template_nums = [
            len(nt) if len(nt) <= template_pad_size else template_pad_size
            for nt in next_template_ids]
        next_template_mask = (torch.tensor(
            [[0.] * nt_num + [-1e7] * int(template_pad_size - nt_num)
             for nt_num in next_template_nums])
                              .to(device=self.device, dtype=torch.float))

        batch_reward = (torch.from_numpy(np.asfarray(reward))
                        .to(device=self.device, dtype=torch.float)
                        .unsqueeze(dim=1))
        termination_mask = (torch.from_numpy(np.asarray(done))
                            .to(device=self.device, dtype=torch.float)
                            .unsqueeze(dim=1))

        return ExperienceBatch(
            batch_state, batch_next_state, batch_state_length,
            batch_next_state_length, batch_template, batch_next_template,
            batch_obj1, batch_obj2, batch_next_obj1,  batch_next_obj2,
            batch_reward, termination_mask, next_template_mask)


def wrap_experience_replay(exp_buffer, config,
                           mode='train', num_workers=None, size_limit=None):
    if not bool(num_workers):
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        num_workers = int(num_workers - 2)

    kwargs = ({'num_workers': num_workers, 'pin_memory': True} if config.cuda
              else {'num_workers': num_workers})
    if mode == 'train':
        size_limit = config.batch_size * config.experiment_monitor_freq
        random_sampling = True
    else:
        size_limit = size_limit
        random_sampling = False

    print('size_limit', size_limit)
    exp_dataset = ExperienceReplayBufferDataset(
        exp_buffer, size_limit=size_limit,
        use_priority=(config.replay_buffer_type == "priority"),
        priority_ratio=config.rho, random_sampling=random_sampling,
        state_length_limit=config.max_obs_seq_len,
        action_num_limit=config.max_template_num,
        action_length_limit=config.max_template_len,
        batch_size=config.batch_size)
    exp_buff_loader = torch.utils.data.DataLoader(
        exp_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=exp_dataset.collate_fn, drop_last=False, **kwargs)
    return exp_buff_loader

