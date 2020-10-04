from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from collections import defaultdict
import hashlib
import itertools
from multiprocessing import Process, Pipe


def clean_obs(s):
    garbage_chars = ['*', '-', '!', '[', ']']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()


# jericho's world change detection is inconsistent with hash code
def _get_world_state_hash(env):
    world_str = ', '.join([str(o) for o in env.get_world_objects()])
    m = hashlib.md5()
    m.update(world_str.encode('utf-8'))
    return m.hexdigest()


def make_env(rom_path, seed, max_episode_steps=None, use_parallel_envs=True):
    if use_parallel_envs:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        num_workers = int(num_workers - 4)
    else:
        num_workers = 0
    env = JerichoEnv(rom_path, seed, max_episode_steps, env_num=num_workers)
    return env


def _check_valid_action_operator(env, state, chunk):
    diff2acts = defaultdict(list)
    if not bool(chunk):
        return diff2acts

    if env.emulator_halted():
        env.reset()

    env.set_state(state)
    orig_score = env.get_score()

    for act in chunk:
        env.set_state(state)
        if isinstance(act, defines.TemplateAction):
            obs, rew, done, info = env.step(act.action)
        else:
            obs, rew, done, info = env.step(act)

        if env.emulator_halted():
            env.reset()
            continue

        if info['score'] != orig_score or done or env.world_changed():
            if '(Taken)' in obs:
                continue
            diff = env._get_world_diff()
            diff2acts[diff].append(act)
    keys = diff2acts.keys()
    for key in keys:
        acts = diff2acts[key]
        acts.sort(key=lambda x: (x.template_id, tuple(x.obj_ids)))
        diff2acts[key] = [acts[0]]

    return diff2acts


def worker(remote, parent_remote, env):
    parent_remote.close()
    # env.create()
    # print('env', env)
    try:
        # done = False
        while True:
            state, candidate_actions = remote.recv()
            valid_act = _check_valid_action_operator(
                env, state, candidate_actions)
            # print('state', state)
            # print('candidate_actions', candidate_actions)
            remote.send(valid_act)
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class JerichoEnv:
    def __init__(self, rom_path, seed, step_limit=None, env_num=8):
        self.rom_path = rom_path
        self.seed = seed

        self.step_limit = step_limit

        self.bindings = load_bindings(rom_path)
        self.seed = self.bindings['seed']
        # some additional templates here, could make it game specific
        # Note: changes to it may cause the template id inconsistent
        self.additional_templates = ['land']
        self.act_gen = TemplateActionGenerator(self.bindings)
        self.act_gen.templates = list(set(self.act_gen.templates
                                          + self.additional_templates))
        self.act_gen.templates.sort()
        self.id2template = None
        self.template2id = None
        self._compute_template()
        self.env = FrotzEnv(self.rom_path, self.seed)

        self.env_num = env_num
        self.ps = None
        self.envs = None
        self.remotes = None
        self.work_remotes = None
        self.parallel = False
        self._init_parallel_workers()

        self.steps = 0
        self.max_word_len = self.bindings['max_word_length']

        self.word2id = None
        self.id2word = None
        self.noun_words = None
        self._compute_vocab_act()
        self.state2valid_acts = {}

    def _init_parallel_workers(self):
        if self.env_num > 0:
            self.parallel = True
            self.envs = [
                FrotzEnv(self.rom_path, self.seed) for _ in range(self.env_num)]
            self.remotes, self.work_remotes = zip(
                *[Pipe() for _ in range(self.env_num)])
            self.ps = [Process(
                target=worker, args=(self.work_remotes[i], self.remotes[i],
                                     self.envs[i]))
                       for i in range(self.env_num)]
            for p in self.ps:
                p.daemon = True
                p.start()
            for remote in self.work_remotes:
                remote.close()

    def _compute_vocab_act(self):
        # loading vocab directly from Jericho
        env_dict = self.env.get_dictionary()
        vocab = {i + 2: str(v) for i, v in enumerate(env_dict)}
        vocab[0] = ' '
        vocab[1] = '<s>'
        vocab_rev = {v: idx for idx, v in vocab.items()}
        self.word2id = vocab_rev
        self.id2word = vocab
        self.noun_words = set([w.word for w in env_dict if w.is_noun])
        return

    def _compute_template(self):
        self.id2template = {}
        self.template2id = {}
        for i, t in enumerate(self.act_gen.templates):
            self.id2template[i] = t
            self.template2id[t] = i
        return

    def get_max_word_len(self):
        return self.max_word_len

    def get_dictionary(self):
        return self.env.get_dictionary()

    def get_bindings(self):
        return self.bindings

    def get_id2act_word(self):
        return self.id2word

    def get_id2template(self):
        return self.id2template

    def get_template2id(self):
        return self.template2id

    def get_act_word2id(self):
        return self.word2id

    def close(self):
        self.env.close()
        for env in self.envs:
            env.close()
        for remote in self.remotes:
            remote.close()

    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        template_str = self.act_gen.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.id2word[o1_id])
        else:
            return (template_str.replace('OBJ', self.id2word[o1_id], 1)
                    .replace('OBJ', self.id2word[o2_id], 1))

    def get_world_state_hash(self, ignore=False):
        if ignore:
            return self.env.get_world_state_hash()
        return _get_world_state_hash(self.env)

    def _identify_objects_on_current_state(self, ob):
        objs_raw = self.env.identify_interactive_objects(ob)
        objs_raw = list(itertools.chain.from_iterable(objs_raw))
        objs_raw.sort()
        objs = []
        obj_ids = []
        for obj in objs_raw:
            if obj[:self.max_word_len] in self.noun_words:
                obj_id = self.word2id[obj[:self.max_word_len]]
                if obj_id not in obj_ids:
                    objs.append(obj)
                    obj_ids.append(obj_id)

        return objs, obj_ids

    def _generate_all_template_actions(self, objs, obj_ids):
        return self.act_gen.generate_template_actions(objs, obj_ids)

    def _check_valid_action_parallel(self, candidate_actions):
        chunks = [
            [act for act_id, act in enumerate(candidate_actions)
             if act_id % self.env_num == i]
            for i in range(self.env_num)]
        state = self.env.get_state()

        for i in range(self.env_num):
            self.remotes[i].send((state, chunks[i]))

        results = [remote.recv() for remote in self.remotes]

        flatten = lambda l: [item for sublist in l for item in sublist]

        keys = list(set(flatten([out.keys() for out in results])))
        keys.sort()

        # merge key-value pairs and sort by
        valid_actions = [
            flatten([results[i][key] for i in range(self.env_num)])
            for key in keys]
        for v in valid_actions:
            v.sort(key=lambda x: (x.template_id, tuple(x.obj_ids)))
        valid_actions = [[v[0]] for v in valid_actions]
        return valid_actions

    # the serial version copied from jericho, as parallel target
    def _check_valid_action_serial(self, candidate_actions):
        diff2acts = defaultdict(list)
        orig_score = self.env.get_score()
        state = self.env.get_state()
        for act in candidate_actions:
            self.env.set_state(state)
            if isinstance(act, defines.TemplateAction):
                obs, rew, done, info = self.env.step(act.action)
            else:
                obs, rew, done, info = self.env.step(act)

            if self.env.emulator_halted():
                self.env.reset()
                continue

            if info['score'] != orig_score or done or self.env.world_changed():
                # Heuristic to ignore actions with side-effect of taking items
                if '(Taken)' in obs:
                    continue
                diff = self.env._get_world_diff()
                diff2acts[diff].append(act)
        # different treatment for return structure
        keys = list(diff2acts.keys())
        keys.sort()
        valid_acts = [diff2acts[key] for key in keys]
        for v in valid_acts:
            v.sort(key=lambda x: (x.template_id, tuple(x.obj_ids)))
        valid_acts = [[v[0]] for v in valid_acts]
        self.env.set_state(state)
        return valid_acts

    def _get_action_on_current_state(self, state_hash, ob, parallel,
                                     compute_actions):
        valid_ao = None
        if state_hash in self.state2valid_acts:
            valid_ao = self.state2valid_acts[state_hash]
        # valid_ao = self.state2valid_acts[
        #     state_hash] if state_hash in self.state2valid_acts else None
        if valid_ao is None and compute_actions:
            # Identifies objects in the current location and inventory
            # that are likely to be interactive.
            # the returned obj may be a list, first one is the noun,
            # followed by some adj.,
            objs, obj_ids = self._identify_objects_on_current_state(ob)
            acts = self._generate_all_template_actions(objs, obj_ids)

            if parallel:
                valid_acts = self._check_valid_action_parallel(acts)
            else:
                valid_acts = self._check_valid_action_serial(acts)
            # also compute what actions are removed
            # v_acts = [a for subacts in valid_acts for a in subacts]
            # invalid_act = [act for act in acts if act not in v_acts]
            invalid_act = []
            valid_ao = (valid_acts, acts, objs, invalid_act)
            self.state2valid_acts[state_hash] = valid_ao

        if state_hash in self.state2valid_acts:
            return self.state2valid_acts[state_hash]
        return [], [], [], []

    def step(self, action, confidence=0, parallel=True, compute_actions=True):
        ob, reward, done, info = self.env.step(action)
        world_changed = self.env.world_changed()
        info['world_changed'] = world_changed
        # Initialize with default values
        look = 'unknown'
        inv = 'unknown'
        if not done:
            try:
                # it is still possible the actions' effect is stochastic,
                # so the world changed is also a random variable
                # reduce the effect of randomness by repeating
                # (similar to some ATARI games)
                if not world_changed and confidence > 0:
                    save = self.env.get_state()
                    for _ in range(confidence):
                        _, _, _, _ = self.env.step(action)
                        world_changed = self.env.world_changed()
                        if world_changed:
                            break
                    self.env.set_state(save)

                info['world_changed'] = world_changed
                self.steps += 1

                state_hash = _get_world_state_hash(self.env)
                save = self.env.get_state()
                look, _, _, _ = self.env.step('look')
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                self.env.set_state(save)
                # Find Valid actions
                act_info = self._get_action_on_current_state(
                    state_hash, ob, parallel, compute_actions)

                info['valid_act'] = act_info[0]
                info['act'] = act_info[1]
                info['objs'] = act_info[2]
                info['invalid_act'] = act_info[3]

            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean_obs(
                    ob), done, info))
                info['valid_act'] = []
                info['act'] = []
                info['objs'] = []
                info['invalid_act'] = []
                done = True

        else:
            info['valid_act'] = []
            info['act'] = []
            info['objs'] = []
            info['invalid_act'] = []

        if self.step_limit and self.steps >= self.step_limit:
            done = True

        ob = (clean_obs(look) + '|' + clean_obs(inv) + '|' + clean_obs(ob) +
              '|' + clean_obs(action))
        return ob, reward, done, info

    def reset(self, parallel=True, compute_actions=True):
        initial_ob, info = self.env.reset()
        save = self.env.get_state()
        self.steps = 0
        look, inv = '', ''
        try:
            state_hash = _get_world_state_hash(self.env)
            look, _, _, _ = self.env.step('look')
            self.env.set_state(save)
            inv, _, _, _ = self.env.step('inventory')
            self.env.set_state(save)

            # compute valid state for initial obs
            act_info = self._get_action_on_current_state(
                state_hash, initial_ob, parallel, compute_actions)

            if len(act_info[0]) == 0:
                done = True
            info['valid_act'] = act_info[0]
            info['act'] = act_info[1]
            info['objs'] = act_info[2]
            info['invalid_act'] = act_info[3]
            info['world_changed'] = True

        except RuntimeError:
            print('RuntimeError: {}, Info: {}'.format(initial_ob, info))
            info['valid_act'] = []
            info['act'] = []
            info['objs'] = []
            info['invalid_act'] = []
            info['world_changed'] = True

            self.steps = self.step_limit

        initial_ob = clean_obs(look) + '|' + clean_obs(inv) + '|' + clean_obs(
            initial_ob) + '|' + clean_obs('look')
        return initial_ob, info

    def align_action_on_current_state(self, target_action, action_groups):
        state = self.env.get_state()
        obs, rew, done, info = self.env.step(target_action)
        target_diff = self.env._get_world_diff()
        orig_score = info['score']
        self.env.set_state(state)
        # print('act_group', action_groups)
        for id, act in enumerate(action_groups):
            # the first action as the archetype action
            act = act[0]
            template_id = act[1]
            obj1_id = act[2][0] if len(act[2]) > 0 else None
            obj2_id = act[2][1] if len(act[2]) > 1 else None

            #
            act_str = self.tmpl_to_str(template_id, obj1_id, obj2_id)
            obs, rew, done, info = self.env.step(act_str)
            # if self.env.emulator_halted():
            #     self.env.reset()
            #     continue
            diff = None
            if info['score'] != orig_score or done or self.env.world_changed():
                # if '(Taken)' in obs:
                #     continue
                diff = self.env._get_world_diff()

            if diff == target_diff:
                self.env.set_state(state)
                return id

            self.env.set_state(state)

        return -1

