import numpy as np
import spacy
from past.builtins import xrange


class EnvTextEncoder:
    def __init__(self, env, state_word2id):
        self.word2id = state_word2id
        self.id2template = env.get_id2template()
        self.template2id = env.get_template2id()
        self.id2obj = env.get_id2act_word()
        self.max_word_length = env.get_max_word_len()
        # en / en_core_web_sm
        self.nlp_pipe = spacy.load('en_core_web_sm')
        self.id2word = {v: k for k, v in self.word2id.items()}
        # reduce unmatched vocabs via prefix matching
        words = [self.id2word[i] for i in range(len(self.id2word))]
        values = self.id2obj.values()
        self.obj2word_id = {
            v: self.word2id[v] for v in values if v in self.word2id}
        self.obj2word_id.update(
            {obj: next((self.word2id[word]
                        for word in words
                        if obj == word[:self.max_word_length]),
                       self.word2id['<OOV>'])
             for obj in [v for v in values if v not in self.word2id]})
        self.obj2word_id['<OOV>'] = self.word2id['<OOV>']

    def get_template_size(self):
        return len(self.id2template)

    def get_act_vocab_size(self):
        return len(self.id2obj)

    def encode_actions(self, actions, objects, state_ids):
        actions = [a[0] for a in actions]
        templates = [self.id2template[a[1]] for a in actions]

        def extend_objects(prefix):
            return next((obj
                         for obj in objects
                         if prefix == obj[:self.max_word_length]),
                        prefix)

        obj1 = [extend_objects(self.id2obj[a[2][0]])
                if len(a[2]) > 0 else '<ACT>' for a in actions]
        obj2 = [extend_objects(self.id2obj[a[2][1]])
                if len(a[2]) > 1 else '<ACT>' for a in actions]

        obj1_ids = [self.word2id[obj]
                    if obj in self.word2id else self.word2id['<OOV>']
                    for obj in obj1]
        obj2_ids = [self.word2id[obj]
                    if obj in self.word2id else self.word2id['<OOV>']
                    for obj in obj2]

        template_ids = []
        template_tokens = []

        for template in templates:
            doc = self.nlp_pipe(template)
            text_tokens = [token.text.lower()
                           for token in doc if token.text.lower() != ' ']
            text_tokens = [token
                           if token in self.obj2word_id else '<OOV>'
                           for token in text_tokens]
            template_tokens.append(text_tokens)
            template_ids.append([self.obj2word_id[token]
                                 for token in text_tokens])

        # the last position of the first objects
        # Note: sometimes, the object may not be found in
        # the observation texts, e.g. the obj_id is <OOV>
        # old one:
        # obj1_pos = [next((i for i in xrange(len(state_ids)-1, -1, -1)
        # if state_ids[i] == obj_id), len(state_ids)-1)
        #             for obj_id in obj1_ids]
        # the first position of the second objects
        # obj2_pos = [next((i for i in xrange(len(state_ids))
        # if state_ids[i] == obj_id), 0)
        #             for obj_id in obj2_ids]

        # new one: average case
        obj1_pos = [[i for i in xrange(len(state_ids) - 1, -1, -1)
                     if state_ids[i] == obj_id]
                    for obj_id in obj1_ids]
        obj2_pos = [[i for i in xrange(len(state_ids))
                     if state_ids[i] == obj_id]
                    for obj_id in obj2_ids]

        for i in range(len(obj1_pos)):
            if len(obj1_pos[i]) == 0:
                obj1_pos[i].append(0)
        for i in range(len(obj2_pos)):
            if len(obj2_pos[i]) == 0:
                obj2_pos[i].append(len(state_ids) - 1)

        # template_tokens/template_ids:
        # N x L, the word ids of the templates,
        #           N is the number of templates, L is the maximum length
        # obj1_pos: N, the pos of the first noun of the template in state,
        #           could be extended to N x M, m is the multi-positions
        # obj2_pos: N, the pos of the second noun of the template in state
        if len(template_ids) == 0:
            template_ids.append([self.word2id['<ACT>']])
            obj1_pos = [[0]]
            obj2_pos = [[0]]
        return template_ids, obj1_pos, obj2_pos, actions

    def encode_observation(self, observation, ignore_action=False):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS',
                  'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            observation = observation.replace(rm, '')

        observation = observation.replace('.\n', '.')
        observation = observation.replace('. \n', '.')
        observation = observation.replace('\n', '.')
        observation = observation.split('|')
        if ignore_action:
            observation = observation[:-1]

        # flatten_tokens = ['<ACT>']
        # spacy_ret = [self.word2id['<ACT>']]
        spacy_ret = []
        for obs_desc in observation:
            doc = self.nlp_pipe(obs_desc)
            spacy_ret.append(self.word2id['<ACT>'])
            text_tokens = [token.text.lower()
                           for token in doc if token.text.lower() != ' ']
            text_tokens = [token
                           if token != '..' else '.'
                           for token in text_tokens]
            text_tokens = [token
                           if token in self.word2id else '<OOV>'
                           for token in text_tokens]
            spacy_ret.extend([self.word2id[token] for token in text_tokens])
        spacy_ret.append(self.word2id['<ACT>'])
        return spacy_ret

    def decode_action(self, template_id, obj1_id, obj2_id):
        template_str = self.id2template[template_id]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.id2obj[obj1_id])

        return template_str.replace('OBJ', self.id2obj[obj1_id], 1). \
            replace('OBJ', self.id2obj[obj2_id], 1)


def diff_state_ids(state_ids, target_ids, id2word):
    if not np.array_equal(state_ids, target_ids):
        print('state reps are not identical.')
        print('state ids', [id2word[state_id] for state_id in state_ids])
        print('target ids', [id2word[target_id] for target_id in target_ids])
        return True
    return False


def diff_action_ids(action, target_action, id2word):
    template, obj1, obj2 = action

    target_template, target_obj1, target_obj2 = target_action
    diff = False
    if not np.array_equal(template, target_template):
        print('template reps are not identical.')
        print('template', [[id2word[t] for t in tt] for tt in template])
        print('target template',
              [[id2word[t] for t in tt] for tt in target_template])
        diff = True

    if not np.array_equal(obj1, target_obj1):
        print('obj1 reps are not identical.')
        print('obj1', obj1)
        print('target_obj1', target_obj1)
        diff = True

    if not np.array_equal(obj2, target_obj2):
        print('obj2 reps are not identical.')
        print('obj2', obj2)
        print('obj2 target', target_obj2)
        diff = True

    return diff
