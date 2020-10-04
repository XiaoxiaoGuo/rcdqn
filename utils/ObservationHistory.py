from collections import defaultdict


class ObservationHistory:
    def __init__(self, hist_window):
        self.entity2obs = defaultdict()
        self.obs = []
        self.hist_window = hist_window
        return

    def extract_entity(self, valid_actions):
        return set([
            obj_id for act in valid_actions for obj_id in act[0].obj_ids])

    def retrieve_obs(self, active_entity):
        obs_ids = list(set([
            self.entity2obs[entity] for entity in active_entity
            if entity in self.entity2obs]))

        last_obs_id = len(self.obs) - 1
        while len(obs_ids) < self.hist_window and last_obs_id >= 0:
            if last_obs_id not in obs_ids:
                obs_ids.append(last_obs_id)
            last_obs_id -= 1
        obs_ids.sort()

        if len(obs_ids) > self.hist_window:
            obs_ids = obs_ids[-self.hist_window:]
        return '|'.join([self.obs[obs_id] for obs_id in obs_ids]) + '|'

    def update_history(self, entity_list, observation):
        observation = observation.split('|')[0]
        self.obs.append(observation)
        for entity in entity_list:
            self.entity2obs[entity] = len(self.obs) - 1
        return

    def reset(self):
        self.entity2obs.clear()
        self.obs.clear()
        return
