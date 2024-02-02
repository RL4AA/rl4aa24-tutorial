import numpy as np
from gymnasium.vector import SyncVectorEnv as SyncVectorEnv_
from gymnasium.vector.utils import concatenate, create_empty_array


class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self, env_fns, observation_space=None, action_space=None, **kwargs):
        super(SyncVectorEnv, self).__init__(
            env_fns,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )
        for env in self.envs:
            if not hasattr(env.unwrapped, "reset_task"):
                raise ValueError(
                    "The environment provided is not a "
                    "meta-learning environment. It does not have "
                    "the method `reset_task` implemented."
                )

    def reset_task(self, task):
        for env in self.envs:
            env.unwrapped.reset_task(task)

    def step_wait(self):
        self._actions = list(self._actions)
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = len(self._actions)
        rewards = np.zeros((num_actions,), dtype=np.float_)
        for i, env in enumerate(self.envs):
            if self._terminateds[i] or self._truncateds[i]:
                continue

            action = self._actions[j]
            (
                observation,
                rewards[j],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)
            batch_ids.append(i)

            if not self._terminateds[i] and not self._truncateds[i]:
                observations_list.append(observation)
                infos.append(info)
            j += 1
        assert num_actions == j

        if observations_list:
            observations = create_empty_array(
                self.single_observation_space, n=len(observations_list), fn=np.zeros
            )
            # concatenate(observations_list,
            #             observations,
            #             self.single_observation_space)
            concatenate(
                self.single_observation_space, observations_list, observations
            )  # (space, items, out)
        else:
            observations = None
        return (
            observations,
            rewards,
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            {"batch_ids": batch_ids, "infos": infos},
        )
