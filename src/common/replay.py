import collections
import datetime
import io
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class ReplayBuffer(IterableDataset):
    def __init__(self, data_specs, directory, meta_specs=None, length=20, min_len=1, max_len=0, capacity=0,
                 save_episodes=False, prioritize_ends=False, device='cuda'):
        self._data_specs = data_specs
        self._meta_specs = meta_specs  # dmc_like sample array for skill/goal
        self._directory = directory
        self._save_episodes = save_episodes
        self._rollout_len = length  # rollout base len for posterior inference
        self._min_len = min_len
        self._max_len = max_len  # rollout len interval when samping to diversify the inference horizon
        self._capacity = capacity
        self._prioritize_ends = prioritize_ends
        self._device = device
        self._random_state = np.random.RandomState()
        try:
            assert self._min_len <= self._rollout_len <= self._max_len
        except:
            print('Invalid rollout length interval. Defaulting to length: ', self._rollout_len)

        # # sample containers
        # load pre-collected trans, get a empty dict {} if no experience is available
        self._aggregated_eps = load_episodes(self._directory, capacity)
        self._ongoing_eps = collections.defaultdict(lambda: collections.defaultdict(list))  # for running rollouts
        # # running statics
        self._total_num_eps, self._total_num_trans = count_episodes(directory)  # offline data counts
        self._loaded_num_eps = len(self._aggregated_eps)
        self._loaded_num_trans = sum(episode_len(x) for x in self._aggregated_eps.values())

    def __len__(self):
        return self._loaded_num_trans

    @property
    def stats(self):
        return {
            'total_num_eps': self._total_num_eps,
            'total_num_trans': self._total_num_trans,
            'loaded_num_eps': self._loaded_num_eps,
            'loaded_num_trans': self._loaded_num_trans
        }

    def add(self, time_step, meta=None, worker=0):
        episode = self._ongoing_eps[worker]
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            episode[spec.name].append(value)
        if self._meta_specs is not None:
            for spec in self._meta_specs:
                value = meta[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                episode[spec.name].append(value)
        if type(time_step) == dict:
            if time_step['is_last']:
                self.add_episode(episode)
                episode.clear()
        else:
            if time_step.last():
                self.add_episode(episode)
                episode.clear()

    def add_episode(self, episode):
        ep_len = episode_len(episode)
        if ep_len < self._min_len:
            print('Skipping short episode of len {0}.'.format(ep_len))
            return
        ep_idx = self._total_num_eps
        self._total_num_trans += ep_len
        self._loaded_num_trans += ep_len
        self._total_num_eps += 1
        self._loaded_num_eps += 1
        episode = {key: convert_dtype(value) for key, value in episode.items()}
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{ep_idx}_{ep_len}.npz'
        fn = self._directory / eps_fn
        if self._save_episodes:
            fn = save_episode(episode, self._directory, eps_fn)
        self._aggregated_eps[str(fn)] = episode
        self.refresh_fifo()  # refresh buffer if its capacity is full

    def refresh_fifo(self):
        if not self._capacity:
            return
        while self._loaded_num_eps > 1 and self._loaded_num_trans > self._capacity:
            idx, episode = next(iter(self._aggregated_eps.items()))
            self._loaded_num_trans -= episode_len(episode)
            self._loaded_num_eps -= 1
            del self._aggregated_eps[idx]

    def __iter__(self):
        rollout = self.sample_rollout()
        # loop for continuous iteration of the aggregated episodes
        while True:
            chunk = collections.defaultdict(list)
            chunk_len = 0
            while chunk_len < self._rollout_len:
                res_len = self._rollout_len - chunk_len
                res_rollout = {k: v[:res_len] for k, v in rollout.items()}
                rollout = {k: v[res_len:] for k, v in rollout.items()}
                for k, v in res_rollout.items():
                    chunk[k].append(v)
                chunk_len += len(res_rollout['action'])
                if len(rollout['action']) < 1:
                    rollout = self.sample_rollout()  # rollout resample for chunk ensemble
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            chunk['is_terminal'] = chunk['discount'] == 0
            chunk = {k: torch.as_tensor(v, device=self._device) for k, v in chunk.items()}
            yield chunk

    def sample_rollout(self):
        episodes = list(self._aggregated_eps.values())
        episode = self._random_state.choice(episodes)
        ep_len = rollout_len = len(episode['action'])
        # randomize the rollout length
        if self._max_len:
            rollout_len = min(ep_len, self._max_len)
        rollout_len -= np.random.randint(self._min_len)
        rollout_len = max(rollout_len, self._min_len)
        #
        upper_idx = ep_len - rollout_len + 1
        if self._prioritize_ends:
            upper_idx += self._min_len
        start_idx = min(self._random_state.randint(upper_idx), ep_len-rollout_len)
        rollout = {k: convert_dtype(v[start_idx:start_idx+rollout_len]) for k, v in episode.items()
                   if not k.startswith('log_')}
        rollout['is_first'] = np.zeros(len(rollout['action']), bool)
        rollout['is_first'][0] = True  # rollout start index as first timestep
        if self._max_len:
            assert self._min_len <= len(rollout['action']) <= self._max_len
        return rollout


# # util functions # #
def episode_len(episode):
    """"
    subtract -1 because the dummy first transition that is
    arranged for the transition tuple sampling: (s, s`)
    """
    return len(episode['action']) - 1


def load_episodes(directory, capacity=None, load_forward=False):
    """"
    loading the pre-collected episodic experience if available,
    if the provided directory is empty, then return an empty {}
    """
    fns = sorted(directory.glob('*.npz'))
    if capacity:
        num_trans, num_episodes = 0, 0
        ordered_fns = fns if load_forward else reversed(fns)
        for fn in ordered_fns:
            _, _, ep_len = fn.stem.glob('*.npz')
            num_episodes += 1
            num_trans += int(ep_len)  # ep_len in file name
            if num_trans >= capacity:
                break
        if load_forward:
            fns = fns[:num_episodes]
        else:
            fns = fns[-num_episodes:]
    episodes = {}  # experience container
    for fn in fns:
        try:
            with fn.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(fn)}: {e}')
            continue
        episodes[str(fn)] = episode
    return episodes


def save_episode(episode, directory, ep_fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        fn = directory / ep_fn
        with fn.open('wb') as f:
            f.write(bs.read())
    return fn


def count_episodes(directory):
    fns = list(directory.glob('*.npz'))
    total_eps = len(fns)
    total_trans = sum(int(fn.stem.glob('*.npz')[-1]) for fn in fns)
    return total_eps, total_trans


def convert_dtype(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def make_replay_loader(buffer, batch_size):
    return DataLoader(buffer,
                      batch_size=batch_size,
                      drop_last=True, )


if __name__ == '__main__':
    from envs.dmc_envs import make
    from dm_env import specs
    from pathlib import Path

    walker_env = make(name='walker_walk', obs_type='states', frame_stack=1, action_repeat=1, seed=42,)
    time_step = walker_env.reset()
    data_spec = (walker_env.observation_spec(),
                 walker_env.action_spec(),
                 specs.Array((1,), np.float32, 'reward'),
                 specs.Array((1,), np.float32, 'discount'))
    meta_spce = None
    work_dir = Path.cwd()
    test_buffer = ReplayBuffer(data_spec, work_dir/'buffer', length=50, min_len=50, max_len=50, capacity=int(1e6))
    test_buffer.add(time_step, )
    test_loader = make_replay_loader(test_buffer, batch_size=4)
    replay_iter = iter(test_loader)
    for i in range(1000):
        action = walker_env.act_space['action'].sample()
        time_step = walker_env.step(action)
        test_buffer.add(time_step)
        if time_step['is_last']:
            time_step = walker_env.reset()
            test_buffer.add(time_step)
            print('current environment step is {0}, which collects samples {1}'.format(i+1, len(test_buffer)))
    for j in range(2):
        batch = next(replay_iter)
        print(batch['observation'].shape)
