import numpy as np
from mpi4py import MPI
from GridKeyEnv import GridWorldKey
from policy import Policy
import plot
import time

comm = MPI.COMM_WORLD


def rollout(env,pi):
    ep_r = 0.0
    observes, actions, rewards = [], [], []
    done = False
    step = 0

    obs = env.reset()  # obs here are unscaled
    while not done:
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[0.001 * step]], axis=1)
        observes.append(obs)

        action = pi.act(obs)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        #reward = reward / self.rew_scale[choose_idx]
        ep_r += reward

        actions.append(action)
        rewards.append(reward)

        if done:
            break
        step += 1
    return np.array(observes), np.array(actions), np.array(rewards),ep_r


if __name__ == '__main__':

    time_start = time.time()

    size = comm.Get_size()
    rank = comm.Get_rank()

    env = GridWorldKey(max_time=3000, n_keys=2, normlize_obs=True, use_nearby_obs=True)

    pi = Policy(name='policy', obs_dim=env.observation_space.shape[0]+1, act_dim=env.action_space.shape[0], n_ways=0, batch_size=20,
           log_path='logfile/process{}'.format(rank))

    for i in range(50):
        ts = time.time()
        if rank==0:
            vars = pi.trpo_net.get_vars()
        else:
            vars = None
        vars = comm.bcast(vars,root=0)
        pi.trpo_net.set_vars(vars)
        te = time.time()

        print('variable copy time cost = {}, w{}'.format(te-ts,rank))

        traj = rollout(env,pi)
        trajs = comm.gather(traj,root=0)
        if rank==0:
            for each in trajs:
                plot.plot('ep_r',each[-1])
                plot.tick('ep_r')
                obs = each[0].reshape(-1, pi.obs_dim)
                acts = each[1].reshape(-1, pi.act_dim)
                rews = each[2]
                pi.update(obs,acts,rews)
                plot.flush('logfile/process{}'.format(rank), verbose=True)

    time_end = time.time()
    print('==============\ntotally time cost = {}, w{}'.format((time_end - time_start),rank))