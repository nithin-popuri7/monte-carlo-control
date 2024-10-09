# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## MONTE CARLO CONTROL ALGORITHM
Initialize the state value function V(s) and the policy π(s) arbitrarily.
Generate an episode using π(s) and store the state, action, and reward sequence.
For each state s appearing in the episode:
G ← return following the first occurrence of s
Append G to Returns(s)
V(s) ← average(Returns(s))
For each state s in the episode:
π(s) ← argmax_a ∑_s' P(s'|s,a)V(s')
Repeat steps 2-4 until the policy converges.
Use the function decay_schedule to decay the value of epsilon and alpha.
Use the function gen_traj to generate a trajectory.
Use the function tqdm to display the progress bar.
After the policy converges, use the function np.argmax to find the optimal policy. The function takes the following arguments:
Q: The Q-table.
axis: The axis along which to find the maximum value.

## MONTE CARLO CONTROL FUNCTION
Developed By:P.Siva Naga Nithin
Reg.NO:212221240037
```
import numpy as np
from tqdm import tqdm

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n

    disc = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    def decay_schedule(init_value, min_value, decay_ratio, n):
        return np.maximum(min_value, init_value * (decay_ratio ** np.arange(n))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    def select_action(state, Q, epsilon):
        return np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        # Generate a trajectory
        traj = gen_traj(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)

        for t, (state, action, reward, _, _) in enumerate(traj):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(traj[t:])
            G = np.sum(disc[:n_steps] * traj[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q

    # Calculate the value function and policy
    V = np.max(Q, axis=1)
    pi = {s: np.argmax(Q[s]) for s in range(nS)}

    return Q, V, pi
```

## OUTPUT:

### State - Value Function

![ex5](https://github.com/user-attachments/assets/650bd50e-932a-4140-93bd-9ed784bd8465)

 ### Action - Value Function

![exp5 1](https://github.com/user-attachments/assets/bb4f5d39-b051-487b-88e0-53d297e03896)

### Policy

![exp5 2](https://github.com/user-attachments/assets/7a1077d4-46ff-4897-9355-6ce9a95016d7)

### Success Percentage of Policy

![exp5 3](https://github.com/user-attachments/assets/d7144752-dbb9-4252-aa84-527aaba1e50b)

![exp5 4](https://github.com/user-attachments/assets/02bee0f3-6ecf-43b1-a678-f01215f2c11d)

## RESULT:

Thus a Python program is developed to find the optimal policy for the given RL environment using the Monte Carlo algorithm.
