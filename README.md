# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## MONTE CARLO CONTROL ALGORITHM
1. Initialize Q(s, a) arbitrarily for all state-action pairs
2. Initialize returns(s, a) to empty for all state-action pairs
3. Initialize policy π(s) to be arbitrary (e.g., ε-greedy)

4. For each episode:<BR>
   a. Generate an episode using policy π<BR>
   b. For each state-action pair (s, a) in the episode:<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; i.   Calculate G (return) for that (s, a) pair<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ii.  Append G to returns(s, a)<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; iii. Calculate the average of returns(s, a)<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; iv.  Update Q(s, a) using the average return<BR>
   c. Update policy π(s) based on Q(s, a)
## MONTE CARLO CONTROL FUNCTION
```
Developed By:P.Siva Naga Nithin
Reg.NO:212221240037
```
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

![image](https://github.com/user-attachments/assets/6bc0a4fd-99ba-4f72-8b95-b8558f7ef08d)

 ### Action - Value Function

![image](https://github.com/user-attachments/assets/f3d32f60-f60f-4c14-a79b-90ad2f1fb916)


### Policy

![exp5 3](https://github.com/user-attachments/assets/f82b9b95-6b42-4145-a3f9-43f6daa16d28)

### Success Percentage of Policy

![image](https://github.com/user-attachments/assets/59a11f16-5802-464d-a080-d428a92c8434)


## RESULT:

Thus a Python program is developed to find the optimal policy for the given RL environment using the Monte Carlo algorithm.
