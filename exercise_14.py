import numpy as np
import matplotlib.pyplot as plt

from gym import spaces


class QLearner:
    def __init__(self, n_states: int, n_actions: int, epsilon: float, alpha: float):
        self.n_actions = n_actions
        self.q_table = np.zeros([n_states, n_actions])

        # Hyperparameters
        self.alpha = alpha
        self.gamma = 0.9
        self.epsilon = epsilon

    def act(self, state, train=True):
        if np.random.uniform(0, 1) < self.epsilon and train is True:
            action = np.random.randint(0, self.n_actions)  # Explore action space
        else:
            action = np.argmax(self.q_table[state, :])  # Exploit learned values
            # action = np.random.choice(np.flatnonzero(self.q_table[state, :] == self.q_table[state, :].max()))
        return action

    def train(self, state, next_state, action, reward, done):

        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * (1 - done) * next_max
        )

        self.q_table[state, action] = new_value


class PendulumEnv:
    def __init__(self):
        self.dt = 0.02
        self.g = 9.81
        self.m = 2.0
        self.l = 1.0

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.period = 0
        self.state = np.array([0.0, 0.0])

        self.action_force_mapping = {1: -4.0, 0: 0.0, 2: 4.0}

    def step(self, action):
        self.period += 1

        # get force from action
        force = self.action_force_mapping[action]

        # get new state
        theta, omega = self.state
        new_omega = (
            omega
            + (
                self.g / self.l * np.sin(theta)
                + force / self.m
                + np.random.normal(loc=0.0, scale=3.0) / self.m
            )
            * self.dt
        )
        new_theta = theta + new_omega * self.dt
        self.state = np.array([new_theta, new_omega])

        # get reward
        reward = -1.0 if np.abs(new_theta) > np.pi / 4.0 else 0.0

        # get done
        done = (
            True if np.abs(new_theta) > np.pi / 4.0 or self.period >= 1000.0 else False
        )

        return self._get_obs(self.state), reward, done, {}

    def reset(self):
        self.period = 0
        self.state = np.array([0.0, 0.0])

        return self._get_obs(self.state)

    def _get_obs(self, state):
        return state


class DiscretePendulum(PendulumEnv):
    def __init__(self, state_dimension=50):
        super().__init__()
        self.state_dimension = state_dimension

    def _get_obs(self, state):
        theta, omega = state

        omega_bin = np.digitize(
            omega,
            bins=np.linspace(
                -3.0 + 6 / (self.state_dimension - 1), 3.0, (self.state_dimension - 1)
            ),
            right=True,
        )
        theta_bin = np.digitize(
            theta,
            bins=np.linspace(
                -np.pi / 4.0 + np.pi * 2 / (4 * (self.state_dimension - 1)),
                np.pi / 4.0,
                (self.state_dimension - 1),
            ),
            right=True,
        )

        return self.state_dimension * omega_bin + theta_bin


def simulate_uncontrolled():

    # simulate 10 uncontrolled episodes
    env = PendulumEnv()
    angles = []

    for _ in range(10):
        angle = []
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(1)  # action 1 means do nothing
            angle.append(obs[0])

        angles.append(angle)

    # plot results
    for seq in np.array(angles):
        plt.plot(seq)
    plt.xlabel("Period")
    plt.ylabel("Theta")
    plt.show()


def check_discretization():

    # discretize the state space
    env = DiscretePendulum()
    continuous_states = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[np.square(np.pi / 8.0), 0.0], [0.0, np.square(3.0 / 2.0)]],
        size=100000,
    )
    discrete_states = np.array([env._get_obs(state) for state in continuous_states])

    # count occurances
    unique, counts = np.unique(discrete_states, return_counts=True)
    plt.plot(unique, counts)
    plt.xlabel("State index")
    plt.ylabel("Count")
    plt.show()

    # plot in 3d
    count_dict = dict(zip(unique, counts))
    possible_discrete_states = range(50 * 50)
    discrete_counts = [
        count_dict.get(discrete_state, 0) for discrete_state in possible_discrete_states
    ]
    discrete_counts = np.array(discrete_counts).reshape((50, 50))

    x, y = np.mgrid[:50, :50]
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, c=discrete_counts)
    fig.colorbar(im, ax=ax)
    plt.ylabel("theta")
    plt.xlabel("omega")
    plt.show()


def train_agent(epsilon=0.0, learning_rate=0.5, state_dimension=50):

    # run with greedy policy
    agent = QLearner(
        n_states=state_dimension * state_dimension,
        n_actions=3,
        epsilon=epsilon,
        alpha=learning_rate,
    )
    env = DiscretePendulum(state_dimension)
    step_numbers = []
    for episode in range(2000):
        cnt = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.train(obs, next_obs, action, reward, done)

            obs = next_obs
            cnt += 1
        step_numbers.append(cnt)

    # plot number of steps
    plt.plot(step_numbers)
    plt.title("e:{} n:{} D:{}".format(epsilon, learning_rate, state_dimension))
    plt.show()

    # plot greedy function
    possible_discrete_states = range(state_dimension * state_dimension)
    greedy_policy = [
        agent.act(discrete_state, train=False)
        for discrete_state in possible_discrete_states
    ]
    x, y = np.mgrid[:state_dimension, :state_dimension]
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, c=greedy_policy)
    fig.colorbar(im, ax=ax)
    plt.ylabel("theta")
    plt.xlabel("omega")
    plt.title("Policy")
    plt.show()

    # plot value function
    possible_discrete_states = range(state_dimension * state_dimension)
    greedy_policy = [
        -np.max(agent.q_table[discrete_state, :])
        for discrete_state in possible_discrete_states
    ]
    x, y = np.mgrid[:state_dimension, :state_dimension]
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, c=-np.log(greedy_policy))
    fig.colorbar(im, ax=ax)
    plt.ylabel("theta")
    plt.xlabel("omega")
    plt.title("-log(-value)")
    plt.show()


if __name__ == "__main__":
    simulate_uncontrolled()
    check_discretization()
    train_agent(epsilon=0.0, learning_rate=0.5, state_dimension=50)
    train_agent(epsilon=0.0, learning_rate=0.5, state_dimension=10)
    train_agent(epsilon=0.0, learning_rate=0.5, state_dimension=100)
    train_agent(epsilon=0.0, learning_rate=0.5, state_dimension=200)
    train_agent(epsilon=0.0, learning_rate=0.5, state_dimension=2)
    train_agent(epsilon=0.1, learning_rate=0.5, state_dimension=50)
    train_agent(epsilon=0.1, learning_rate=1.0, state_dimension=50)
