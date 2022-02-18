import numpy as np
import matplotlib.pyplot as plt
import scipy


def load_mazes(file_location):
    f = open(file_location, "r")
    mazes_raw = f.readlines()
    # process maze input
    mazes = []
    current_maze = []
    for line in mazes_raw:
        if repr(line) == repr("\n"):
            # skip between two mazes
            if len(current_maze) > 0:
                mazes.append(current_maze)
            current_maze = []
        else:
            maze_row = []
            for el in line:
                if el == " ":
                    maze_row.append(1)
                elif el == "#":
                    maze_row.append(0)
                elif el == "X":
                    maze_row.append(2)
                # strategy specific:
                elif el == "<":
                    maze_row.append(3)
                elif el == ">":
                    maze_row.append(4)
                elif el == "^":
                    maze_row.append(5)
                elif el == "v":
                    maze_row.append(6)
            current_maze.append(maze_row)
    return np.array(mazes)


def get_transition_model_from(maze: np.array) -> np.array:
    n_actions = 4
    transition_model = np.zeros((maze.size, maze.size, n_actions))
    state_idx = np.reshape([range(maze.size)], maze.shape)
    for row_idx in range(maze.shape[0]):
        for col_idx in range(maze.shape[1]):
            if maze[row_idx, col_idx] != 0.0:  # do nothing if state is wall (0)
                state_id = state_idx[row_idx, col_idx]

                # what happens for action = 1, i.e. move right
                if maze[row_idx, col_idx + 1] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx, col_idx + 1]
                transition_model[state_id, new_state_id, 0] = 1

                # what happens for action = 2, i.e. move down
                if maze[row_idx + 1, col_idx] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx + 1, col_idx]
                transition_model[state_id, new_state_id, 1] = 1

                # what happens for action = 3, i.e. move left
                if maze[row_idx, col_idx - 1] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx, col_idx - 1]
                transition_model[state_id, new_state_id, 2] = 1

                # what happens for action = 4, i.e. move up
                if maze[row_idx - 1, col_idx] == 0:
                    new_state_id = state_id
                else:
                    new_state_id = state_idx[row_idx - 1, col_idx]
                transition_model[state_id, new_state_id, 3] = 1

    return transition_model


def get_policy(maze: np.array) -> np.array:
    n_actions = 4
    policy = np.zeros((maze.size, n_actions))
    policy = np.full_like(policy, 1 / n_actions)
    return policy


def get_reward(maze: np.array) -> np.array:
    reward = np.where(maze == 2, 1, 0)
    return reward.flatten()


def plot_value_function(maze, value: np.array):
    value = value.reshape(maze.shape)
    plt.plot(value)
    plt.show()


if __name__ == "__main__":
    # load mazes
    mazes = load_mazes("mazes.txt")
    num_mazes = mazes.shape[0]
    fig = plt.figure(figsize=(20, 10))
    for m_idx, maze in enumerate(mazes[:-1]):
        plt.subplot(1, num_mazes, m_idx + 1)
        plt.title("maze {}".format(m_idx + 1))
        plt.imshow(maze, interpolation="none", cmap="jet")
        plt.axis("off")
    plt.show()

    # Visualize the first 4 mazes
    analytical_Vs = []
    for m_idx, maze in enumerate(mazes[:-1]):
        # implement transition model
        p = get_transition_model_from(maze)

        # c calculate value function
        pi = get_policy(maze)
        r = get_reward(maze)

        # since we choose every action with equal probability we can just take the average over all actions
        p_pi = p.mean(axis=2)
        gamma = 0.9
        V = np.linalg.inv(np.identity(400) - 0.9 * p_pi).dot(r)
        analytical_Vs.append(V)

        plt.subplot(1, num_mazes, m_idx + 1)
        plt.title("value {}".format(m_idx + 1))
        plt.imshow(np.log(V.reshape(maze.shape)), interpolation="none", cmap="jet")
        plt.axis("off")

    plt.show()

    # d estimation via value iteration
    mses = []
    for m_idx, maze in enumerate(mazes[:-1]):
        # implement transition model
        p = get_transition_model_from(maze)

        # c calculate value function
        pi = get_policy(maze)
        r = get_reward(maze)

        # since we choose every action with equal probability we can just take the average over all actions
        p_pi = p.mean(axis=2)
        gamma = 0.9

        mse = []
        V = np.zeros(shape=maze.size)
        for i in range(5000):
            V = r + gamma * p_pi.dot(V)
            if i < 50:
                mse.append((np.square(V - analytical_Vs[m_idx])).mean())
        mses.append(mse)

        plt.subplot(1, num_mazes, m_idx + 1)
        plt.title("value iter {}".format(m_idx + 1))
        plt.imshow(np.log(V.reshape(maze.shape)), interpolation="none", cmap="jet")
        plt.axis("off")

    plt.show()

    # plot mse over iterations
    plt.plot(np.array(mses).transpose())
    plt.show()

    # e different initialisations
    mses = []
    for m_idx, maze in enumerate(mazes[:-1]):
        # implement transition model
        p = get_transition_model_from(maze)

        # c calculate value function
        pi = get_policy(maze)
        r = get_reward(maze)

        # since we choose every action with equal probability we can just take the average over all actions
        p_pi = p.mean(axis=2)
        gamma = 0.9

        mse1 = []
        V = np.random.normal(size=maze.size)
        for i in range(5000):
            V = r + gamma * p_pi.dot(V)
            if i < 50:
                mse1.append((np.square(V - analytical_Vs[m_idx])).mean())
        mses.append(mse1)

        mse2 = []
        V = np.random.normal(size=maze.size)
        for i in range(5000):
            V = r + gamma * p_pi.dot(V)
            if i < 50:
                mse2.append((np.square(V - analytical_Vs[m_idx])).mean())
        mses.append(mse2)

    # plot mse over iterations
    plt.plot(np.array(mses).transpose())
    plt.show()
