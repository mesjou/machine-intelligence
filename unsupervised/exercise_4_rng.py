# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# exercise 1


def inverse_laplace(x: float, mean: float, b: float):
    if x >= 0.5:
        return mean - b * np.log(2 - 2 * x)
    else:
        return b * np.log(2 * x) + mean


def laplace_density(x: float, mean: float, b: float):
    return 1 / (2 * b) * np.exp(-np.abs(x - mean) / b)


def sample_from_laplace(mean: float = 1.0, b: float = 2.0, n: int = 500):
    uniform_samples = np.random.uniform(0, 1, size=n)
    return np.array([inverse_laplace(xi, mean, b) for xi in uniform_samples])


if __name__ == "__main__":
    # sample
    sampled_density = sample_from_laplace(1.0, 2.0, n=500)
    evaluate_at = np.arange(min(sampled_density), max(sampled_density), 0.01)
    true_density = [laplace_density(xi, 1.0, 2.0) for xi in evaluate_at]

    # plot
    plt.hist(sampled_density, bins=40, density=True)
    plt.plot(evaluate_at, true_density)
    plt.show()
