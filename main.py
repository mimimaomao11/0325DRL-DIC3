import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# problem setup
T = 10000
runs = 300

true_means = np.array([0.8, 0.7, 0.5])
optimal_mean = np.max(true_means)


def ab_testing():
    rewards = np.zeros(T)

    # A/B test phase (ignore C)
    for t in range(1000):
        rewards[t] = np.random.binomial(1, true_means[0])

    for t in range(1000, 2000):
        rewards[t] = np.random.binomial(1, true_means[1])

    meanA = rewards[:1000].mean()
    meanB = rewards[1000:2000].mean()

    best = 0 if meanA > meanB else 1

    for t in range(2000, T):
        rewards[t] = np.random.binomial(1, true_means[best])

    return rewards


def optimistic_init():
    Q = np.ones(3)  # optimistic initial values
    N = np.zeros(3)

    rewards = np.zeros(T)

    for t in range(T):
        a = np.argmax(Q)

        r = np.random.binomial(1, true_means[a])

        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t] = r

    return rewards


def epsilon_greedy(eps=0.1):
    Q = np.zeros(3)
    N = np.zeros(3)

    rewards = np.zeros(T)

    for t in range(T):

        if np.random.rand() < eps:
            a = np.random.randint(3)
        else:
            a = np.argmax(Q)

        r = np.random.binomial(1, true_means[a])

        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t] = r

    return rewards


def softmax(tau=0.1):
    Q = np.zeros(3)
    N = np.zeros(3)

    rewards = np.zeros(T)

    for t in range(T):

        probs = np.exp(Q / tau)
        probs /= probs.sum()

        a = np.random.choice(3, p=probs)

        r = np.random.binomial(1, true_means[a])

        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t] = r

    return rewards


def ucb(c=2):
    Q = np.zeros(3)
    N = np.zeros(3)

    rewards = np.zeros(T)

    for t in range(T):

        if 0 in N:
            a = np.argmin(N)
        else:
            bonus = c * np.sqrt(np.log(t + 1) / N)
            a = np.argmax(Q + bonus)

        r = np.random.binomial(1, true_means[a])

        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t] = r

    return rewards


def thompson():
    alpha = np.ones(3)
    beta = np.ones(3)

    rewards = np.zeros(T)

    for t in range(T):

        samples = np.random.beta(alpha, beta)
        a = np.argmax(samples)

        r = np.random.binomial(1, true_means[a])

        alpha[a] += r
        beta[a] += 1 - r

        rewards[t] = r

    return rewards


def simulate(method):

    regret = np.zeros(T)

    for _ in range(runs):

        rewards = method()

        optimal_rewards = np.random.binomial(1, optimal_mean, T)

        regret += np.cumsum(optimal_rewards - rewards)

    regret /= runs

    return regret


regrets = {
    "A/B Testing": simulate(ab_testing),
    "Optimistic": simulate(optimistic_init),
    "Epsilon-Greedy": simulate(epsilon_greedy),
    "Softmax": simulate(softmax),
    "UCB": simulate(ucb),
    "Thompson": simulate(thompson),
}


plt.figure()

for name, reg in regrets.items():
    plt.plot(reg, label=name)

plt.yscale("log")

plt.xlabel("Round Index")
plt.ylabel("Cumulative Expected Regret")
plt.title("Simulated Bandit Performance (3 arms)")

plt.legend()

plt.show()