import numpy as np
import matplotlib.pyplot as plt

def simulate_birth_and_death(start, brate, drate, max_time=float("inf")):
    """Simulate the birth and death process outlined in the text.

    :start: Nonnegative state to begin at.
    :drate: Nonnegative death rate.
    :brate: Nonnegative birth rate. (`drate + brate` must be positive.)
    :returns: Generator of (time, state) tuples, ending at (time, 0).

    """
    state = start
    time = 0

    while state > 0 and time < max_time:
        yield (time, state)
        # Numpy uses our 1/β exponential distribution. Like a sane person.
        time += np.random.exponential(1 / state / (drate + brate))

        random = np.random.uniform()

        # Move up with probability b/(b + d).
        # P(u <= b / (b + d)) = b / (b + d) for uniform u on [0, 1].
        if random <= brate / (brate + drate):
            state += 1
        else:
            state -= 1

    # Give the time that we went extinct at. (Or just stopped.)
    yield (time, state)

def plot_birth_death(start, brate, drate, num=1, continuous=False,
                        show_eet=False):
    """Plot the birth and death process.

    :start: Initial population value.
    :brate: Birth rate.
    :drate: Death rate.
    :num: Number of stochastic realizations to plot.
    :continuous: Flag to plot an exponential curve with rate (brate - drate).
    """

    for run in range(num):
        results = np.array(list(simulate_birth_and_death(start, brate, drate)))
        times = results[:, 0]
        states = results[:, 1]
        plt.step(times, states, where="post")

    if show_eet:
        eet = sum(1 / (k + 1) for k in range(start)) / drate
        plt.vlines(eet, 0, start, label="Expected death process extinction time")

    if continuous:
        # Plot exponential decay for comparison.
        xs = np.linspace(0, max(times[-1], eet))
        ys = start * np.exp((brate - drate) * xs)
        plt.plot(xs, ys, label="Exponential population")

    plt.legend()

def stacked_death_plot(expected_extinction, top):
    """
    Plot death processes with different intial values but identical expected
    extinction time.

    This is essentially just an interesting drawing.

    (Question: What is the probability that, of `top` plots, k <= `top` of them
    will become extinct at exactly the same time? Should be answerable by
    finding the p.d.f. p_0(t) for the processes, then multiplying them together
    for some t_0. That won't answer the probability of _any_ t_0, though, just
    a particular one. The function p_0(t) can be obtained through the matrix
    exponential, since the state space is finite for the death process.)

    The drawing is `top` death processes plotted together, where the kth
    process begins with initial value k, and has expected extinction time
    `expected_extinction`.

    :expected_extinction: Expected extinction time of the process.
    :top: Number of plots, and also the largest initial population value.
    :returns: Nothing.

    """

    plot_death = lambda start, drate: plot_birth_death(start, 0, drate)
    harmonic = lambda n: sum(1/k for k in range(1, n + 1))

    for start in range(1, top + 1):
        # Expected Extinction Time (EET) = H_n / drate,
        # so drate = H_n / EET.
        Hn = harmonic(start)
        drate = Hn / expected_extinction
        brate = 0
        results = np.array(list(simulate_birth_and_death(start, brate, drate)))
        ts, xs = results.T

        plt.step(ts, xs, where="post", alpha=.6, lw=3, label=str(start))
