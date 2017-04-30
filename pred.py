#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program simulates realizations of a stochastic predator-prey model. The
model is described by Linda Allen in "An Introduction to Stochastic Processes
with Applications to Biology."

The stochastic work is done in the function stochastic_pred_prey(). The
deterministic version is included for comparison in deterministic_pred_prey().

Copyright © 2017 Robert Dougherty-Bliss

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def stochastic_pred_prey(prey_start, pred_start, params, tend):
    """Simulate the stochastic predator-prey model given by Allen.

    :prey_start: Initial prey population.
    :pred_start: Initial predator population.
    :params: 2x2 matrix:
        [[ prey_birth   prey_death
           pred_birth   pred_death ]]

    :returns: Generator of (prey, pred, time) tuples.

    At each step the process has five possibilities:
      - prey +=1 (lam0)
      - pred +=1 (lam1)
      - prey -=1 (mu0)
      - pred -=1 (mu1)

    They are chosen according to the following probabilities:
        P(prey += 1) = lam0 / r
        P(pred += 1) = lam1 / r
        P(prey -= 1) =  mu0 / r
        P(pred -= 1) =  mu1 / r,
    where
        r = lam0 + lam1 + mu0 + m1.

    This process is used by Allen in "An Introduction to Stochastic Processes
    with Applications to Biology."
    """
    yield (prey_start, pred_start, 0)

    prey_birth = params[0, 0]
    prey_death = params[0, 1]
    pred_birth = params[1, 0]
    pred_death = params[1, 1]

    equilibrium_prey = prey_death / pred_birth
    equilibrium_pred = prey_birth / prey_death
    print("Equilibrium: ({}, {})".format(equilibrium_prey, equilibrium_pred))

    prey = prey_start
    pred = pred_start
    t = 0
    step = 0

    while (prey > 0 or pred > 0) and t < tend:
        step += 1
        if step % 100 == 0:
            print((prey, pred, t))

        # Prey birth rate.
        lam0 = prey_birth * prey
        # Predator birth rate.
        lam1 = pred_birth * prey * pred

        # Prey death rate.
        mu0 = prey_death * prey * pred
        # Predator death rate.
        mu1 = pred_death * pred

        # The below "random selection" process can be shown to have the
        # following probabilities:
        #       P(pred += 1) = lam0 / r
        #       P(pred += 1) = lam1 / r
        #       P(pred += 1) =  mu0 / r
        #       P(pred += 1) =  mu1 / r.

        r = lam0 + lam1 + mu0 + mu1
        u = np.random.uniform()
        marks = np.cumsum([lam0, lam1, mu0, mu1]) / r

        if u < marks[0]:
            prey += 1
        elif marks[0] <= u < marks[1]:
            pred += 1
        elif marks[1] <= u < marks[2]:
            prey -= 1
        elif marks[2] <= u < marks[3]:
            pred -= 1
        else:
            # Since marks[-1] = r / r = 1 and np.random.uniform() samples on
            # [0, 1), we shouldn't reach this.
            pass

        # According to Allen, the holding times are exponentially distributed
        # with mean r. For us and numpy, this means an exponential distribution
        # with mean 1/r.

        t += np.random.exponential(1/r)

        yield (prey, pred, t)

def deterministic_pred_prey(prey_start, pred_start, params, tend, npoints=1000):
    """
    Return points of solutions to the Lotka-Volterra predator-prey equations.

    :prey_start: Initial prey population.

    :pred_start: Initial predator population.

    :params: 2x2 numpy array of parameters of the following form:
        [[ prey_birth_rate      prey_death_rate
           predator_birth_rate  predator_death_rate ]]

    :tend: Time to integrate to.

    :npoints: Number of solution points to return.

    :returns: Tuple of arrays (ts, ys), where `ts` is a a `tend` x 1 array of
              time points, and `ys` is a `tend` x 2 array of population points,
              with prey being the first component and predators the second.

    """
    def dxdt(state, time, params):
        prey, pred = state
        prey_birth = params[0, 0]
        prey_death = params[0, 1]
        pred_birth = params[1, 0]
        pred_death = params[1, 1]

        dprey = prey_birth * prey - prey_death * pred * prey
        dpred = pred_birth * prey * pred - pred_death * pred

        return (dprey, dpred)

    ts = np.linspace(0, tend, npoints)
    ys = scipy.integrate.odeint(dxdt, [prey_start, pred_start], ts, (params,))

    return (ts, ys)

def plot_deterministic_pred_prey(*args, **kwargs):
    """Plot solutions to the Lotka-Volterra predator-prey equations.

    Phase plane is plotted on figure 0, and the time plot is on figure 1.

    :*args: Extra arguments to pass to deterministic_pred_prey().
    :**kwargs: Keyword arguments to pass to deterministic_pred_prey().
    :returns: Nothing.

    """
    ts, ys = deterministic_pred_prey(*args, **kwargs)

    prey, pred = ys[:, 0], ys[:, 1]

    plt.figure(0)
    plt.plot(prey, pred)

    plt.figure(1)
    plt.plot(ts, prey, label="Deterministic Prey")
    plt.plot(ts, pred, label="Deterministic Predator")
    plt.legend()

def step_plot_pred_prey(*args, **kwargs):
    """Plot a step plot of solutions to the stochastic predator-prey process.

    Phase plane is plotted on figure 0, and the time plot is on figure 1.

    :*args: Extra arguments to pass to stochastic_pred_prey().
    :**kwargs: Keywork arguments to pass to stochastic_pred_prey().
    :returns: Nothing.
    """
    res = stochastic_pred_prey(*args, **kwargs)
    X, Y, T = np.array(list(res)).T

    plt.figure(0)
    plt.step(X, Y, where="post")
    plt.plot(X[0], Y[0], "ko", ms=10)
    plt.plot(X[-1], Y[-1], "ro", ms=10)
    plt.xlabel("Prey")
    plt.ylabel("Predators")
    plt.title("Stochastic Predator-Prey Realization")

    plt.figure(1)
    plt.step(T, X, label="Prey", where="post")
    plt.step(T, Y, label="Predators", where="post")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()

def compare_models(*args, **kwargs):
    """Plot both the deterministic and stochastic models at once.

    :*args: Arguments to pass to plot_deterministic_pred_prey() and
            step_plot_pred_prey().
    :**kwargs: Keyword arguments to pass to plot_deterministic_pred_prey() and
               step_plot_pred_prey().
    :returns: Nothing.

    """
    plot_deterministic_pred_prey(*args, **kwargs)
    step_plot_pred_prey(*args, **kwargs)

def _main():
    # This set of parameters will produce cycles and extinctions depending on the
    # intial conditions.
    cycle_params = np.array(
        [[10, .01],
         [.01, 10]]
    )

    prey_start = 100
    pred_start = 30

if __name__ == "__main__":
    _main()
