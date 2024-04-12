"""Generate time series based on the adapted Gillepie algorithm.

@author: Rokas Astrauskas
"""


import random
from typing import Optional

import numpy as np
from numba import njit  # type: ignore


@njit
def __get_lambda_plus(
    n_agents: int, epsi_0: float, epsi_1: float, X: int, poll: int
) -> float:
    return (n_agents - X) * (epsi_1 + poll)


@njit
def __get_lambda_minus(
    n_agents: int, epsi_0: float, epsi_1: float, X: int, poll: int
) -> float:
    return X * (epsi_0 + (n_agents - poll))


@njit
def generate_series(
    n_polls: int,
    tau: float = 1,
    n_inter: int = 1,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: int = 1000,
    initial_state: int = 500,
    initial_poll: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate time series based on the adapted Gillepie algorithm.

    Input:
        n_polls:
            How many polls should be in the generated series. So, the overall
            duration of the series will be close to `n_polls*tau` (additional
            point for the initial condition will be present).
        tau: (default: 1)
            Time interval between successive polls.
        n_inter: (default: 1)
            This parameter can be used to over- or undersample the series. If
            the value is > 1, then oversampling will be conducted - there will
            be `n_inter` points representing single interval between the
            points. If the value is 1, then single interval will be represented
            by a single point. If the value is <-1, then undersampling will be
            conducted - `-n_inter` polls will be represented by a single data
            point. In other cases, an error will be raised.
        epsi_0: (default: 1)
            Independent transition rate from state 1 to state 0.
        epsi_1: (default: 1)
            Independent transition rate from state 0 to state 1.
        n_agents: (default: 1000)
            Number of agents acting in the system.
        initial_state: (default: 500)
            Initial system state (i.e., X(0)). Recall that system state is a
            number of agents in state 1.
        initial_poll: (default: None)
            Previous poll results (i.e., X(-tau)). If `None` (which is the
            default), then initial_poll will be set to initial_state.
        seed: (default: None)
            If number is supplied, it will be used as a seed for the random
            number generator.

    Output:
        Numpy array with the observed values. Note that the first value will
        always be equal to the initial condition.
    """
    if seed is not None:
        random.seed(seed)

    if initial_poll is None:
        initial_poll = initial_state
          
    # Algorithm 1. Step 1. Set parameters, variables, etc
    t = 0.0  # system clock
    
    current_state = initial_state
    next_poll = initial_state
    last_poll = initial_poll
    k = 1  # index of next reaction

    # prepare for the chosen sampling method
    if n_inter > 1:
        sim_step = tau / n_inter
        history = np.zeros(n_polls * n_inter + 1, dtype=np.int64)
    elif n_inter < -1:
        n_inter = -n_inter
        sim_step = tau * n_inter
        history = np.zeros(n_polls // n_inter + 1, dtype=np.int64)
    elif n_inter == 1:
        sim_step = tau
        history = np.zeros(n_polls + 1, dtype=np.int64)
    else:
        raise ValueError("Bad value for n_inter!")

    index_sample = 1  # index of sampled series
    time_to_next_sample = sim_step
    history[0] = current_state

    event_nr = 0
    # main loop
    while k <= n_polls:
        event_nr += 1
        # Algorithm 1. Steps 2 - 3
        lambda_plus = __get_lambda_plus(
            n_agents, epsi_0, epsi_1, current_state, last_poll
        )
        lambda_minus = __get_lambda_minus(
            n_agents, epsi_0, epsi_1, current_state, last_poll
        )
        lambda_T = lambda_plus + lambda_minus
        
        # Algorithm 1. Step 4
        delta_t = random.expovariate(lambda_T)
        # Algorithm 1. Step 5
        while t + delta_t >= k * tau:
            last_poll = next_poll
            next_poll = current_state
            P = lambda_T * (t + delta_t - k * tau)
            lambda_plus = __get_lambda_plus(
                n_agents, epsi_0, epsi_1, current_state, last_poll
            )
            lambda_minus = __get_lambda_minus(
                n_agents, epsi_0, epsi_1, current_state, last_poll
            )
            lambda_T = lambda_plus + lambda_minus
            delta_t = P / lambda_T
            t = k * tau
            k += 1
        # Algorithm 1. Step 6
        t += delta_t
        # save the state
        while time_to_next_sample < t:
            history[index_sample] = current_state
            index_sample += 1
            time_to_next_sample += sim_step
            if index_sample >= history.size:
                break
        # Algorithm 1. Step 7
        r = random.uniform(0, lambda_T)
        if r <= lambda_plus:
            current_state += 1
        else:
            current_state -= 1

    return history
