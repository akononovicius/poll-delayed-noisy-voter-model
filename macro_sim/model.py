from enum import Enum, auto
from typing import Optional

import numpy as np
from numba import njit  # type: ignore


class SamplingMethod(Enum):
    """Describes sampling mode used during the simulation."""

    oversample = auto()
    undersample = auto()
    propersample = auto()


@njit
def sample_binomial(n: int, p: float, use_normal: bool = False) -> int:
    """Sample from binomial distribution ensuring proper sampling for large N.

    This function is needed because `np.random.binomial` seem to work poorly
    for larger N. With pure `np.random.binomial` the model generates
    trajectories heavily biased towards the lower values.
    """
    # Use `np.random.binomial` with smaller N (the cutoff value is picked
    # "by-the-eye", and could be set more rigorously).
    generation_step = 100
    if n <= generation_step:
        return np.random.binomial(n, p)

    # if desired (though it is not so by default), for larger N, normal
    # approximation could be used (given that p is not close to 0 or 1)
    if use_normal and (n * p > 5) and (n * (1 - p) > 5):
        mean = n * p
        std = np.sqrt(n * p * (1 - p))
        result = -1
        while result <= -0.5 or result >= n + 0.5:
            result = int(np.round(np.random.normal(loc=mean, scale=std)))
        return result

    # otherwise, generate multiple binomial rvs (with the number of trials equal
    # to our cutoff value) and sum them:
    res = np.sum(np.random.binomial(generation_step, p, size=(n // generation_step)))
    # conduct the remaining number of trials to conduct (if needed)
    remainder = n % generation_step
    if remainder > 0:
        res += np.random.binomial(remainder, p)
    return res


@njit
def __step(
    tau: float,
    epsi_0: float,
    epsi_1: float,
    n_agents: int,
    initial_state: int,
) -> int:
    """Do a single simulation step."""
    # precompute certain terms in probability expressions
    epsi_sum = epsi_0 + epsi_1
    prob_stationary = epsi_1 / epsi_sum
    exp_term = np.exp(-epsi_sum * tau)

    # probability for an agent in state 1 to be in state 1 after tau interval
    prob_1_1 = prob_stationary + (1 - prob_stationary) * exp_term
    # probability for an agent in state 0 to be in  state 1 after tau interval
    prob_0_1 = prob_stationary - prob_stationary * exp_term

    # execute "multi"-agent model
    new_state = 0
    if initial_state > 0:
        new_state = new_state + sample_binomial(initial_state, prob_1_1)
    if initial_state < n_agents:
        new_state = new_state + sample_binomial(n_agents - initial_state, prob_0_1)
    return new_state


@njit
def __single_interval(
    current_state: int,
    history: np.ndarray,
    poll_id: int,
    tau: float,
    n_inter: int,
    n_inter_offset: int,
    sim_step: float,
    epsi_eff_0: float,
    epsi_eff_1: float,
    n_agents: int,
    sample_method: SamplingMethod,
) -> tuple[int, int, np.ndarray]:
    """Simulate up to the next poll (single interval)."""
    last_poll = current_state

    if sample_method == SamplingMethod.oversample:
        # simulate and record multiple steps per interval
        start_write_at = (poll_id - 1) * n_inter + 1
        for i in range(n_inter):
            current_state = __step(
                sim_step, epsi_eff_0, epsi_eff_1, n_agents, current_state
            )
            history[start_write_at + i] = current_state
    elif sample_method == SamplingMethod.undersample:
        # simulate every poll, but record only `1` in `n_inter` of them
        current_state = __step(
            sim_step, epsi_eff_0, epsi_eff_1, n_agents, current_state
        )
        if (poll_id % n_inter) == n_inter_offset:
            idx = poll_id // n_inter
            history[idx] = current_state
    else:
        # default mode: simulate only polls (no oversampling),
        #               record every one of them (no undersampling)
        current_state = __step(
            sim_step, epsi_eff_0, epsi_eff_1, n_agents, current_state
        )
        history[poll_id] = current_state
    return last_poll, current_state, history


@njit
def generate_series(
    n_polls: int,
    tau: float = 1,
    n_inter: int = 1,
    n_inter_offset: int = 0,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: int = 1000,
    initial_state: int = 500,
    initial_poll: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate series based on the binomial approximation algorithm.

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
        n_inter_offset: (default: 0)
            Offset value, which is used in the undersampling mode. For example,
            to get even polls only (n_inter = -2, n_inter_offset = 0), to get
            odd polls only (n_inter = -2, n_inter_offset = 1).
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
            default), then a warmup will be conducted. Otherwise no warmup
            will be conducted.
        seed: (default: None)
            If number is supplied, it will be used as a seed for the random
            number generator.

    Output:
        Numpy array with the observed values. Note that the first value will
        always be equal to the initial condition.
    """
    if seed is None:
        seed = np.random.randint(2**20)
    # can't use 'default_rng' because of numba limitations
    np.random.seed(seed)

    # prepare for the chosen sampling method
    sim_step = tau
    if n_inter > 1:
        sample_method = SamplingMethod.oversample
        sim_step = tau / n_inter
        history = np.zeros(n_polls * n_inter + 1)
    elif n_inter < -1:
        sample_method = SamplingMethod.undersample
        n_inter = -n_inter
        history = np.zeros(n_polls // n_inter + 1)
    elif n_inter == 1:
        sample_method = SamplingMethod.propersample
        history = np.zeros(n_polls + 1)
    else:
        raise ValueError("Bad value for n_inter!")

    current_state: int = initial_state
    last_poll: int = -1
    history[0] = current_state

    if initial_poll is None:
        # conduct a single warmup step
        poll_id = 1
        last_poll, current_state, history = __single_interval(
            current_state,
            history,
            poll_id,
            tau,
            n_inter,
            n_inter_offset,
            sim_step,
            epsi_0,
            epsi_1,
            n_agents,
            sample_method,
        )
    else:
        # skip warmup if initial_poll is provided
        last_poll = initial_poll

    # main loop
    initial_poll_id = 1 + (initial_poll is None)
    for poll_id in range(initial_poll_id, n_polls + 1):
        epsi_eff_0 = epsi_0 + (n_agents - last_poll)
        epsi_eff_1 = epsi_1 + last_poll

        last_poll, current_state, history = __single_interval(
            current_state,
            history,
            poll_id,
            tau,
            n_inter,
            n_inter_offset,
            sim_step,
            epsi_eff_0,
            epsi_eff_1,
            n_agents,
            sample_method,
        )

    return history
