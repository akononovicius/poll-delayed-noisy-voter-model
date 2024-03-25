from typing import Any, Tuple

import numpy as np
from numba import njit  # type: ignore
from scipy.sparse import dok_matrix  # type: ignore
from scipy.sparse.linalg import eigs  # type: ignore
from scipy.stats import binom  # type: ignore


@njit
def _estimate_prob_11(
    epsi_0: float, epsi_1: float, poll_outcome: int, tau: float, n_agents: int
) -> float:
    """Estimate probability of 1->1 individual agent transition."""
    rate = epsi_0 + epsi_1 + n_agents
    p_stationary = (epsi_1 + poll_outcome) / rate
    return p_stationary + (1 - p_stationary) * np.exp(-rate * tau)


@njit
def _estimate_prob_01(
    epsi_0: float, epsi_1: float, poll_outcome: int, tau: float, n_agents: int
) -> float:
    """Estimate probability of 0->1 individual agent transition."""
    rate = epsi_0 + epsi_1 + n_agents
    p_stationary = (epsi_1 + poll_outcome) / rate
    return p_stationary - p_stationary * np.exp(-rate * tau)


def _get_conditional_prob(
    x_2: np.ndarray, x_1: int, n_agents: int, p_11: float, p_01: float
) -> float:
    """Calculate probability to observe `x_2`, conditioned on initial condition `x_1` and transition probabilities `p_11` and `p_01`."""
    k = np.arange(0, np.max(x_2) + 1)
    p = binom.pmf(k, x_1, p_11) * binom.pmf(x_2[:, None] - k, n_agents - x_1, p_01)
    return np.sum(p, axis=1)


def make_transition_matrix(
    tau: float = 1, epsi_0: float = 1, epsi_1: float = 1, n_agents: int = 100
) -> Any:
    """Make transition matrix for the model.

    Input:
        tau: (default: 1)
            Time interval between successive polls.
        epsi_0: (default: 1)
            Independent transition rate from state 1 to state 0.
        epsi_1: (default: 1)
            Independent transition rate from state 0 to state 1.
        n_agents: (default: 100)
            Number of agents acting in the system.

    Output:
        Sparse matrix in CSC format encoding the transition matrix of the
        model.
    """
    transition_matrix = dok_matrix(
        ((n_agents + 1) ** 2, (n_agents + 1) ** 2), dtype=np.float64
    )
    x_2 = np.arange(0, n_agents + 1)
    for m in range(0, (n_agents + 1) ** 2):
        k = (m % (n_agents + 1)) * (n_agents + 1)
        x_0 = np.floor(m / (n_agents + 1))
        x_1 = np.floor(k / (n_agents + 1))
        const_p_11 = _estimate_prob_11(epsi_0, epsi_1, x_0, tau, n_agents)
        const_p_01 = _estimate_prob_01(epsi_0, epsi_1, x_0, tau, n_agents)
        probs = _get_conditional_prob(x_2, x_1, n_agents, const_p_11, const_p_01)
        transition_matrix[k : k + n_agents + 1, m] = probs
    return transition_matrix.tocsc()


def simulate_pmfs(
    n_steps: int,
    *,
    tau: float = 1,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: int = 100,
    initial_state: int = 50,
    initial_poll: int = 50,
) -> Any:
    """Make transition matrix for the model.

    Input:
        n_steps:
            Number of steps to simulate.
        tau: (default: 1)
            Time interval between successive polls.
        epsi_0: (default: 1)
            Independent transition rate from state 1 to state 0.
        epsi_1: (default: 1)
            Independent transition rate from state 0 to state 1.
        n_agents: (default: 100)
            Number of agents acting in the system.
        initial_state: (default: 50)
            Initial system state (i.e., X(0)). Recall that system state is a
            number of agents in state 1.
        initial_poll: (default: 50)
            Previous poll results (i.e., X(-tau)).

    Output:
        Numpy array containing outcome PMFs after each step. Rows index steps,
        columns index PMF values.
    """

    def __to_index(x_new: int, x_old: int, n: int) -> int:
        return (n + 1) * x_old + x_new

    transition_matrix = make_transition_matrix(tau, epsi_0, epsi_1, n_agents)

    histogram = np.zeros(__to_index(n_agents, n_agents, n_agents) + 1)
    histogram[__to_index(initial_state, initial_poll, n_agents)] = 1

    history = np.zeros((n_steps + 1, n_agents + 1, 2))
    history[0] = _extract_first_order_pmf(histogram)
    for idx in range(1, n_steps + 1):
        histogram = transition_matrix.dot(histogram)
        history[idx] = _extract_first_order_pmf(histogram)

    return history


@njit
def _extract_first_order_pmf(
    second_order_pmf: np.ndarray,
) -> np.ndarray:
    def __from_index(k: int, n: int) -> Tuple[int, int]:
        x_prev = int(np.floor(k / (n + 1)))
        return k - (n + 1) * x_prev, x_prev

    n_agents = int(np.sqrt(second_order_pmf.shape[0])) - 1
    new_pmf = np.zeros(n_agents + 1)
    for idx, val in enumerate(second_order_pmf):
        x, _ = __from_index(idx, n_agents)
        new_pmf[x] = new_pmf[x] + val
    return np.vstack((np.arange(0, n_agents + 1), new_pmf)).T


def get_stationary_pmf(
    tau: float = 1, epsi_0: float = 1, epsi_1: float = 1, n_agents: int = 100
) -> np.ndarray:
    """Obtain stationary PMF for the model.

    Input:
        tau: (default: 1)
            Time interval between successive polls.
        epsi_0: (default: 1)
            Independent transition rate from state 1 to state 0.
        epsi_1: (default: 1)
            Independent transition rate from state 0 to state 1.
        n_agents: (default: 100)
            Number of agents acting in the system.

    Output:
        Numpy array containing PMF: first column is the value, second column is
        the probability to observe the value.
    """
    transition_matrix = make_transition_matrix(tau, epsi_0, epsi_1, n_agents)
    _, stationary_pmf = eigs(transition_matrix, k=1, which="LR")
    stationary_pmf = np.real(stationary_pmf[:, 0])
    stationary_pmf = stationary_pmf / np.sum(stationary_pmf)
    return _extract_first_order_pmf(stationary_pmf)
