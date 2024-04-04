from typing import Optional

import numpy as np
from numpy.typing import NDArray


def get_mean(
    poll_ids: list[int] | NDArray[np.int_],
    *,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: float = 1000,
    initial_state: float = 500,
    initial_poll: Optional[float] = None,
) -> np.ndarray:
    """Calculate temporal dependence of the mean."""
    if initial_poll is None:
        initial_poll = initial_state

    mean_final = get_stationary_mean(epsi_0=epsi_0, epsi_1=epsi_1, n_agents=n_agents)
    change_rate = n_agents / (epsi_0 + epsi_1 + n_agents)
    even_distance = initial_state - mean_final
    odd_distance = initial_poll - mean_final
    change_term = np.zeros(len(poll_ids))
    for idx, poll_idx in enumerate(poll_ids):
        if poll_idx % 2 == 0:
            change_term[idx] = even_distance * (change_rate ** (poll_idx // 2))
        else:
            change_term[idx] = odd_distance * (change_rate ** ((poll_idx + 1) // 2))
    return mean_final + change_term


def get_stationary_mean(
    *, epsi_0: float = 1, epsi_1: float = 1, n_agents: float = 1000
) -> float:
    """Calculate stationary mean."""
    return n_agents * epsi_1 / (epsi_0 + epsi_1)


def _get_sq_mean(
    poll_ids: list[int] | NDArray[np.int_],
    *,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: float = 1000,
    initial_state: float = 500,
    initial_poll: Optional[float] = None,
) -> np.ndarray:
    """Calculate temporal dependence of the mean of the square."""
    if initial_poll is None:
        initial_poll = initial_state

    epsi_sum = epsi_0 + epsi_1

    mean_final = get_stationary_mean(epsi_0=epsi_0, epsi_1=epsi_1, n_agents=n_agents)
    sq_mean_final = _get_stationary_sq_mean(
        epsi_0=epsi_0, epsi_1=epsi_1, n_agents=n_agents, use_approximation=False
    )
    sq_mean_even_mid = (
        (initial_state - mean_final)
        * (epsi_0 - epsi_1 + n_agents * (1 + 2 * epsi_1))
        / (epsi_sum + 1)
    )
    sq_mean_odd_mid = (
        (initial_poll - mean_final)
        * (epsi_0 - epsi_1 + n_agents * (1 + 2 * epsi_1))
        / (epsi_sum + 1)
    )

    even_mult = initial_state**2 - sq_mean_final - sq_mean_even_mid
    odd_mult = initial_poll**2 - sq_mean_final - sq_mean_odd_mid
    mid_rate = n_agents / (epsi_sum + n_agents)
    main_rate = mid_rate * (n_agents - 1) / (epsi_sum + n_agents)

    result = np.zeros(len(poll_ids))
    for idx, poll_idx in enumerate(poll_ids):
        if poll_idx % 2 == 0:
            exp_term = poll_idx // 2
            term_1 = even_mult * (main_rate**exp_term)
            term_2 = sq_mean_even_mid * (mid_rate**exp_term)
            result[idx] = sq_mean_final + term_1 + term_2
        else:
            exp_term = (poll_idx + 1) // 2
            term_1 = odd_mult * (main_rate**exp_term)
            term_2 = sq_mean_odd_mid * (mid_rate**exp_term)
            result[idx] = sq_mean_final + term_1 + term_2
    return result


def _get_stationary_sq_mean(
    *,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: float = 1000,
    use_approximation: bool = False,
) -> float:
    """Calculate stationary mean of the square."""
    epsi_sum = epsi_0 + epsi_1
    if use_approximation:
        num = epsi_1 * (epsi_1 + 0.5) * (n_agents**2)
        denom = epsi_sum * (epsi_sum + 0.5)
        return num / denom
    term_1 = n_agents * epsi_1 / epsi_sum
    num = (epsi_0 + n_agents + epsi_1 * n_agents) * epsi_sum
    num = num + n_agents * (epsi_0 - epsi_1 + n_agents * (1 + 2 * epsi_1))
    denom = (epsi_sum + n_agents) ** 2 - n_agents * (n_agents - 1)
    return term_1 * num / denom


def get_var(
    poll_ids: list[int] | NDArray[np.int_],
    *,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: float = 1000,
    initial_state: float = 500,
    initial_poll: Optional[float] = None,
) -> np.ndarray:
    """Calculate temporal dependence of the variance."""
    mean_time = get_mean(
        poll_ids,
        epsi_0=epsi_0,
        epsi_1=epsi_1,
        n_agents=n_agents,
        initial_state=initial_state,
        initial_poll=initial_poll,
    )
    sq_mean_time = _get_sq_mean(
        poll_ids,
        epsi_0=epsi_0,
        epsi_1=epsi_1,
        n_agents=n_agents,
        initial_state=initial_state,
        initial_poll=initial_poll,
    )
    return sq_mean_time - mean_time**2


def get_stationary_var(
    *,
    epsi_0: float = 1,
    epsi_1: float = 1,
    n_agents: float = 1000,
    use_approximation: bool = False,
) -> float:
    """Calculate stationary variance."""
    mean = get_stationary_mean(epsi_0=epsi_0, epsi_1=epsi_1, n_agents=n_agents)
    sq_mean = _get_stationary_sq_mean(
        epsi_0=epsi_0,
        epsi_1=epsi_1,
        n_agents=n_agents,
        use_approximation=use_approximation,
    )
    return sq_mean - mean**2
