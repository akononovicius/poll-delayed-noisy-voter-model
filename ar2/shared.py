from typing import Tuple, TypeVar

import numpy as np

float_or_array = TypeVar("float_or_array", float, np.ndarray)


def _get_phi(
    tau: float_or_array,
    epsi_0: float,
    epsi_1: float,
    n_agents: int,
) -> Tuple[float_or_array, float_or_array]:
    """Calculate AR(2) process coefficients (phi parameters)."""
    phi_1 = np.exp(-(epsi_0 + epsi_1 + n_agents) * tau)
    _phi_2 = n_agents / (epsi_0 + epsi_1 + n_agents)

    if isinstance(tau, np.ndarray):
        phi_2 = np.empty(tau.shape)
        phi_2.fill(_phi_2)
    else:
        phi_2 = _phi_2

    return phi_1, phi_2


def _get_psi(
    tau: float_or_array,
    epsi_0: float,
    epsi_1: float,
    n_agents: int,
) -> Tuple[
    float_or_array, float_or_array, float_or_array, float_or_array, float_or_array
]:
    """Calculate psi parameters which describe process for conditional deviation variance."""
    phi_1, _ = _get_phi(tau, epsi_0, epsi_1, n_agents)

    psi_0 = n_agents * epsi_0 * epsi_1 * (1 - phi_1**2) / ((epsi_0 + epsi_1) ** 2)
    psi_1 = (epsi_0 - epsi_1) * phi_1 * (1 - phi_1) / (epsi_0 + epsi_1)
    psi_2 = (
        n_agents
        * (epsi_0 - epsi_1)
        * (1 - phi_1)
        / ((epsi_0 + epsi_1 + n_agents) * (epsi_0 + epsi_1))
    )
    psi_12 = -2 * phi_1 * (1 - phi_1) / (epsi_0 + epsi_1 + n_agents)
    psi_22 = -n_agents * ((1 - phi_1) ** 2) / ((epsi_0 + epsi_1 + n_agents) ** 2)

    return psi_0, psi_1, psi_2, psi_12, psi_22


def _get_scaling_law(
    tau: float_or_array,
    variance: float_or_array,
    epsi_0: float = 1.0,
    epsi_1: float = 1.0,
    n_agents: int = 1000,
) -> float_or_array:
    """Calculate the scaling law when variance is given."""
    numerator = ((epsi_0 + epsi_1) ** 2) * variance
    numerator = epsi_0 * epsi_1 * (n_agents**2) - numerator

    denominator = ((epsi_0 + epsi_1) ** 3) * variance
    denominator = denominator - epsi_0 * epsi_1 * (epsi_0 + epsi_1) * n_agents
    return numerator / denominator
