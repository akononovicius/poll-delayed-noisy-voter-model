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


def _get_rho(
    tau: float_or_array,
    epsi_0: float,
    epsi_1: float,
    n_agents: int,
) -> Tuple[float_or_array, float_or_array]:
    """Calculate stationary correlation values (rho parameters) for AR(2) process."""
    phi_1, phi_2 = _get_phi(tau, epsi_0, epsi_1, n_agents)

    rho_1 = phi_1 / (1 - (1 - phi_1) * phi_2)
    rho_2 = (1 - phi_1) * phi_2 + (phi_1**2) / (1 - (1 - phi_1) * phi_2)

    return rho_1, rho_2


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


def _get_variance(
    tau: float_or_array,
    epsi_0: float,
    epsi_1: float,
    n_agents: int,
) -> float_or_array:
    """Calculate stationary variance (from AR(2) approach) for given parameters."""
    phi_1, phi_2 = _get_phi(tau, epsi_0, epsi_1, n_agents)

    psi_0, _, _, psi_12, psi_22 = _get_psi(tau, epsi_0, epsi_1, n_agents)

    rho_1, rho_2 = _get_rho(tau, epsi_0, epsi_1, n_agents)

    numerator = psi_0
    denominator = 1 - (phi_1 + psi_12) * rho_1 - (1 - phi_1) * phi_2 * rho_2 - psi_22

    return numerator / denominator


def get_scaling_law(
    tau: float_or_array,
    epsi_0: float = 1.0,
    epsi_1: float = 1.0,
    n_agents: int = 1000,
) -> float_or_array:
    """Calculate the scaling law of the shape parameters.

    Input:
        tau:
            Time interval between successive polls. Expects either single
            value, or a multiple value stored as numpy array.
        epsi_0: (default: 1)
            Independent transition rate from state 1 to state 0.
        epsi_1: (default: 1)
            Independent transition rate from state 0 to state 1.
        n_agents: (default: 1000)
            Number of agents acting in the system.

    Output:
        The value(s) of the scaling law for given tau(s) and model parameters.
    """
    variance = _get_variance(tau, epsi_0, epsi_1, n_agents)

    numerator = ((epsi_0 + epsi_1) ** 2) * variance
    numerator = epsi_0 * epsi_1 * (n_agents**2) - numerator

    denominator = ((epsi_0 + epsi_1) ** 3) * variance
    denominator = denominator - epsi_0 * epsi_1 * (epsi_0 + epsi_1) * n_agents
    return numerator / denominator
