from .shared import _get_phi, _get_psi, _get_scaling_law, float_or_array


def _get_variance(
    tau: float_or_array,
    epsi_0: float,
    epsi_1: float,
    n_agents: int,
) -> float_or_array:
    """Calculate stationary variance when poll outcome is announced immediately."""
    phi_1, phi_2 = _get_phi(tau, epsi_0, epsi_1, n_agents)
    phi_3 = phi_1 + (1 - phi_1) * phi_2

    psi_0, _, _, psi_12, psi_22 = _get_psi(tau, epsi_0, epsi_1, n_agents)

    numerator = psi_0
    denominator = 1 - phi_3**2 - psi_12 - psi_22

    return numerator / denominator


def get_scaling_law(
    tau: float_or_array,
    epsi_0: float = 1.0,
    epsi_1: float = 1.0,
    n_agents: int = 1000,
) -> float_or_array:
    """Calculate the scaling law when poll outcome is announced immediately.

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
    return _get_scaling_law(
        tau, variance, epsi_0=epsi_0, epsi_1=epsi_1, n_agents=n_agents
    )
