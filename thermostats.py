"""
Thermostats for Molecular Dynamics Simulations
==============================================

This module contains implementations of various thermostats for controlling
temperature in molecular dynamics simulations. These allow converting NVE
(microcanonical) simulations to NVT (canonical ensemble) simulations.

The module includes:
- Andersen thermostat: Stochastic velocity reassignment
- Langevin thermostat: Stochastic dynamics with friction
- Berendsen thermostat: Velocity rescaling (weak coupling)
- Bussi thermostat: Canonical sampling velocity rescaling

Author: E.A.J.F. Peters, Copilot
Date: 2026
"""

import numpy as np
from numba import njit
from scipy.stats import ncx2 # install with 'pip install scipy'


# =============================================================================
# HELPER FUNCTIONS: Temperature and Velocity Utilities
# =============================================================================

def compute_temperature(velocities, masses, kB=1.0, remove_com=True):
    """
    Compute the instantaneous kinetic temperature from velocities.
    
    The kinetic temperature is derived from the equipartition theorem:
        (1/2) * m * <v^2> = (1/2) * k_B * T per degree of freedom
    
    For N particles in d dimensions with f degrees of freedom:
        T = (1/(f * k_B)) * sum_i (1/2 * m_i * v_i^2)
    
    Parameters
    ----------
    velocities : ndarray
        Velocity array of shape (N, 3) where N is the number of particles
    masses : ndarray or float
        Mass of each particle, shape (N,) or scalar if all masses are equal
    kB : float, optional
        Boltzmann constant (default: 1.0 in reduced units)
    remove_com : bool, optional
        If True, remove center-of-mass motion before computing temperature
        (default: True). This is important for correct temperature in finite
        systems where COM motion doesn't contribute to thermal motion.
    
    Returns
    -------
    T : float
        Instantaneous kinetic temperature
    v_com : ndarray
        Center-of-mass velocity that was removed (if remove_com=True), shape (3,)
    
    Notes
    -----
    Degrees of freedom:
    - Without COM removal: f = N * dim (typically N * 3)
    - With COM removal: f = N * dim - dim (typically N * 3 - 3)
    """
    # Convert masses to array with proper shape for broadcasting
    N, dim = velocities.shape
    masses = np.broadcast_to(np.asarray(masses).reshape(-1, 1), (N, 1))  # Shape: (N, 1)
    
    
    # Copy velocities to avoid modifying the original array
    v = velocities.copy()
    
    # Remove center-of-mass velocity if requested
    # This is physically important: COM translation doesn't contribute to temperature
    if remove_com:
        # HINT: v_com = sum(m_i * v_i) / sum(m_i)
        total_mass = np.sum(masses)
        v_com = np.sum(masses * v, axis=0) / total_mass
        v = v - v_com  # Subtract COM velocity from all particles
    
    # Calculate total kinetic energy
    # K = (1/2) * sum_i (m_i * v_i^2)
    kinetic_energy = 0.5 * np.sum(masses * v**2)
    
    # Determine degrees of freedom
    # Full system: N * dim
    # With COM constraint: N * dim - dim (we lose 'dim' DOF for COM motion)
    degrees_of_freedom = N * dim
    if remove_com:
        degrees_of_freedom -= dim
    
    # Apply equipartition theorem: K = (1/2) * f * k_B * T
    # Therefore: T = (2 * K) / (f * k_B)
    temperature = (2.0 * kinetic_energy) / (degrees_of_freedom * kB)
    
    return temperature, v_com if remove_com else None

# =============================================================================
# THERMOSTAT IMPLEMENTATIONS
# =============================================================================

def andersen_thermostat(velocities, masses, temperature, nu, dt, kB=1.0, rng=None):
    """
    Andersen thermostat: Stochastic velocity reassignment.
    
    The Andersen thermostat maintains temperature by randomly selecting
    particles and reassigning their velocities from the Maxwell-Boltzmann
    distribution at the target temperature.
    
    Algorithm:
    1. For each particle i, generate random number r ~ U(0,1)
    2. If r < (nu * dt), reassign velocity from MB distribution at T
    3. Otherwise, keep the current velocity
    
    Parameters
    ----------
    velocities : ndarray
        Current velocity array, shape (N, 3)
    masses : ndarray or float
        Mass of each particle
    temperature : float
        Target temperature
    nu : float
        Collision frequency (1/time). Higher values = stronger coupling.
        Typical value: 0.1 to 1.0 in reduced units
    dt : float
        Timestep size
    kB : float, optional
        Boltzmann constant (default: 1.0)
    if rng is None:
        rng = np.random.default_rng()
    
    Returns
    -------
    velocities_new : ndarray
        Updated velocity array, shape (N, dim)
    
    Notes
    -----
    Properties:
    - Generates correct canonical ensemble (exact NVT)
    - Destroys momentum conservation (COM can drift)
    - Strongly affects dynamics (transport properties altered)
    - Simple to implement
    - Good for equilibration, not for production runs
    
    The probability that a particle undergoes a collision in timestep dt is:
        P_collision = nu * dt
    
    This should be << 1 for the algorithm to be valid (typically < 0.1).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Copy velocities to avoid modifying the input
    v_new = velocities.copy()
    
    # Convert masses to array
    masses = np.asarray(masses)
    masses = masses.reshape((-1,1))
    
    N, dim = velocities.shape
    
    # probability that a particle does not collide: P_no_collision = exp(-nu * dt)
    c = np.exp(-nu * dt)
        
    # For each particle, decide whether it collides with heat bath
    # Generate N random numbers between 0 and 1
    random_numbers = rng.random(N)
    
    # Identify particles that undergo collision
    collision_mask = random_numbers > c
    num_collisions = np.count_nonzero(collision_mask)  # For debugging: how many particles collide on average?
    
    # Sample new velocities for colliding particles
    # Each component is sampled independently
    # Calculate standard deviation for velocity components
    sigma = np.broadcast_to(np.sqrt(kB * temperature / masses), (N, dim))
    for i in range(N):
        if collision_mask[i]:
            # Sample new velocity from Gaussian with correct sigma
            v_new[i] = rng.normal(0.0, sigma[i], size=3)
          
    return v_new

def langevin_thermostat(velocities, masses, temperature, nu, dt, kB=1.0, rng=None):
    """
    Langevin thermostat: Stochastic dynamics with friction and noise.
    
    The Langevin equation adds friction and random forces to Newton's equations:
        m * dv/dt = F_conservative + F_friction + F_random
    
    where:
        F_friction = -gamma * v  (friction proportional to velocity)
        F_random ~ sqrt(2 * gamma * kB * T / dt) * Gaussian
    
    This is integrated using the Euler-Maruyama scheme or more sophisticated
    integrators. Here we use a simple update:
        v_new = c * v_old + sqrt(1 - c^2) * sigma * R
    
    where:
        c = exp(-nu * dt)  (velocity correlation factor, nu = gamma/m)
        sigma = sqrt(kB * T / m)  (thermal velocity scale)
        R ~ N(0, 1)  (Gaussian random numbers)
    
    Parameters
    ----------
    velocities : ndarray
        Current velocity array, shape (N, 3)
    masses : ndarray or float
        Mass of each particle
    temperature : float
        Target temperature
    nu : float
        Relaxation rate for the thermostat. Smaller nu = weaker coupling.
        Typical value: 0.01-0.1 in reduced units
    dt : float
        Timestep size
    kB : float, optional
        Boltzmann constant (default: 1.0)
    rng : Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    velocities_new : ndarray
        Updated velocity array after Langevin step, shape (N, 3)
    
    Notes
    -----
    Properties:
    - Generates correct canonical ensemble (exact NVT)
    - Smooth dynamics (friction + continuous noise)
    - Affects transport properties (lowers diffusion coefficient)
    - Can be integrated with position updates using various schemes
    - Good for both equilibration and production (but dynamics are modified)
    
    The friction coefficient nu determines the coupling strength:
    - Small nu (< 0.1): Weak coupling, nearly Newtonian dynamics
    - Medium nu (0.1-1.0): Moderate coupling, good for most applications
    - Large nu (> 1.0): Strong damping, overdamped regime
    
    For more accurate integration with position updates, use the BAOAB or
    similar splitting schemes. This implementation is a simple velocity update
    that can be inserted into a Velocity-Verlet integrator.
    
    TASK FOR STUDENTS:
    - Calculate the velocity correlation factor c = exp(-nu * dt)
    - Calculate the noise amplitude for each particle (mass-dependent)
    - Update velocities using the Langevin equation
    - Verify that the algorithm preserves the Maxwell-Boltzmann distribution
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert masses to array
    masses = np.asarray(masses)
    masses = masses.reshape((-1,1))
    
    N, dim = velocities.shape
    
    # Calculate velocity correlation factor
    # c = exp(-nu * dt) gives exponential decay of velocity autocorrelation
    c = np.exp(-nu * dt)
    
    # Calculate thermal velocity scale for each particle
    # From equipartition: <(1/2)*m*v^2> = (1/2)*kB*T
    # So: sigma = sqrt(kB*T/m)
    sigma = np.sqrt(kB * temperature / masses)
    
    # Calculate noise amplitude
    # The noise term must balance the dissipation to maintain correct temperature
    # amplitude = sqrt(1 - c^2) ensures correct equilibrium distribution
    noise_amplitude = np.sqrt(1.0 - c * c)
    
    # Sample random numbers from standard normal distribution
    # Shape: (N, dim) to match velocity array
    random_noise = rng.normal(0, 1, size=(N, dim))
    
    # STUDENT TASK:
    # Update velocities using the Langevin / Ornstein-Uhlenbeck equation here
    
    return velocities_new


def berendsen_thermostat(velocities, masses, temperature, nu, dt, kB=1.0, remove_com=True):
    """
    Berendsen thermostat: Velocity rescaling with exponential relaxation.
    
    The Berendsen thermostat rescales all velocities by a factor that makes
    the temperature relax exponentially toward the target temperature:
        dT/dt = (T_target - T) / tau
    To make all thermostats have the same coupling strength, we can express tau in terms of nu:
        dT/dt = 2*nu * (T_target - T)
    
    The rescaling factor conventionally used is:
        scaling_factor = sqrt(1 + (dt/tau) * (T_target/T_current - 1))
    We take the form that ensures smooth exponential relaxation:
        scaling_factor = sqrt(c**2 + (1-c**2) * (T_target/T_current))
        where c = exp(-nu*dt) is the velocity correlation factor.    
    
    Then all velocities are multiplied by scaling_factor:
        v_new = scaling_factor * v_old
    
    Parameters
    ----------
    velocities : ndarray
        Current velocity array, shape (N, 3)
    masses : ndarray or float
        Mass of each particle
    temperature : float
        Target temperature
    nu : float
        Relaxation rate for the thermostat. Smaller nu = weaker coupling.
        Typical value: 0.01 to 0.1 in reduced units
    dt : float
        Timestep size
    kB : float, optional
        Boltzmann constant (default: 1.0)
    remove_com : bool, optional
        Whether to remove the center-of-mass velocity before computing temperature.
        Default is True.
    
    Returns
    -------
    velocities_new : ndarray
        Rescaled velocity array, shape (N, 3)
    
    Notes
    -----
    Properties:
    - Simple and efficient
    - Smooth temperature control (no sudden jumps)
    - Does NOT generate correct canonical ensemble!
      (Temperature fluctuations are suppressed)
    - Minimal perturbation to dynamics
    - Good for equilibration, not for production runs where correct
      ensemble is required
    
    The Berendsen thermostat is a "weak coupling" method that gradually
    pushes the system temperature toward the target. It's deterministic
    (no random numbers) and preserves the relative velocities of particles.
    
    TASK FOR STUDENTS:
    - Compute the rescaling factor
    - Consider what happens when T_current is very close to the imposed temperature
    """
    # Calculate current temperature
    T_current, v_com = compute_temperature(velocities, masses, kB=kB, remove_com=remove_com)
    
    # If T_current is close to zero, we might get division by zero
    # or extreme rescaling. Add a check and handle it appropriately.
    if T_current < 1e-10:
        print("Warning: Current temperature is nearly zero. Skipping rescaling.")
        return velocities.copy()
    
    # STUDENT TASK: Compute the scaling_factor using the formula that ensures exponential relaxation
    c_sq = np.exp(-2*nu*dt)
    scaling_factor = np.sqrt(c_sq + (1-c_sq) * (temperature / T_current))
    
    # STUDENT TASK:
    # Rescale all velocities by the same factor
    # This preserves the velocity distribution shape but changes its width
    if remove_com and v_com is not None:
        # Add back COM velocity after rescaling
        
    else:
        # rescale all velocities
    
    return velocities_new

def bussi_thermostat(velocities, masses, temperature, nu, dt, kB=1.0, 
                     remove_com=True, rng=None):
    """
    Bussi-Donadio-Parrinello thermostat (Canonical Sampling Velocity Rescaling).
    
    The Bussi thermostat (also known as CSVR) combines the smooth temperature
    control of Berendsen with stochastic sampling to produce the correct
    canonical ensemble. It's one of the best thermostats for production runs.
    
    Algorithm:
    1. Calculate current normalized kinetic energy K = sum(m * v^2) / (kB * T)
    2. Sample new normalized kinetic energy K' from the correct conditional distribution
    3. Rescale all velocities by alpha = sqrt(K'/K)
    
    The key insight is that K' is sampled from a non-central chi-squared
    distribution that depends on K, the target temperature, and the coupling
    strength (determined by tau).
    
    Parameters
    ----------
    velocities : ndarray
        Current velocity array, shape (N, dim)
    masses : ndarray or float
        Mass of each particle
    temperature : float
        Target temperature
    nu : float
        Collision frequency for the thermostat. Determines the strength of coupling to the heat bath.
    dt : float
        Timestep size
    kB : float, optional
        Boltzmann constant (default: 1.0)
    remove_com : bool, optional
        If True, remove COM motion when computing kinetic energy (default: True)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    velocities_new : ndarray
        Rescaled velocity array, shape (N, dim)
    
    Notes
    -----
    Properties:
    - Generates correct canonical ensemble (exact NVT)
    - Smooth temperature control (like Berendsen)
    - Minimal perturbation to dynamics
    - Preserves momentum conservation (if remove_com=True)
    - Best choice for production MD runs
    
    optional FOR STUDENTS:
    Study the Python notebook bussi_thermostat.ipynb to understand the implementation.

    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Copy velocities
    v_work = np.array(velocities, copy=True)
    N, dim = v_work.shape
    
    # Convert masses to array with shape (N, 1) for broadcasting
    masses = np.broadcast_to(np.asarray(masses).reshape((-1,1)), (N, 1))
    
    # Degrees of freedom
    f = N * dim
    if remove_com:
        # Remove COM velocity before computing kinetic energy
        total_mass = np.sum(masses)
        v_com = np.sum(masses * v_work, axis=0) / total_mass
        v_work = v_work - v_com
        f = f - dim  # Lose dim degrees of freedom for COM constraint
    
    # Calculate current kinetic energy in normalized units
    # K_normalized = K / (0.5 * f * kB * T) = sum(m*v^2) / (kB * T)
    K_norm = np.sum(masses * v_work**2) / (kB * temperature)
    
    # Calculate coupling parameter
    # c = exp(-nu*dt) determines the strength of coupling to heat bath
    # c → 1: weak coupling (small nu)
    # c → 0: strong coupling (large nu)
    c = np.exp(-nu * dt)
    
    # Sample new kinetic energy from the conditional distribution
    # This is the heart of the Bussi algorithm
    K_norm_new = sample_bussi_kinetic_energy(K_norm, f, c, rng=rng)
    
    # Calculate rescaling factor
    alpha = np.sqrt(K_norm_new / K_norm)
    
    # Rescale velocities
    v_work = alpha * v_work
    
    # Add back COM velocity if it was removed
    if remove_com:
        v_work = v_work + v_com
    
    return v_work

def sample_bussi_kinetic_energy(K_norm, f, c, rng=None):
    """
    Sample new normalized kinetic energy for Bussi thermostat.
    
    This implements the sampling from the conditional distribution:
        P(K' | K) for the Bussi thermostat
    
    The distribution is constructed to satisfy detailed balance for the
    canonical ensemble while providing smooth temperature dynamics.
    
    Parameters
    ----------
    K_norm : float
        Current normalized kinetic energy = sum(m*v^2) / (kB*T)
    f : int
        Degrees of freedom
    c : float
        Coupling parameter = exp(-dt/tau)
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default: None)
    
    Returns
    -------
    K_norm_new : float
        New normalized kinetic energy sampled from conditional distribution
    
    """
    if rng is None:
        rng = np.random.default_rng()

    a2 = 1.0 - c**2
    if a2 <= 0.0:
        return float(K_norm)  # No rescaling if c=1 (nu=0)

    lam = (c**2 / a2) * K_norm
    X = ncx2.rvs(df=f, nc=lam, random_state=rng)
    K_norm_new = a2 * X
    return K_norm_new
