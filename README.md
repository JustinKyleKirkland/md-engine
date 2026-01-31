# MD Engine

A modular, composable molecular dynamics engine in Python.

## Quick Start

The simplest way to run a simulation:

```python
from mdcore import simulate

# Run a Lennard-Jones fluid simulation
result = simulate.lj_fluid(n_atoms=64, temperature=1.0)

print(f"Mean temperature: {result.mean_temperature:.3f}")
print(f"Diffusion coefficient: {result.diffusion_coefficient:.4f}")
```

That's it! One line to run a complete MD simulation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/md-engine.git
cd md-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Usage

### 1. Simple Simulations (Recommended for Beginners)

```python
from mdcore import simulate

# LJ fluid - default parameters
result = simulate.lj_fluid()

# LJ fluid - custom parameters
result = simulate.lj_fluid(
    n_atoms=108,        # Number of atoms
    temperature=1.0,    # Reduced temperature T* = kT/ε
    density=0.5,        # Reduced density ρ* = Nσ³/V
    n_steps=2000,       # Production steps
)

# Harmonic oscillator (great for testing)
result = simulate.harmonic_oscillator(n_steps=1000)

# Diatomic molecule with harmonic bond
result = simulate.diatomic_molecule(bond_length=1.0, spring_constant=100.0)

# NVT simulation with Langevin thermostat
result = simulate.nvt_lj_fluid(temperature=1.0, friction=1.0)
```

### 2. Plotting Results

```python
from mdcore import simulate, plotting

result = simulate.lj_fluid(n_atoms=64, n_steps=1000)

# One-line plots
plotting.energy(result)           # Energy vs time
plotting.temperature(result)      # Temperature vs time
plotting.rdf(result)              # Radial distribution function
plotting.msd(result)              # Mean square displacement
plotting.summary(result)          # All-in-one summary plot

# Save plots
plotting.energy(result, show=False)
plotting.save("energy.png")

# Trajectory visualization
plotting.trajectory_2d(result, atom_indices=[0, 1, 2], projection="xy")
```

### 3. Access Results

```python
result = simulate.lj_fluid(n_atoms=64, n_steps=1000)

# Thermodynamic properties
print(f"Mean temperature: {result.mean_temperature}")
print(f"Mean potential energy: {result.mean_potential_energy}")
print(f"Energy conservation: {result.energy_fluctuation:.2e}")

# Trajectory data
positions = result.positions      # Shape: (n_steps, n_atoms, 3)
velocities = result.velocities    # Shape: (n_steps, n_atoms, 3)

# Time series
ke = result.kinetic_energy        # Shape: (n_steps,)
pe = result.potential_energy      # Shape: (n_steps,)
temp = result.temperature         # Shape: (n_steps,)

# Analysis
rdf = result.rdf                  # Radial distribution function
msd = result.msd                  # Mean square displacement
D = result.diffusion_coefficient  # Diffusion coefficient
```

### 4. Advanced Usage (Full Control)

For users who need more control:

```python
import numpy as np
from mdcore.system import Box, MDState
from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList

# Create system
n_atoms = 64
box = Box.cubic(10.0)
positions = np.random.uniform(0, 10, (n_atoms, 3))
velocities = np.random.randn(n_atoms, 3) * 0.1
masses = np.ones(n_atoms)

state = MDState(
    positions=positions,
    velocities=velocities,
    forces=np.zeros((n_atoms, 3)),
    masses=masses,
    box=box,
)

# Setup force field
lj = LennardJonesForce(
    epsilon=np.array([1.0]),
    sigma=np.array([1.0]),
    atom_types=np.zeros(n_atoms, dtype=np.int32),
    cutoff=2.5,
)
forcefield = ForceField([lj])

# Setup neighbor list and integrator
neighbor_list = VerletList(cutoff=2.5, skin=0.3)
integrator = VelocityVerletIntegrator(dt=0.001)

# Run simulation
for step in range(1000):
    neighbor_list.build(state.positions, state.box)
    forces, energy = forcefield.compute_with_energy(state, neighbor_list)
    state = integrator.step(state, forces)
```

## Examples

Run the examples to see the engine in action:

```bash
# Quick start - run all simulation types
python examples/quickstart.py

# LJ fluid with full analysis
python examples/run_lj_simulation.py

# Harmonic oscillator (test energy conservation)
python examples/run_harmonic_oscillator.py

# NVT simulation with thermostat
python examples/run_nvt_simulation.py

# Full analysis (RDF, MSD, diffusion)
python examples/run_analysis.py
```

## Features

### Implemented

- **Force Fields**
  - Lennard-Jones (LJ) potential
  - Harmonic bonds, angles, dihedrals
  - Coulomb interactions

- **Integrators**
  - Velocity Verlet (leapfrog)
  - Langevin dynamics
  - BAOAB integrator

- **Thermostats**
  - Velocity rescaling
  - Berendsen
  - Andersen
  - Nosé-Hoover

- **Neighbor Lists**
  - Verlet list
  - Cell list

- **Analysis**
  - Radial distribution function (RDF)
  - Mean square displacement (MSD)
  - Velocity autocorrelation (VACF)
  - Energy/temperature monitoring

- **I/O**
  - XYZ format
  - PDB format
  - Checkpointing

### Planned

- ML potentials (SchNet, NequIP, MACE)
- GPU acceleration (JAX/PyTorch)
- MPI parallelization
- Enhanced sampling methods

## Reduced Units

LJ simulations use reduced (dimensionless) units:

| Quantity | Reduced Unit | Symbol |
|----------|--------------|--------|
| Length | σ (LJ size) | σ |
| Energy | ε (LJ well depth) | ε |
| Mass | m (particle mass) | m |
| Time | τ = σ√(m/ε) | τ |
| Temperature | T* = kT/ε | T* |
| Density | ρ* = Nσ³/V | ρ* |

Typical state points:
- **Liquid**: ρ* = 0.8, T* = 1.0
- **Gas**: ρ* = 0.3, T* = 1.5
- **Solid**: ρ* = 1.0, T* = 0.5

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integrators.py

# Run with coverage
pytest --cov=mdcore
```

## Project Structure

```
md-engine/
├── mdcore/                 # Main package
│   ├── simulate.py         # High-level simulation API
│   ├── system/             # Box, State, Topology
│   ├── forcefields/        # Force providers
│   ├── integrators/        # Time integration
│   ├── neighborlists/      # Neighbor lists
│   ├── analysis/           # Analysis tools
│   └── io/                 # File I/O
├── examples/               # Example scripts
├── tests/                  # Unit tests
└── benchmarks/             # Performance benchmarks
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines first.
