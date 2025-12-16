# Cylinder: $p$-Poisson distance-to-boundary problem

This folder contains a Newton finite element solver for the $p$-Laplacian on a 3D cylinder, implemented using deal.II version 9.6.1.

## Files

- pLapNewtoncyl.cc — $p$-Laplacian Newton solver on a cylinder
- PARAMScyl.prm — solver parameters
- CMakeLists.txt — local CMake configuration

## Cylinder geometry

The computational domain $\Omega \in \mathbb{R}^3$ is a finite cylinder defined by

$$ \Omega = \lbrace (x,y,z) \in \mathbb{R}^3 | y^2 + z^2 \leq 1, -1 \leq x \leq 1 \rbrace.$$

The boundary $\partial \Omega$ consists of the lateral surface $y^2+z^2=1$ and the end caps $x = \pm 1$.

## Boundary value problem

This example solves the distance-to-boundary $p$-Laplacian problem

$$
\begin{cases}
-\Delta_p u_p = 1, & \text{in } \Omega, \\
u_p = 0, & \text{on } \partial \Omega.
\end{cases}
$$

This is the only cylinder configuration explicitly implemented in this folder.

## Limiting solution and error measurement

- The solver uses continuation in $p$, starting from a small initial value and increasing to a target value.
- As $p \to \infty$, the solution $u_p$ converges to the distance-to-boundary function on the cylinder.
- Setting 'known_solution = true' enables comparison against this limiting $p \to \infty$ solution.
- Errors are measured relative to the limiting solution and decrease as $p$ increases.

## Parameter file (PARAMScyl.prm)

Key parameters include:

Global parameters:
- p — target (final) value of $p$
- known_solution — setting 'true' enables comparison against the $p \to \infty$ limit

Mesh and refinement parameters:
- RHS is fixed to 1, corresponding to the distance-to-boundary problem
- No of initial refinements controls the global mesh resolution
- Adaptive refinement is disabled by default

Algorithm parameters:
- init_p — initial value of $p$
- delta_p — increment in $p$ during continuation
- Newton, CG, and line-search tolerances control nonlinear solver behavior

## Building and running

From inside the cylinder directory, run

```
cmake -DDEAL_II_DIR=/path/to/dealii .
make run
```

This configures the example against an existing deal.II installation and runs the solver using PARAMScyl.prm.
