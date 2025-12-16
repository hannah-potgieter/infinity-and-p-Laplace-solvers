# Torus: $p$-Poisson distance-to-boundary problem

This folder contains a Newton finite element solver for the $p$-Laplacian on a 3D torus, implemented using deal.II version 9.6.1.

## Files

- pLapNewtontorus.cc — $p$-Laplacian Newton solver on a torus
- PARAMStorus.prm — solver parameters
- CMakeLists.txt — local CMake configuration

## Torus geometry

The computational domain $\Omega \subset \mathbb{R}^3$ is a torus with major radius $R = 2$ and minor radius $r = 1$, defined implicitly by

$$\Omega = \lbrace (x,y,z) \in \mathbb{R}^3 | ( 2 - \sqrt(x^2 + z^2) )^2 + y^2 \leq 1 \rbrace.$$

The boundary is then $\partial \Omega = \lbrace (x,y,z) \in \mathbb{R}^3 | ( 2 - \sqrt(x^2 + z^2) )^2 + y^2 = 1 \rbrace.$

## Boundary value problem 

This example solves the distance-to-boundary $p$-Laplacian problem

$$ \begin{cases} -\Delta_p  u_p = 1 \qquad   &\text{ in } \Omega\\ 
 u_p = 0   \qquad   &\text{ on } \partial \Omega \end{cases}$$

This is the only torus configuration explicitly implemented in this folder.

## Limiting solution and error measurement

- The solver uses continuation in $p$, starting from a small initial value and increasing to a target value.
- As $p$ tends to infinity, the solution $u_p$ converges to the distance-to-boundary function on the torus.
- Setting 'known_solution = true' enables comparison against the limiting $p \to \infty$ solution.
- Reported errors are measured relative to the limiting solution and decrease as $p$ increases.

## Parameter file (PARAMStorus.prm)

Key parameters include:

Global Parameters:
- p: target (final) value of $p$
- known_solution: setting 'true' enables error computation against the $p \to \infty$ limit

Mesh and Refinement Parameters:
- RHS is fixed to 1, corresponding to the distance-to-boundary problem
- No. of initial refinements controls the global mesh resolution
- Adaptive refinement is disabled by default 

Algorithm Parameters:
- init_p: initial value of $p$
- delta_p: increment in $p$ during continuation
- Newton, CG, and line-search tolerances control nonlinear solver behavior

## Building and running

From inside the torus directory, run

cmake -DDEAL_II_DIR=/path/to/dealii .
make run

This configures the example against an existing deal.II installation and runs the solver using PARAMStorus.prm.
