# Balls and Squares / Hypercubes: p-Laplacian Test Problems

This folder contains a Newton finite element solver for the $p$-Laplacian on standard reference domains in both 2D and 3D, implemented using deal.II version 9.6.1.

Supported domains include the ball and the square (2D) or cube (3D), with multiple boundary value problem configurations.

## Files

- pLapNewton.cc — $p$-Laplacian Newton solver for balls and squares / hypercubes
- PARAMS.prm — solver parameters
- CMakeLists.txt — local CMake configuration

---

## Dimensions and geometries

Both two- and three-dimensional problems are supported.

### Dimension selection

The spatial dimension is selected via

```
set Dimension = 2   # 2D
set Dimension = 3   # 3D
```

### Domain selection

Two domain types are available:

- **Ball**  
  $$\Omega = \lbrace x \in \mathbb{R}^d : |x| \le 1 \rbrace$$

- **Square / Cube**  
  $$\Omega = [-1,1]^d$$

The domain is selected via

```
set Code for the domain = 0   # ball
set Code for the domain = 1   # square (2D) / cube (3D)
```

---

## Boundary value problems

Several boundary value problems are supported through combinations of boundary conditions, feature constraints, and right-hand sides.

---

### 1. Distance-to-origin problem

This problem computes the distance to the origin in the large-$p$ limit.

$$
\begin{cases}
-\Delta_p u_p = 1, & \text{in } \Omega, \\
u_p = 0, & \text{at the origin}, \\
\frac{\partial u_p}{\partial n} = 0, & \text{on } \partial \Omega.
\end{cases}
$$

Parameter choices:

```
set Code for the natural boundary = 3   # Neumann zero
set Code for the feature boundary = 1   # origin
set Code for the RHS = 1
```

As $p \to \infty$, the solution converges to the distance-to-origin function.

---

### 2. Distance-to-boundary problem

This problem computes the distance to the boundary in the large-$p$ limit.

$$
\begin{cases}
-\Delta_p u_p = 1, & \text{in } \Omega, \\
u_p = 0, & \text{on } \partial \Omega.
\end{cases}
$$

Parameter choices:

```
set Code for the natural boundary = 0   # Dirichlet zero
set Code for the feature boundary = 0   # none
set Code for the RHS = 1
```

As $p \to \infty$, the solution converges to the distance-to-boundary function.

---

### 3. Aronsson Dirichlet problem

This problem prescribes Dirichlet boundary data corresponding to the limiting $p \to \infty$ Aronsson solution.

In 2D, the exact solution is

$$
u_\infty(x,y) = |x|^{4/3} - |y|^{4/3}.
$$

The $p$-Laplacian problem solved is

$$
\begin{cases}
-\Delta_p u_p = 0, & \text{in } \Omega, \\
u_p = u_\infty, & \text{on } \partial \Omega.
\end{cases}
$$

Parameter choices:

```
set Code for the natural boundary = 2   # Dirichlet Aronsson
set Code for the feature boundary = 0   # none
set Code for the RHS = 0
```

As $p \to \infty$, the numerical solution converges to the exact Aronsson solution.

---

### 4. Absolute-value Dirichlet problem

This problem prescribes Dirichlet boundary data given by an absolute-value function, analogous to the Aronsson case.

In 2D, the boundary data is

$$
u_\infty(x,y) = |x| - |y|.
$$

The $p$-Laplacian problem solved is

$$
\begin{cases}
-\Delta_p u_p = 0, & \text{in } \Omega, \\
u_p = u_\infty, & \text{on } \partial \Omega.
\end{cases}
$$

Parameter choices:

```
set Code for the natural boundary = 1   # Dirichlet Absolute
set Code for the feature boundary = 0   # none
set Code for the RHS = 0
```

As $p \to \infty$, the numerical solution converges to the prescribed absolute-value solution.

---

## Limiting solution and error measurement

- The solver uses continuation in $p$, starting from a small initial value and increasing to a target value.
- Setting
  `set known_solution = true`
  enables comparison against the limiting $p \to \infty$ solution.
- Errors are measured relative to the limiting solution and decrease as $p$ increases for all supported problem types listed above.

---

## Parameter file (PARAMS.prm)

Key parameters include:

Global parameters:
- `p`: target (final) value of $p$
- `known_solution`: setting `true` enables comparison against the $p \to \infty$ limiting solution

Mesh and refinement parameters:
- Dimension selection via `Dimension`
- Domain selection via `Code for the domain`
- Boundary condition selection via `Code for the natural boundary` and `Code for the feature boundary`
- RHS selection via `Code for the RHS`
- `No of initial refinements` controls the global mesh resolution
- Adaptive refinement is disabled by default

Algorithm parameters:
- `init_p`: initial value of $p$
- `delta_p`: increment in $p$ during continuation
- Newton, CG, and line-search tolerances control nonlinear solver behavior

---

## Building and running

From inside the balls / squares / hypercubes directory, run

```
cmake -DDEAL_II_DIR=/path/to/dealii .
make run
```

This configures the example against an existing deal.II installation and runs the solver using PARAMS.prm.
