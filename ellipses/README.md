# Ellipses: p-Laplacian Test Problems

This folder contains a Newton finite element solver for the $p$-Laplacian on 2D elliptical domains, implemented using deal.II version 9.6.1.

## Files

- pLapNewtonellipse.cc — $p$-Laplacian Newton solver for 2D ellipses
- PARAMSellipse.prm — solver parameters
- ellipse.msh — mesh for ellipse 1
- ellipse3.msh — mesh for ellipse 3
- CMakeLists.txt — local CMake configuration

---

## Ellipse geometries

Three ellipses are supported:

- **Ellipse 1:**  
  $$x^2 + 16y^2 \leq 1$$

- **Ellipse 2 (rotated):**  
  $$8.5x^2 + 8.5y^2 - 15xy \leq 1$$  
  Ellipse 2 is obtained by rotating Ellipse 1 using `GridTools::rotate`, so no separate mesh file is required.

- **Ellipse 3:**  
  $$4x^2 + y^2 \leq 1$$

The domain is selected via

```
set Code for the domain = 0  # ellipse 1
set Code for the domain = 1  # ellipse 2 (rotated version of ellipse 1)
set Code for the domain = 2  # ellipse 3
```

---

## Boundary value problems

Three different boundary value problems are supported.

### 1. Distance-to-origin problem

This problem computes the distance to the origin in the large $p$ limit.

$$
\begin{cases}
-\Delta_p u_p = 1, & \text{in } \Omega, \\
u_p = 0, & \text{at the origin}, \\
\frac{\partial u_p}{\partial n} = 0, & \text{on } \partial \Omega.
\end{cases}
$$

Parameter choices:

```
set Code for the natural boundary = 2   # Neumann zero
set Code for the feature boundary = 1   # origin
set Code for the RHS = 1
```

As $p \to \infty$, the solution converges to the distance-to-origin function.

---

### 2. Distance-to-boundary problem

This problem computes the distance to the boundary in the large $p$ limit.

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

This problem uses Dirichlet boundary data corresponding to the limiting $p \to \infty$ solution of an Aronsson-type problem.

The exact solution is

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
set Code for the natural boundary = 1   # Dirichlet Aronsson
set Code for the feature boundary = 0   # none
set Code for the RHS = 0
```

As $p \to \infty$, the numerical solution converges to the exact Aronsson solution.

---

## Limiting solution and error measurement

- The solver uses continuation in p, starting from a small initial value and increasing to a target value.
- Setting
  `set known_solution = true`
  enables comparison against the limiting $p \to \infty$ solution.
- Errors are measured relative to the limiting solution and decrease as $p$ increases for all supported (described above) problem types.

---

## Parameter file (PARAMSellipse.prm)

Key parameters include:

Global parameters:
- `p` — target (final) value of $p$
- `known_solution`: setting `true` enables comparison against the $p \to \infty$ limiting solution

Mesh and refinement parameters:
- Domain selection via `Code for the domain`
- Boundary condition selection via `Code for the natural boundary` and `Code for the feature boundary`
- RHS selection via `Code for the RHS`
- `No. of initial refinements` controls the global mesh resolution
- Adaptive refinement is disabled by default

Algorithm parameters:
- `init_p`: initial value of $p$
- `delta_p`: increment in $p$ during continuation
- Newton, CG, and line-search tolerances control nonlinear solver behavior

---

## Building and running

From inside the ellipses directory, run

```
cmake -DDEAL_II_DIR=/path/to/dealii .
make run
```

This configures the example against an existing deal.II installation and runs the solver using PARAMSellipse.prm.
