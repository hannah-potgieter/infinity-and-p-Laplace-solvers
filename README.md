# FEM Training Data Generation for p-Laplace Problems

This repository contains C++ codes for generating training data using the finite element method (FEM) for p-Laplace and related large‑p / infinity‑Laplace problems. The codes are intended to produce solution data for downstream machine‑learning workflows.

All solvers are implemented using the **deal.II finite element library (version 9.6.1)**. Each group of examples compiles and runs independently using CMake.

---

## Example Categories

The repository is organized into four main categories of example sets:

* **Ellipses**: axis‑aligned and rotated ellipses in 2D
* **Balls & Hypercubes**: standard reference domains in 2D and 3D
* **Cylinder**: 3D cylindrical domain
* **Torus**: 3D toroidal geometry

Each category contains one or more examples.

---

## Mathematical Problems

The solvers target nonlinear elliptic problems of the form


$$-\nabla \cdot (|\nabla u|^{p-2} \nabla u) = f \quad \text{in } \Omega,$$


with Dirichlet or mixed Dirichlet–Neumann boundary conditions. Large values of $p$ are handled via continuation in $p$, enabling approximation of limiting infinity‑Laplace and distance‑type problems.

---

## Dependencies

Required:

* **deal.II 9.6.1** (explicitly tested)
* C++ compiler with C++17 support
* CMake

Other versions of deal.II may work but are not guaranteed.

---

## Building and Running

Each example set is built and run independently from its own directory.

A typical workflow is:

```bash
cmake -DDEAL_II_DIR=/path/to/dealii .
make run
```

This configures the example against an existing deal.II installation and builds the corresponding executable.

---

## Output

Each example produces numerical solutions of the p‑Laplace problem on the specified domain, suitable for use as training data. Output formats and exact contents may vary between example sets.

---

## Notes

* The codes assume familiarity with deal.II and nonlinear FEM workflows.
* Each example directory is expected to document its own specific setup and parameters in a local README.
