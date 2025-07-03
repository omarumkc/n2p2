# Why Two Different Solvers for Charge Equilibration?

## The Two Approaches

### 1. **Training (nnp-train)**: Direct QR Solver
```cpp
// From Structure.cpp - charge equilibration during training
VectorXd charges = A.colPivHouseholderQr().solve(b);
```

### 2. **MD Simulation (LAMMPS)**: BFGS Iterative Optimization
```cpp
// From fix_hdnnp.cpp - charge equilibration during MD
T = gsl_multimin_fdfminimizer_vector_bfgs2;
s = gsl_multimin_fdfminimizer_alloc(T, nsize);
gsl_multimin_fdfminimizer_set(s, &QEq_minimizer, Q, step, min_tol);
```

## Key Differences in Context

| Aspect | Training (`nnp-train`) | MD Simulation (LAMMPS) |
|--------|------------------------|-------------------------|
| **Problem** | Solve **AQ = b** | Minimize **E_QEq(Q)** |
| **Method** | Direct linear solver | Iterative optimization |
| **Frequency** | Once per training update | Every MD timestep |
| **Performance Priority** | Accuracy & Derivatives | Speed per timestep |
| **Initial Guess** | No initial guess needed | Previous timestep charges |
| **Parallelization** | Matrix distributed | Function evaluation distributed |

## Why Different Approaches?

### **1. Mathematical Formulation Difference**

#### Training: Linear System
```cpp
// Training solves the linear equilibration equations directly:
// ∂E_elec/∂Qᵢ = χᵢ + JᵢᵢQᵢ + Σⱼ JᵢⱼQⱼ = μ  (for all i)
// ∑Qᵢ = Q_total

// This gives: AQ = b where
// A[i,j] = Jᵢⱼ (Coulomb matrix) + constraint row/column
// b[i] = -χᵢ + constraint value
```

#### LAMMPS: Energy Minimization
```cpp
// LAMMPS minimizes the total electrostatic energy:
// E_QEq = Σᵢ χᵢQᵢ + 0.5 Σᵢ JᵢᵢQᵢ² + 0.5 Σᵢ≠ⱼ JᵢⱼQᵢQⱼ
// Subject to: ∑Qᵢ = Q_total

double FixHDNNP::QEq_f(const gsl_vector *v)  // Energy function
{
    double E_qeq = 0.0;
    for (i = 0; i < nlocal; i++) {
        double qi = gsl_vector_get(v, tag[i]-1);
        E_qeq += hdnnp->chi[i] * qi + 0.5 * hdnnp->hardness[type[i]-1] * qi*qi;
        // + Coulomb interactions...
    }
    return E_qeq;
}

void FixHDNNP::QEq_df(const gsl_vector *v, gsl_vector *dEdQ)  // Gradient
{
    // ∂E_QEq/∂Qᵢ = χᵢ + JᵢᵢQᵢ + Σⱼ JᵢⱼQⱼ
}
```

### **2. Performance Requirements**

#### Training: Accuracy Priority
- **Called infrequently**: Once per training update (maybe 100-1000 times total)
- **Need exact derivatives**: ∂Q/∂χ required for backpropagation
- **Matrix reuse**: Same A matrix used for multiple derivative calculations
- **Numerical precision**: Training requires high accuracy for gradient calculations

```cpp
// Training needs these derivatives:
void Structure::calculateDQdChi(vector<Eigen::VectorXd> &dQdChi)
{
    for (size_t i = 0; i < numAtoms; ++i)
    {
        VectorXd b_deriv(numAtoms+1);
        b_deriv.setZero();
        b_deriv(i) = -1.0;
        // Reuse the SAME QR decomposition for multiple RHS vectors
        dQdChi.push_back(A.colPivHouseholderQr().solve(b_deriv));
    }
}
```

#### LAMMPS: Speed Priority
- **Called frequently**: Every MD timestep (millions of times)
- **Good initial guess**: Previous timestep charges are excellent starting point
- **Constraint handling**: Must enforce charge conservation during optimization
- **Parallel efficiency**: Distributed function evaluation across MPI ranks

```cpp
// LAMMPS optimization with warm start:
for (i = 0; i < nsize; i++)
{
    if (hdnnp->minim_init_style == 1)
    {
        gsl_vector_set(Q, i, qall[i]);   // Use previous charges as initial guess
    }
}
```

### **3. Constraint Handling**

#### Training: Built into Linear System
```cpp
// Constraints built into matrix structure:
A.row(numAtoms).head(numAtoms).setOnes();    // ∑Qᵢ = Q_total
A.col(numAtoms).head(numAtoms).setOnes();
A(numAtoms, numAtoms) = 0.0;
b(numAtoms) = totalCharge;
```

#### LAMMPS: Projection During Optimization
```cpp
// Constraints enforced by projection after each iteration:
do {
    status = gsl_multimin_fdfminimizer_iterate(s);
    
    // Charge conservation projection:
    qsum_it = 0.0;
    for(i = 0; i < nsize; i++) {
        qsum_it += gsl_vector_get(s->x, i);
    }
    for(i = 0; i < nsize; i++) {
        qi = gsl_vector_get(s->x,i);
        gsl_vector_set(s->x,i, qi - (qsum_it-qRef)/nsize);  // Project back
    }
} while (status == GSL_CONTINUE && iter < maxit);
```

### **4. Parallelization Strategy**

#### Training: Matrix-Based Parallelization
```cpp
// Parallel matrix construction and factorization
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (size_t i = 0; i < numAtoms; ++i)
{
    // Build A matrix entries in parallel
    A(i,j) = coulombInteraction(i,j);
}
// Single solve with optimized linear algebra libraries
```

#### LAMMPS: Function Evaluation Parallelization
```cpp
// Each MPI rank computes local contributions to energy/gradient:
E_qeq_loc = 0.0;
for (i = 0; i < nlocal; i++)  // Only local atoms
{
    E_qeq_loc += hdnnp->chi[i] * qi + interactions_with_neighbors;
}
MPI_Allreduce(&E_qeq_loc, &E_qeq, 1, MPI_DOUBLE, MPI_SUM, world);
```

### **5. Convergence Properties**

#### Training: Exact Solution
- **Direct method**: Gets exact solution (within machine precision)
- **No iterations**: Fixed O(N³) cost regardless of condition number
- **Robust**: Handles ill-conditioned matrices via pivoting

#### LAMMPS: Iterative Convergence
- **Warm start advantage**: Previous charges are near-optimal
- **Fast convergence**: Typically 5-20 BFGS iterations
- **Adaptive**: Can adjust tolerance based on MD accuracy needs

```cpp
// LAMMPS convergence control:
status = gsl_multimin_test_gradient(s->gradient, hdnnp->grad_tol);
// Typical grad_tol: 1e-6 to 1e-4 (less strict than training)
```

## Performance Analysis

### **Training Context**
```
Per structure in training:
- QR decomposition: O(N³) ≈ 50-100 μs for 50 atoms
- Multiple derivatives: N × O(N²) ≈ 200 μs for 50 atoms  
- Total per structure: ~300 μs
- Called: ~1000 times total → 0.3 seconds total
```

### **MD Context**
```
Per timestep in MD:
- BFGS optimization: 10 iterations × O(N²) ≈ 20 μs for 50 atoms
- Called: 1,000,000 timesteps → 20 seconds total
- Warm start reduces to ~5 iterations → 10 seconds total
```

## Why BFGS is Better for MD

### **1. Warm Start Advantage**
```cpp
// Charges change slowly between timesteps:
Q(t+Δt) ≈ Q(t) + small_change
// BFGS can converge in 3-5 iterations vs. O(N³) direct solve
```

### **2. Memory Efficiency**
```cpp
// BFGS only needs:
// - Current charges: N doubles
// - Gradient: N doubles  
// - BFGS history: ~10N doubles
// Total: ~12N doubles ≈ 2.4 KB for 50 atoms

// QR solver needs:
// - Full matrix A: (N+1)² doubles ≈ 20 KB for 50 atoms
// - QR factors: Similar to A
// Total: ~40 KB for 50 atoms
```

### **3. Numerical Stability in MD**
```cpp
// MD can tolerate looser convergence:
grad_tol = 1e-4;  // vs. 1e-12 in training

// Energy conserving MD only needs relative accuracy
// Small charge errors don't accumulate over time
```

## Summary

The two different solvers reflect the **different priorities** and **computational contexts**:

| **Training** | **MD Simulation** |
|--------------|-------------------|
| **Exact solutions needed** | **Approximate solutions sufficient** |
| **Infrequent calls** | **Every timestep** |
| **Need derivatives** | **Just need charges** |
| **Cold start** | **Warm start available** |
| **High precision required** | **Speed critical** |

**QR decomposition** is perfect for training: robust, exact, handles derivatives naturally.

**BFGS optimization** is perfect for MD: fast with warm start, memory efficient, easily parallelized.

This is a **classic example** of choosing the right algorithm for the right context!