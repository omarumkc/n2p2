# Step-by-Step Analysis: `nnp-train 1` for 4G Model Electrostatic Training

## Overview

When you run `nnp-train 1`, you are executing **Stage 1** of a **4th Generation (4G) Neural Network Potential** training process. This stage focuses specifically on training the **electrostatic neural networks** to predict atomic charges.

## What is a 4G Neural Network Potential?

A 4G neural network potential is an advanced type of machine learning potential that includes:
- **Short-range interactions**: Handled by traditional neural networks
- **Long-range electrostatic interactions**: Handled by separate neural networks that predict atomic charges
- **Non-local charge transfer**: Allowing for more accurate representation of electronic effects

The training happens in **two distinct stages**:
1. **Stage 1**: Train charge prediction neural networks (electrostatic)
2. **Stage 2**: Train energy and force prediction neural networks (short-range)

## Detailed Technical Process: Electronegativity → Charges

### 1. **Neural Network Electronegativity Prediction**

In Stage 1, the electrostatic neural networks **do not directly predict charges**. Instead, they predict **atomic electronegativity (χ)** values:

```cpp
// From Training.cpp - charge update process
NeuralNetwork& nn = elements.at(ak.element).neuralNetworks.at("elec");
nn.setInput(&(ak.G.front()));  // Input: symmetry functions
nn.propagate();
nn.getOutput(&(ak.chi));       // Output: electronegativity χ
```

#### What happens:
- Each element (H, O, etc.) has its own **electrostatic neural network**
- **Input**: Symmetry functions describing the local atomic environment
- **Output**: Atomic electronegativity χ (single value per atom)
- **Normalization**: χ values are normalized with `normalized("negativity", χ)`

### 2. **Charge Equilibration (QEq) Process**

The conversion from electronegativity to charges happens through **charge equilibration** - solving a system of linear equations based on electronegativity equalization principle.

#### Mathematical Foundation:

The charge equilibration is based on the principle that **all atoms in a molecule should have equal electronegativity** at equilibrium. The total electrostatic energy is:

```
E_elec = Σᵢ χᵢQᵢ + 0.5 Σᵢ JᵢᵢQᵢ² + 0.5 Σᵢ≠ⱼ JᵢⱼQᵢQⱼ
```

Where:
- **χᵢ**: Electronegativity of atom i (from neural networks)
- **Qᵢ**: Charge of atom i (to be determined)
- **Jᵢᵢ**: Atomic hardness (element-specific parameter)
- **Jᵢⱼ**: Coulomb interaction between atoms i and j

#### The QEq System:

Minimizing the electrostatic energy with respect to charges gives the **charge equilibration equations**:

```
∂E_elec/∂Qᵢ = χᵢ + JᵢᵢQᵢ + Σⱼ≠ᵢ JᵢⱼQⱼ = μ  (for all i)
Σᵢ Qᵢ = Q_total                                    (charge conservation)
```

This forms a linear system **AQ = b** where:
- **A**: Matrix containing hardness and Coulomb interactions
- **Q**: Vector of atomic charges (unknowns)
- **b**: Vector containing electronegativities and total charge constraint

### 3. **Implementation Details**

```cpp
// From Structure.cpp - calculateElectrostaticEnergy()
void Structure::calculateElectrostaticEnergy(...)
{
    // Setup the A matrix (hardness + Coulomb interactions)
    A.resize(numAtoms + 1, numAtoms + 1);
    
    // Diagonal terms: atomic hardness
    for (size_t i = 0; i < numAtoms; ++i)
    {
        A(i,i) = hardness(atoms[i].element);
    }
    
    // Off-diagonal terms: Coulomb interactions Jᵢⱼ
    for (size_t i = 0; i < numAtoms; ++i)
    {
        for (size_t j = i+1; j < numAtoms; ++j)
        {
            double Jij = calculateCoulombInteraction(i, j);
            A(i,j) = A(j,i) = Jij;
        }
    }
    
    // Charge conservation constraint
    for (size_t i = 0; i < numAtoms; ++i)
    {
        A(numAtoms, i) = A(i, numAtoms) = 1.0;
    }
    A(numAtoms, numAtoms) = 0.0;
    
    // Setup RHS vector b
    VectorXd b(numAtoms + 1);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        b(i) = -atoms[i].chi;  // Negative electronegativity
    }
    b(numAtoms) = totalCharge;  // Usually 0 for neutral molecules
    
    // Solve the linear system: AQ = b
    VectorXd charges = A.colPivHouseholderQr().solve(b);
    
    // Extract charges (exclude Lagrange multiplier)
    for (size_t i = 0; i < numAtoms; ++i)
    {
        atoms[i].charge = charges(i);
    }
}
```

### 4. **Coulomb Interaction Calculation**

The Coulomb interactions **Jᵢⱼ** include several components:

1. **Basic Coulomb interaction**: `1/(4πε₀rᵢⱼ)`
2. **Gaussian charge distribution**: Using element-specific widths σᵢ, σⱼ
3. **Ewald summation**: For periodic boundary conditions
4. **Screening function**: For finite-range electrostatics

```cpp
// Gaussian-smeared Coulomb interaction
double J_ij = erfc(r_ij / sqrt(2*(σᵢ² + σⱼ²))) / (4πε₀ * r_ij)
```

### 5. **Training Process for Charges**

During Stage 1 training, the process for each update is:

1. **Forward pass**: 
   - Calculate electronegativity χᵢ for all atoms using current NN weights
   - Solve QEq system to get charges Qᵢ from χᵢ values

2. **Error calculation**:
   - Compare predicted charges Qᵢ with reference charges Q_ref,i
   - Calculate RMSE: `√(Σᵢ(Qᵢ - Q_ref,i)²/N)`

3. **Gradient calculation**:
   - Compute `∂RMSE/∂χᵢ` using chain rule through QEq system
   - Compute `∂χᵢ/∂w` using backpropagation through neural networks
   - Combine: `∂RMSE/∂w = Σᵢ (∂RMSE/∂χᵢ)(∂χᵢ/∂w)`

4. **Weight update**:
   - Update neural network weights using Kalman filter or gradient descent

### 6. **Derivative Calculations**

The key challenge is computing derivatives of charges with respect to electronegativities:

```cpp
// From Structure.cpp - calculateDQdChi()
void Structure::calculateDQdChi(vector<Eigen::VectorXd> &dQdChi)
{
    for (size_t i = 0; i < numAtoms; ++i)
    {
        VectorXd b(numAtoms+1);
        b.setZero();
        b(i) = -1.0;  // ∂b/∂χᵢ = -1 (since b contains -χ)
        
        // Solve: A * (∂Q/∂χᵢ) = ∂b/∂χᵢ
        dQdChi.push_back(A.colPivHouseholderQr().solve(b).head(numAtoms));
    }
}
```

This gives **∂Qⱼ/∂χᵢ** for all atoms j with respect to electronegativity of atom i.

## Step-by-Step Process for `nnp-train 1`

### 1. **Initialization and Setup**
```cpp
// From nnp-train.cpp main()
Training training;
training.setupMPI();
training.initialize();
training.loadSettingsFile();
training.setStage(stage);  // stage = 1
```

#### What happens:
- MPI parallelization is initialized
- Neural network architecture is loaded from `input.nn`
- Training parameters are parsed
- **Stage 1 is set**, which configures the training for charge prediction

### 2. **Stage 1 Configuration**
```cpp
// From Training.cpp setStage()
if (stage == 1)
{
    pk.push_back("charge");  // Training property: charges
    nnId = "elec";          // Neural network ID: electrostatic
}
```

#### What happens:
- Training is configured to focus on **charge prediction**
- The **electrostatic neural networks** are selected as the target networks
- Force and energy training are disabled for this stage

### 3. **Data Preparation**
```cpp
training.setupGeneric();
training.setupSymmetryFunctionScaling();
training.setupSymmetryFunctionStatistics();
training.setupRandomNumberGenerator();
training.distributeStructures(true);
```

#### What happens:
- **Symmetry functions** are set up (mathematical descriptors of atomic environments)
- **Scaling parameters** are loaded from `scaling.data`
- Training data is distributed across MPI processes
- Random number generators are initialized for stochastic training

### 4. **Training/Test Set Selection**
```cpp
training.selectSets();
training.writeSetsToFiles();
```

#### What happens:
- Structures are randomly divided into training and test sets
- Test fraction is controlled by `test_fraction` parameter (default: 0.1 = 10%)
- Files `train.data` and `test.data` are written

### 5. **Neural Network Weight Initialization**
```cpp
training.initializeWeights();
```

#### What happens:
- **Electrostatic neural network weights** are randomly initialized
- Initial values are set between `weights_min` and `weights_max` (default: -1.0 to 1.0)
- Each element (H, O in water example) gets its own electrostatic neural network

### 6. **Data Normalization (Optional)**
```cpp
if (training.settingsKeywordExists("normalize_data_set"))
{
    training.dataSetNormalization();
}
```

#### What happens:
- If enabled, reference data statistics are computed
- Normalization factors are calculated to improve training stability
- **Note**: Charge normalization is not yet fully implemented for Stage 1

### 7. **Unit Conversion**
```cpp
if (training.useNormalization()) training.toNormalizedUnits();
```

#### What happens:
- All data is converted to normalized units if normalization is enabled
- This helps with numerical stability during training

### 8. **Training Setup**
```cpp
training.setupTraining();
```

#### What happens:
- **Updater algorithm** is configured (Gradient Descent or Kalman Filter)
- **Parallel training mode** is set up
- **Error metrics** are configured (RMSE for charges)
- Training focuses only on **charge prediction** (no forces or energies)

### 9. **Neighbor List Calculation**
```cpp
training.calculateNeighborLists();
```

#### What happens:
- For each structure, neighbor lists are computed
- These determine which atoms interact with each other
- Required for symmetry function calculations

### 10. **Main Training Loop**
```cpp
training.loop();
```

#### The core training process:

**For each epoch (iteration):**

1. **Error Calculation**
   - Current neural networks predict charges for all atoms
   - Errors are computed compared to reference charges
   - RMSE (Root Mean Square Error) is calculated for training and test sets

2. **Update Candidate Selection**
   - Structures with high charge prediction errors are selected for training
   - Selection can be random, sorted by error, or threshold-based
   - Number of updates per epoch is controlled by `charge_fraction`

3. **Gradient Computation**
   - For selected structures, gradients of the error with respect to neural network weights are computed
   - This involves backpropagation through the electrostatic neural networks

4. **Weight Update**
   - Neural network weights are updated using the chosen algorithm:
     - **Kalman Filter** (recommended, `updater_type = 1`)
     - **Gradient Descent** (`updater_type = 0`)

5. **Progress Monitoring**
   - Training and test charge RMSEs are printed
   - Learning curves are updated
   - Timing information is recorded

### 11. **Output Files Generated**

During and after training, several files are created:

#### Always Generated:
- **`nnp-train.log.XXXX`**: Detailed training log for each MPI process
- **`learning-curve.out.stage-1`**: Error evolution during training
- **`train.data`** and **`test.data`**: Training and test set data
- **`timing.out.stage-1`**: Performance timing information

#### If `write_weights_epoch` > 0:
- **`weightse.XXX.YYYYYY.out`**: Electrostatic NN weights at epoch YYYYYY
- **`hardness.XXX.YYYYYY.out`**: Atomic hardness parameters at epoch YYYYYY

#### If `write_traincharges` > 0:
- **`traincharges.YYYYYY.out`**: Charge prediction vs. reference comparison
- **`testcharges.YYYYYY.out`**: Test set charge comparison

### 12. **Training Completion**

After the specified number of epochs:
- Final weights are saved
- Best epoch (lowest RMSE) should be identified
- Files need to be renamed for Stage 2:
  - `weightse.XXX.YYYYYY.out` → `weightse.XXX.data`
  - `hardness.XXX.YYYYYY.out` → `hardness.XXX.data`

## Key Configuration Parameters for Stage 1

From the `input.nn` file:

```ini
# Neural network architecture for electrostatic networks
global_hidden_layers_electrostatic 2
global_nodes_electrostatic 15 15
global_activation_electrostatic t t l

# Training parameters
epochs 10
updater_type 1                    # Kalman filter
charge_fraction 1.000             # Use all charge data per epoch
task_batch_size_charge 1          # Batch size for charge updates
```

## What Happens Next?

After Stage 1 completion:
1. **Evaluate results**: Check `learning-curve.out.stage-1` for convergence
2. **Select best epoch**: Usually the one with lowest test RMSE
3. **Rename files**: Prepare for Stage 2 training
4. **Run Stage 2**: `nnp-train 2` to train short-range networks for energies and forces

The electrostatic neural networks trained in Stage 1 will be frozen during Stage 2, and only the short-range neural networks will be optimized for energy and force prediction.