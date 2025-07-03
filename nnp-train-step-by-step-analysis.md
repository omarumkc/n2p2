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