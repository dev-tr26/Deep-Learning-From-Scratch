# IPL Match Outcome Prediction with Deep Learning

This project demonstrates an **end-to-end deep learning workflow** using the IPL matches dataset. The main goal is to explore the impact of **different weight initialization techniques** on model performance.

## Dataset

The dataset contains historical IPL match records with columns such as:

- data of ipl matches from 2008 to 2017

* Categorical features: `team1`, `team2`, `city`, `venue`, `toss_winner`, `toss_decision`, `player_of_match`, `umpire1`, `umpire2`, `umpire3`
* Numerical features: `season`, `dl_applied`, `win_by_runs`, `win_by_wickets`
* Target: `winner` (the match-winning team)

Preprocessing steps:

1. Dropped irrelevant columns (`id`, `date`, `result`)
2. Label-encoded categorical variables
3. Standardized numerical features
4. Split into training and testing sets

## Model Architecture

We use a simple **feedforward neural network** implemented in PyTorch:

* Input layer: size = number of encoded features
* Hidden layers: two fully connected layers with ReLU activation (128 units each)
* Output layer: size = number of unique teams (classification task)

```
Input → Linear(128) → ReLU → Linear(128) → ReLU → Linear(Output) → Softmax
```

Loss function: **CrossEntropyLoss**
Optimizer: **Adam (lr=0.001)**
Epochs: 30

## Weight Initialization Techniques

We compare four popular weight initialization strategies:

1. **Xavier Initialization** (`nn.init.xavier_uniform_`)

   * Balances variance between layers for stable gradients.

2. **Kaiming Initialization** (`nn.init.kaiming_uniform_`)

   * Designed for ReLU activations, helps deeper networks converge faster.

3. **Normal Initialization** (`nn.init.normal_`)

   * Weights sampled from a Gaussian distribution (mean=0, std=0.01).

4. **Uniform Initialization** (`nn.init.uniform_`)

   * Weights uniformly distributed in a small range (−0.1, 0.1).

## Evaluation Metrics

* **Training Loss Curves** across epochs
* **Test Accuracy Curves** across epochs
* **Confusion Matrices** for each initialization method
* **Classification Reports** (precision, recall, F1-score per team)

## Observations

* **Xavier**: Generally stable training and good convergence.
* **Kaiming**: Performs better when using ReLU activations, faster convergence.
* **Normal & Uniform**: Training is less stable, often slower, sometimes worse accuracy.

