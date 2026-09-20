# GraphLens: Explainable Graph Neural Network Predictions

> A lightweight explainability framework for understanding **Graph Neural Network (GNN) node predictions** using GraphSAGE, gradient-based feature attribution, and graph-neighborhood analysis.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.6.0-orange)
![GraphSAGE](https://img.shields.io/badge/GNN-GraphSAGE-purple)
![Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon-black)

---

## Overview

Graph Neural Networks are highly effective for learning from graph-structured data, but understanding **why a GNN makes a particular prediction** can be challenging.

**GraphLens** is an explainable GNN pipeline designed to make node-level predictions easier to inspect and interpret.

The project combines:

* **GraphSAGE** for node classification
* **Gradient-based feature attribution** for identifying influential feature dimensions
* **Neighborhood analysis** for examining the predictions of directly connected nodes
* **Multi-hop graph traversal** for providing additional local structural context
* **Human-readable explanation generation** for converting model outputs into concise explanations

The goal is not simply to report whether a prediction is correct, but to investigate:

> **Why did the model make this prediction, and what local graph evidence supports or conflicts with it?**

---

## Key Features

### Graph Neural Network Prediction

GraphLens trains a GraphSAGE model for node classification using graph structure and node features.

### Feature Attribution

Gradient-based analysis identifies feature dimensions that have a strong influence on the model's output for a selected node.

### Neighborhood Evidence

The framework examines the predictions of a node's direct neighbors and measures how strongly the local neighborhood agrees with the model's prediction.

### Multi-Hop Context

Graph traversal provides additional structural context beyond immediate neighbors.

### Error Analysis

GraphLens can identify and inspect:

* High-confidence correct predictions
* Low-confidence correct predictions
* High-confidence incorrect predictions
* Nodes with mixed neighborhood evidence

### Human-Readable Explanations

Instead of exposing raw tensors or debugging information, GraphLens converts model outputs into concise explanations that describe:

* The predicted class
* Prediction confidence
* Important feature dimensions
* Neighbor agreement
* Local graph context

---

# Explainability Pipeline

```text
                    Graph Dataset
                         │
                         ▼
              ┌─────────────────────┐
              │      GraphSAGE       │
              │  Node Classification │
              └──────────┬──────────┘
                         │
                         ▼
                Node Prediction
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
     Feature Attribution      Neighborhood Analysis
             │                       │
             ▼                       ▼
     Important Feature       Direct Neighbor Support
        Dimensions                    │
             │                       ▼
             └───────────┬───────────┘
                         │
                         ▼
                 Multi-Hop Context
                         │
                         ▼
              Human-Readable Explanation
```

---

# Datasets

GraphLens has been evaluated on two citation-network benchmarks.

## Cora

The Cora citation network contains:

| Property           |  Value |
| ------------------ | -----: |
| Nodes              |  2,708 |
| Feature dimensions |  1,433 |
| Classes            |      7 |
| Edges              | 10,556 |
| Test nodes         |    542 |

The seven classes are:

* Case_Based
* Genetic_Algorithms
* Neural_Networks
* Probabilistic_Methods
* Reinforcement_Learning
* Rule_Learning
* Theory

## CiteSeer

The CiteSeer citation network contains:

| Property           | Value |
| ------------------ | ----: |
| Nodes              | 3,327 |
| Feature dimensions | 3,703 |
| Classes            |     6 |
| Edges              | 9,104 |
| Test nodes         | 1,000 |

The six classes are:

* Agents
* AI
* DB
* IR
* ML
* HCI

---

# Model

GraphLens uses **GraphSAGE** for node classification.

The model consists of:

```text
Input Features
      │
      ▼
SAGEConv
      │
      ▼
Batch Normalization
      │
      ▼
ReLU
      │
      ▼
Dropout
      │
      ▼
SAGEConv
      │
      ▼
Class Predictions
```

### Cora Configuration

| Parameter         | Value |
| ----------------- | ----: |
| Input dimensions  | 1,433 |
| Hidden dimensions |   128 |
| Output classes    |     7 |
| GraphSAGE layers  |     2 |
| Dropout           |   0.5 |
| Learning rate     |  0.01 |
| Weight decay      |  5e-4 |
| Training epochs   |   100 |

### CiteSeer Configuration

The CiteSeer model uses the same two-layer GraphSAGE architecture while adapting the input and output dimensions to the dataset.

---

# Results

## Cora

The GraphSAGE model achieved:

**Test Accuracy: 89.30%**

| Metric                   |     Result |
| ------------------------ | ---------: |
| Test nodes               |        542 |
| Correct predictions      |        484 |
| Incorrect predictions    |         58 |
| Accuracy                 | **89.30%** |
| Average confidence       |     93.56% |
| Average neighbor support |     86.74% |

## CiteSeer

The GraphSAGE model achieved:

**Test Accuracy: 66.20%**

| Metric                   |     Result |
| ------------------------ | ---------: |
| Test nodes               |      1,000 |
| Correct predictions      |        662 |
| Incorrect predictions    |        338 |
| Accuracy                 | **66.20%** |
| Average confidence       |     70.37% |
| Average neighbor support |     76.86% |

The difference between the datasets also provides useful material for analyzing how graph structure, feature dimensionality, and neighborhood consistency affect GNN predictions.

---

# Explainability

GraphLens combines two complementary forms of evidence.

## 1. Feature-Level Evidence

Gradient-based attribution estimates which feature dimensions have the greatest influence on the prediction for a particular node.

For example:

```text
Node 1250
Prediction: Neural_Networks
Confidence: 100%

Important feature dimensions:
- 774
- 19
- 1014
```

These are reported as **feature dimensions**, rather than word-level explanations, because the current dataset representation does not provide a vocabulary mapping from feature indices to human-readable words.

---

## 2. Graph-Level Evidence

GraphSAGE incorporates information from neighboring nodes during message passing.

GraphLens therefore examines the model's predictions for directly connected neighbors.

For example:

```text
Node 1250
       │
       ├── Node 399   → Neural_Networks
       │
       └── Node 1251  → Neural_Networks
```

Both direct neighbors support the predicted class.

This provides local graph-context evidence for the prediction.

---

# Representative Explanation Cases

GraphLens supports several useful cases for analyzing GNN behavior.

### High-Confidence Correct

**Node 1250**

```text
Prediction: Neural_Networks
Actual: Neural_Networks
Confidence: 100%
Neighbor support: 2/2
```

The prediction agrees with the ground-truth label and both direct neighbors support the predicted class.

---

### Low-Confidence Correct

**Node 2648**

```text
Prediction: Neural_Networks
Actual: Neural_Networks
Confidence: 47.63%
Neighbors: 2
```

This represents a case where the model reaches the correct classification despite relatively low confidence and limited local support.

---

### High-Confidence Incorrect

**Node 1381**

```text
Prediction: Neural_Networks
Actual: Probabilistic_Methods
Confidence: 99.98%
Neighbors: 6
Neighbor support: 100%
```

This case is particularly useful for studying model errors because the model is highly confident and its local neighborhood also strongly supports the predicted class, yet the prediction does not match the ground-truth label.

---

### Mixed Neighborhood

**Node 624**

```text
Prediction: Genetic_Algorithms
Actual: Theory
Confidence: 69.04%
Neighbor support: 50%
```

This example demonstrates a less consistent local graph context where neighboring predictions are divided between classes.

---

# Why Graph Explanations Matter

A conventional classifier might provide:

```text
Prediction: Neural_Networks
Confidence: 100%
```

GraphLens adds local evidence:

```text
Prediction: Neural_Networks
Confidence: 100%

Important feature dimensions:
774, 19, 1014

Direct neighbors:
2 / 2 support Neural_Networks

Local context:
Node 399 → Node 15, Node 427
```

This makes the prediction easier to inspect from both the **feature perspective** and the **graph-structure perspective**.

---

# Example Human-Readable Explanation

For a correctly classified node:

> **Node 1250 is classified as Neural_Networks with 100% confidence. The prediction is supported by its local graph structure, as both directly connected neighbors are also classified as Neural_Networks. Gradient-based attribution identifies several feature dimensions as influential to the prediction, including features 774 and 19. A nearby second-hop connection through Node 399 provides additional local graph context.**

The explanation is intended to summarize model evidence rather than claim that any individual feature or neighbor is necessarily the causal reason for the prediction.

---

# Technology Stack

### Machine Learning

* Python
* PyTorch
* PyTorch Geometric
* GraphSAGE

### Explainable AI

* Gradient-based feature attribution
* Neighborhood-based analysis
* Multi-hop graph traversal
* Rule/template-based natural-language explanations

### Data Processing

* NumPy
* Pandas
* SciPy

### Development Environment

* Python 3.10
* Apple Silicon / MPS
* macOS

---

# Project Structure

```text
GraphLens-Explainable-Graph-Neural-Network-Predictions/
│
├── models/
│   └── GNNs/
│       └── sage.py
│
├── dataset/
│   ├── cora_orig/
│   └── citeseer/
│
├── run_sage_cora.py
├── run_sage_citeseer.py
│
├── evaluate_datasets.py
├── find_representative_nodes.py
│
├── explain_node.py
├── explain_one.py
├── explain_citeseer.py
├── final_graph_explanation.py
│
├── tag_graphsage_cora.py
├── generate_graphsage_natural_explanations.py
├── generate_local_explanations.py
├── save_graphsage_explanations.py
│
├── graphnarrator_structure.py
│
├── requirements-mac.txt
├── .gitignore
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/RoshiniEvangelin/GraphLens-Explainable-Graph-Neural-Network-Predictions.git

cd GraphLens-Explainable-Graph-Neural-Network-Predictions
```

Create a Python environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements-mac.txt
```

For Apple Silicon, PyTorch can use the Metal Performance Shaders backend when supported:

```python
import torch

print(torch.backends.mps.is_available())
```

---

# Running GraphSAGE

## Cora

Train GraphSAGE on Cora:

```bash
python run_sage_cora.py
```

The trained model is saved as:

```text
sage_cora.pt
```

## CiteSeer

Train GraphSAGE on CiteSeer:

```bash
python run_sage_citeseer.py
```

The trained model is saved as:

```text
sage_citeseer.pt
```

---

# Evaluating the Models

Run the dataset evaluation:

```bash
python evaluate_datasets.py
```

This reports:

* Test accuracy
* Correct and incorrect predictions
* Prediction confidence
* Neighbor support
* High-confidence errors

---

# Finding Representative Nodes

GraphLens can automatically identify useful nodes for explanation and error analysis:

```bash
python find_representative_nodes.py --dataset cora
```

The script identifies examples such as:

```text
HIGH-CONFIDENCE CORRECT
LOW-CONFIDENCE CORRECT
HIGH-CONFIDENCE INCORRECT
MIXED NEIGHBORHOOD
STRONG NEIGHBORHOOD AGREEMENT
```

---

# Generating an Explanation

For example, to inspect Cora node 1250:

```bash
python final_graph_explanation.py --dataset cora --id 1250
```

The resulting explanation combines:

1. Model prediction
2. Confidence
3. Feature attribution
4. Direct-neighbor predictions
5. Neighbor support
6. Multi-hop graph context

---

# Explainability Method

The current GraphLens explanation pipeline can be summarized as:

```text
Node
 │
 ├──► GraphSAGE Prediction
 │        │
 │        └──► Class + Confidence
 │
 ├──► Gradient Attribution
 │        │
 │        └──► Important Feature Dimensions
 │
 ├──► Neighbor Analysis
 │        │
 │        └──► Local Class Support
 │
 └──► Graph Traversal
          │
          └──► Multi-Hop Context
                   │
                   ▼
          Human-Readable Explanation
```

---

# Important Limitations

### Feature Interpretability

The current Cora feature representation provides feature dimensions rather than a direct mapping from each feature index to a human-readable word.

Therefore, GraphLens reports:

```text
Feature 774
```

rather than claiming:

```text
Feature 774 = "specific word"
```

unless an explicit feature-to-word mapping is available.

### Gradient Attribution

Gradient-based attribution measures local model sensitivity. It should not automatically be interpreted as proof that a feature is the causal reason for a prediction.

### Neighborhood Evidence

Neighbor agreement provides graph-context evidence. A supporting neighbor does not necessarily mean that the neighbor caused the prediction.

### Dataset Representation

Cora and CiteSeer are evaluated using their respective preprocessing and graph representations. Dataset-specific node ordering and masks should remain consistent between training and evaluation.

### Explanation Generation

The current natural-language explanation layer is lightweight and template/rule based. It is designed for transparent local explanations rather than open-ended language generation.

---

# Future Improvements

Potential extensions include:

* [ ] Add SHAP-based graph explanations
* [ ] Add Integrated Gradients
* [ ] Add GNNExplainer
* [ ] Add PGExplainer
* [ ] Visualize important subgraphs
* [ ] Add feature-to-word mappings where available
* [ ] Compare GraphSAGE with GCN and GAT
* [ ] Add explanation faithfulness metrics
* [ ] Evaluate explanation stability
* [ ] Build an interactive Streamlit dashboard
* [ ] Add support for additional graph datasets
* [ ] Add graph-level and edge-level explanations

---

# Relationship to GraphNarrator

GraphLens was developed as a **Mac-compatible, lightweight explainability pipeline inspired by the broader idea of generating human-readable explanations for GNN predictions**.

The original GraphNarrator project contains a larger CUDA-oriented pipeline involving additional components such as language-model-based explanation generation and training infrastructure.

GraphLens focuses specifically on:

```text
GraphSAGE
   +
Gradient Attribution
   +
Neighborhood Analysis
   +
Graph Traversal
   +
Human-Readable Explanations
```

This separation keeps the project lightweight and reproducible on Apple Silicon systems.

---

# Research / Engineering Goals

GraphLens explores several questions relevant to explainable graph machine learning:

1. **Can feature attribution identify the dimensions most influential to a GNN prediction?**

2. **Does local neighborhood agreement provide useful evidence for interpreting predictions?**

3. **Can high-confidence errors be analyzed using graph structure and feature attribution?**

4. **Can raw GNN outputs be transformed into concise, human-readable explanations?**

5. **How does explanation behavior differ between graph datasets such as Cora and CiteSeer?**

---

# Reproducibility

The project records the model configuration, dataset-specific evaluation results, and representative node cases used during explanation analysis.

For reproducible experiments:

* Use the same dataset preprocessing for training and evaluation.
* Keep the model architecture consistent with the saved checkpoint.
* Use the corresponding dataset-specific checkpoint.
* Run explanation scripts against the same graph representation used during training.

---

# License

This repository is intended as a research and portfolio project.

Please review the licensing terms of any datasets, libraries, or upstream research code used by the project before redistribution.

---

# Author

**Roshini Evangelin**

MS Computer Science
Purdue University Northwest

Interested in:

* Machine Learning
* Artificial Intelligence
* Explainable AI
* Graph Neural Networks
* Computer Vision
* Applied Deep Learning

---

## Project

**GraphLens: Explainable Graph Neural Network Predictions**

GitHub:

https://github.com/RoshiniEvangelin/GraphLens-Explainable-Graph-Neural-Network-Predictions
