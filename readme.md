ComTree: Official Code Repository for ACM MM 2025 Paper
===

This is the official source code implementation for the ACM Multimedia 2025 paper: **Beyond Interpretability: Exploring the Comprehensibility of Adaptive Video Streaming through Large Language Models**

Introduction
---

In the field of Adaptive Bitrate (ABR) streaming, deep learning algorithms are notoriously difficult to maintain due to their "black-box" nature. While existing work has improved "interpretability" by converting these models into decision trees, this is not synonymous with "**comprehensibility**"—the ability for developers to easily understand and optimize the models. Complex decision logic remains a significant barrier.

To address this, we introduce **ComTree**—a novel framework designed to generate highly **comprehensible** ABR algorithms. ComTree's core operates in two stages: first, it constructs a "**Rashomon Set**" containing multiple, structurally diverse decision trees with equivalent high performance. Second, it innovatively uses **Large Language Models (LLMs)** as expert proxies to select the final model with the clearest and most human-understandable logic from this set.

Ultimately, ComTree produces ABR algorithms that are simpler in structure and easier for developers to maintain and optimize, all without compromising performance.

File Structure
---

```
.
├── experiment_logs/
│   ├── logs/              # Stores raw log data
│   └── plot_fig.py        # Python script to reproduce figures from the paper (Fig4, Fig5, Fig6)
│
├── src/
│   ├── Rashomon_Set_Construction/
│   │   ├── loop.py            # Script to generate the dataset required for the Rashomon set
│   │   └── test_treefarm.py   # Core script to build the Rashomon set
│   │
│   ├── Large_Language_Model_for_Interpretability_Assessment/
│   │   └── server.py          # Service script to evaluate decision tree comprehensibility using an LLM
│   │
│   └── Test/
│       ├── test_tree.py       # Script to test the performance of ComTree(P) and ComTree_C
│       └── ...                # Other testing-related files
│
└── README.md                  # This README file
```

Environment Setup
---

Before running the code, please ensure you have installed all the necessary dependencies.

1. **Base Dependencies**:
    First, install the base libraries required by `treefarms`.

   ```bash
   pip3 install attrs packaging editables pandas scikit-learn sortedcontainers gmpy2 matplotlib
   ```

2. **Core Libraries**:
    Next, install `treefarms` and `fastapi_poe`.

   ```bash
   pip3 install treefarms
   pip3 install fastapi_poe
   ```

Reproducing Results
---

If you wish to reproduce the figures from our paper, follow these steps:

1. Navigate to the `experiment_logs/` directory.

2. Run the `plot_fig.py` script. It will use the data in the `logs/` directory to regenerate Figures 4, 5, and 6.

   ```bash
   cd experiment_logs/
   python plot_fig.py
   ```

Running the ComTree Framework
---

The execution of ComTree is divided into two main stages: **Rashomon Set Construction** and **LLM-based Assessment**.

### Stage 1: Rashomon Set Construction

This stage aims to generate a set of decision tree models that have comparable predictive accuracy but differ in structure.

1. **Generate the Dataset**:
    Navigate to the `src/Rashomon_Set_Construction/` directory and run `loop.py` to generate the required data.

   ```bash
   cd src/Rashomon_Set_Construction/
   python loop.py
   ```

2. **Configure Paths**:
    Open the `test_treefarm.py` file, find and modify the following two variables to point to your local paths:

   - `path_model`: The storage path for the model file.
   - `path_columns`: The path to the data columns definition file.

3. **Build the Set**:
    Run the `test_treefarm.py` script to execute the TreeFarms algorithm and obtain the Rashomon set.

   ```bash
   python test_treefarm.py
   ```

### Stage 2: LLM-based Assessment

This stage uses a Large Language Model to assess the comprehensibility of the decision trees in the Rashomon set.

1. **Configure API Key and Path**:
    Open the `src/Large_Language_Model_for_Interpretability_Assessment/server.py` file.

   - Replace the `api_key` variable with your own Poe API key.
   - Update the `your_path` variable to the actual storage path of the Rashomon set generated in Stage 1.

2. **Start the Assessment Service**:
    Run the `server.py` script to start the assessment process. This service will call the LLM to analyze each decision tree.

   ```bash
   cd ../Large_Language_Model_for_Interpretability_Assessment/
   python server.py
   ```

Quick Test of ComTree Performance
---

If you just want to quickly verify the performance of ComTree, we provide a convenient test script.

1. Navigate to the `src/Test/` directory.

2. Run `test_tree.py`.

   ```bash
   cd src/Test/
   python test_tree.py
   ```

- **Default Behavior**: The script runs and outputs the performance results for **ComTree(P)** by default.

- Testing **ComTree_C**

   To test ComTree_C, find and uncomment the following line in the 

  ```
  test_tree.py
  ```

   file:

  ```python
  # import ComTree_C as decision_tree
  ```

  Change it to:

  ```python
  import ComTree_C as decision_tree
  ```

  Then, run the script again.

