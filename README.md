# AI Workflow Engineer Handbook – Predictive Test Selection

This repository demonstrates how to build and deploy a **predictive test selection system** for software engineering workflows. Instead of running all regression tests, the system learns from historical commit and test failure data to **prioritize tests most likely to fail**, reducing build times and saving compute resources.

Inspired by real-world responsibilities of an **AI Workflow Engineer**, this project illustrates how AI can make developer workflows faster, more reliable, and cost-effective.


## Key Features

- **Commit Feature Engineering**: Captures metadata such as files changed, lines added/removed, subsystems, and author history.  
- **Predictive Modeling (XGBoost)**: Classifies which tests are most likely to fail for a given commit.  
- **Selective Test Execution**: Runs only the top predicted tests, reducing regression time dramatically.  
- **Metrics & Savings**: Demonstrates cycle time reduction and cost efficiency (e.g., from 12,000 tests at 50 minutes → ~300 targeted tests at ~7 minutes).  
- **Evidence Pack Outputs**: Generates reproducible results with metrics, logs, and experiment configurations.

## Project Structure

```bash
ai-workflow-predictive-test-selection/
│
├── data/
│   ├── commit_features.csv     # Dummy commit metadata (files, lines, subsystem, author history)
│   ├── test_failures.csv       # Dummy labels for test pass/fail
│
├── notebooks/
│   └── predictive_test_selection.ipynb   # Walkthrough notebook
│
├── src/
│   ├── train.py                # Model training script
│   ├── inference.py            # Test selection given commit features
│   └── utils.py                # Helper functions
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # Open source license
````

## Installation

Clone the repo:

```bash
git clone https://github.com/richardbigegapersonal/ai-workflow-predictive-test-selection.git
cd ai-workflow-predictive-test-selection
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Usage

### 1. Train the model

```bash
python src/train.py --data data/commit_features.csv --labels data/test_failures.csv
```

### 2. Run inference (select tests for a new commit)

```bash
python src/inference.py --data data/commit_features.csv --threshold 0.05
```

### 3. Explore the notebook

Open the Jupyter Notebook for an end-to-end walkthrough:

```bash
jupyter notebook notebooks/predictive_test_selection.ipynb
```

## Example Results

* Baseline regression suite: **12,000 tests**, \~**50 min runtime**
* Predicted relevant tests: **\~300 tests**, \~**7 min runtime**
* Compute savings: **\~86% reduction** per build
* Accuracy: >90% coverage of failing tests with <1% false skips

## Tech Stack

* **Languages**: Python 3.11
* **Libraries**: pandas, numpy, xgboost, scikit-learn
* **Tools**: Jupyter, GitHub Actions (for CI/CD), matplotlib/seaborn (for analysis)

## Evidence Pack (Outputs)

The project produces evidence artifacts for reproducibility:

* Model file (`.json`)
* Feature importance plots
* Metrics summary (`.csv`, `.json`)
* Selected test lists per commit

## Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a PR.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This repository is part of the **AI Workflow Engineer Handbook** series, illustrating how AI/ML techniques can transform developer workflows in real-world engineering environments.

