## Mini AI Pipeline — News Classification


### 📂 File Structure
```
.
├── report.pdf                  ← Overleaf-generated final project report
├── requirements.txt
├── README.md
├── notebooks/
│   └── news_classification.ipynb
└── src/
    └── news_classification.py
```


### 🚀 Running the AI Pipeline
Install dependencies:
```bash
pip install -r requirements.txt
```
To reproduce the results shown in the project report, run:
```bash
python src/news_classification.py
```
This script trains and evaluates the DistilBERT-based news classifier and prints the test accuracy and F1 score.

For step-by-step and interactive execution, ```notebooks/news_classification.ipynb``` is provided. The script ```src/news_classification.py``` is the Python equivalent of the notebook and performs the same pipeline and produces the same results, optimized for command-line execution (e.g., on local GPU or remote workstation).



### 💡 Notes
- No checkpoints or large artifacts are included in the repository.
- The DistilBERT model is downloaded automatically during training.
- All results are fully reproducible using the script above.