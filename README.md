# AI Mini Projects

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](#roadmap)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue)](#contributing)

A growing collection of small, focused **AI and machine learning projects**, each built to demonstrate one core idea end to end: data, model, evaluation and a runnable demo.

---

## Table of Contents
- [Projects](#projects)
- [Featured: Spam Classifier](#featured-spam-classifier)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Authors](#authors)

---

## Projects

| # | Project | Description | Tech | Status |
|---|---------|-------------|------|--------|
| 1 | [Spam Classifier](spam_classifier.py) | Classifies text messages as spam or not spam | Python, scikit-learn, TF-IDF, Naive Bayes | Done |
| 2 | Coming soon | Next mini project | | Planned |

---

## Featured: Spam Classifier

A beginner-friendly NLP project that detects spam messages using **TF-IDF features** and a **Multinomial Naive Bayes** model.

**How it works**
1. Text messages are converted into numeric features with TF-IDF (stop words removed).
2. A Naive Bayes classifier learns which words signal spam.
3. The model is evaluated on a held-out test split (accuracy, precision, recall, F1).
4. An interactive prompt lets you type any message and get a spam probability.

**Example**
```text
> Win a free prize now click here
SPAM (spam probability: 0.78)
> Can you send me the lecture notes
NOT SPAM (spam probability: 0.36)
```

**Results:** about 80% accuracy on a small built-in dataset. The dataset is intentionally tiny so the project runs instantly. See the roadmap for scaling it to a real dataset.

---

## Getting Started

**Prerequisites:** Python 3.9 or higher and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/devisriparvathi-junjuri/ai_mini_projects.git
cd ai_mini_projects

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the spam classifier
python spam_classifier.py
```

Type a message to classify it, or enter `quit` to exit.

---

## Repository Structure

```text
ai_mini_projects/
├── spam_classifier.py   # Spam detection model + interactive demo
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Roadmap

- [x] Spam classifier with TF-IDF and Naive Bayes
- [ ] Train on the full SMS Spam Collection dataset
- [ ] Compare Logistic Regression and SVM against Naive Bayes
- [ ] Add a confusion matrix plot
- [ ] Sentiment analysis mini project
- [ ] Simple web demo with Streamlit

---

## Contributing

Contributions are welcome. To add a project or improvement:

1. Create a branch: `git checkout -b feature/your-feature`
2. Make your changes and commit them
3. Push the branch and open a **pull request** against `main`

Please keep each mini project self-contained with a short description and clear run instructions.

---

## Authors

- **Devi Sri Parvathi Junjuri**: [GitHub](https://github.com/devisriparvathi-junjuri)
- Collaborator: Purna Chandra Nandigana :[GitHub](https://github.com/purnanandiganaa-alt)

---

<p align="center">If you find this useful, consider giving the repo a star.</p>
