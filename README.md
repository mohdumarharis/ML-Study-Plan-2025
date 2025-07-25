# ML-Study-Plan-2025

# Overview of ML Algorithms and Mathematics

The algorithms are categorized into basic, intermediate, and advanced, with their mathematical foundations, forming the basis of the plan.

## Basic Algorithms (Supervised Learning)

- **Linear Regression:**  
  y = wᵀx + b, minimize MSE:  
  J(w) = (1/m) Σ (yᵢ - ŷᵢ)²

- **Logistic Regression:**  
  P(y=1|x) = σ(wᵀx), minimize cross-entropy loss

- **Naive Bayes:**  
  P(y|x) ∝ P(y) ∏ P(xᵢ|y)

## Intermediate Algorithms

- **Decision Trees:**  
  Entropy: H = -Σ pᵢ log(pᵢ)  
  Information Gain

- **Random Forests:**  
  Ensemble via bagging, variance reduction

- **Support Vector Machines (SVM):**  
  Maximize margin: 2 / ||w||  
  Kernel trick: K(xᵢ, xⱼ)

## Advanced Algorithms

- **K-Nearest Neighbors (KNN):**  
  Euclidean distance: d(x, y) = sqrt(Σ (xᵢ - yᵢ)²)

- **K-Means Clustering:**  
  Minimize: J = Σₖ Σ_{x ∈ Cₖ} ||x - μₖ||²

- **Principal Component Analysis (PCA):**  
  Eigenvalue decomposition of covariance matrix

- **Neural Networks:**  
  Backpropagation, ∂L/∂w via chain rule

- **Gradient Boosting (XGBoost, LightGBM):**  
  Minimize: L = Σ l(yᵢ, ŷᵢ) + Ω(f)


---

## Mathematical Foundations

- **Linear Algebra:** Matrices, eigenvalues, SVD.
- **Calculus:** Gradients, optimization (Gradient Descent).
- **Probability:** Distributions, Bayes’ theorem.
- **Optimization:** Convex optimization, Lagrange multipliers.

---

# 3-Month Study Plan with Day-Wise Breakdown

The 12-week plan progresses from foundations to advanced topics, with each week requiring 15–20 hours (~2–3 hours/day, 5–6 days/week). Daily tasks cover:

- **Theory/Mathematics:** Reading, derivations (6–8 hours/week).
- **Coding:** From scratch + libraries (scikit-learn, PyTorch) (5–7 hours/week).
- **Practice/Projects:** Datasets, experimentation (3–5 hours/week).

---

## Month 1: Foundations and Basic Algorithms

### Week 1: Mathematical Prerequisites

**Goal:** Build a foundation in linear algebra, calculus, and probability.  
**Resources:**
- Mathematics for Machine Learning by Deisenroth et al. (Ch. 1–3, free PDF).
- Khan Academy: Linear Algebra, Calculus (free).
- 3Blue1Brown YouTube: “Essence of Linear Algebra,” “Calculus” (free).

**Day-Wise Plan:**

- **Day 1 (Mon):** Study vectors, matrices, dot products (Ch. 1). Watch 3Blue1Brown “Vectors” (2 hours).
- **Day 2 (Tue):** Study matrix operations, determinants. Solve exercises (Khan Academy) (2.5 hours).
- **Day 3 (Wed):** Study derivatives, partial derivatives (Ch. 2). Watch 3Blue1Brown “Derivatives” (2.5 hours).
- **Day 4 (Thu):** Study probability, Bayes’ theorem (Ch. 3). Solve probability exercises (2.5 hours).
- **Day 5 (Fri):** Code matrix operations (multiplication, inverse) in NumPy. Solve linear algebra exercises (2.5 hours).
- **Day 6 (Sat):** Review weak areas, revisit Khan Academy problems (2.5 hours).

_Total Time: 15 hours._

---

### Week 2: Linear Regression

**Goal:** Understand and implement linear regression.  
**Resources:**
- Introduction to Statistical Learning (ISLR) by James et al. (Ch. 3, free PDF).
- Andrew Ng’s Coursera ML course (Weeks 1–2, free audit).
- Scikit-learn Linear Regression docs.

**Day-Wise Plan:**

- **Day 1 (Mon):** Study model, MSE, Normal Equation. Derive $ w = (X^TX)^{-1}X^Ty $ (2.5 hours).
- **Day 2 (Tue):** Study Gradient Descent. Derive update rule. Watch Andrew Ng (2.5 hours).
- **Day 3 (Wed):** Implement Linear Regression from scratch (NumPy) (2.5 hours).
- **Day 4 (Thu):** Apply to Boston Housing dataset (scikit-learn). Plot predictions (2.5 hours).
- **Day 5 (Fri):** Study Ridge regularization. Derive L2 penalty. Apply Ridge (scikit-learn) (2.5 hours).
- **Day 6 (Sat):** Compare Linear vs. Ridge on Boston Housing. Document results (2.5 hours).

_Total Time: 15 hours._

---

### Week 3: Logistic Regression and Naive Bayes

**Goal:** Master classification algorithms.  
**Resources:**
- ISLR (Ch. 4).
- Scikit-learn Logistic Regression, Naive Bayes docs.
- StatQuest YouTube: “Logistic Regression,” “Naive Bayes” (free).

**Day-Wise Plan:**

- **Day 1 (Mon):** Study Logistic Regression, sigmoid, cross-entropy. Derive loss (2.5 hours).
- **Day 2 (Tue):** Implement Logistic Regression from scratch (NumPy) (2.5 hours).
- **Day 3 (Wed):** Study Naive Bayes, Gaussian variant. Derive model (2.5 hours).
- **Day 4 (Thu):** Implement Gaussian Naive Bayes from scratch (2.5 hours).
- **Day 5 (Fri):** Apply Logistic Regression and Naive Bayes to Iris dataset (scikit-learn) (2.5 hours).
- **Day 6 (Sat):** Analyze results, plot decision boundaries (2.5 hours).

_Total Time: 15 hours._

---

### Week 4: Evaluation Metrics and Consolidation

**Goal:** Learn evaluation techniques and consolidate basics.  
**Resources:**
- ISLR (Ch. 5).
- Kaggle: “Model Evaluation Tutorial” (free).
- Scikit-learn metrics docs.

**Day-Wise Plan:**

- **Day 1 (Mon):** Study regression metrics (MSE, RMSE, R²). Read ISLR (2.5 hours).
- **Day 2 (Tue):** Study classification metrics (Accuracy, F1, ROC-AUC). Watch Kaggle tutorial (2.5 hours).
- **Day 3 (Wed):** Evaluate Week 2–3 models with metrics (scikit-learn) (2.5 hours).
- **Day 4 (Thu):** Mini-project: Apply Linear and Logistic Regression to Titanic dataset (2.5 hours).
- **Day 5 (Fri):** Mini-project: Compare models, plot ROC curves (2.5 hours).
- **Day 6 (Sat):** Document findings, write summary (2.5 hours).

_Total Time: 15 hours._

---

## Month 2: Intermediate Algorithms

**Goal:** Understand tree-based models, SVMs, and unsupervised learning.

### Week 5: Decision Trees and Random Forests

**Goal:** Master tree-based models.  
**Resources:**
- ISLR (Ch. 8).
- Hands-On Machine Learning by Géron (Ch. 6).
- Scikit-learn Random Forest docs.

**Day-Wise Plan:**

- **Day 1 (Mon):** Study Decision Trees, Gini, Entropy. Derive Information Gain (2.5 hours).
- **Day 2 (Tue):** Implement Decision Tree from scratch (Python) (2.5 hours).
- **Day 3 (Wed):** Study Random Forests, bagging. Read Géron (2.5 hours).
- **Day 4 (Thu):** Apply Random Forest to Wine dataset (scikit-learn) (2.5 hours).
- **Day 5 (Fri):** Tune Random Forest (n_estimators, max_depth) (2.5 hours).
- **Day 6 (Sat):** Compare Decision Tree vs. Random Forest. Document results (2.5 hours).

_Total Time: 15 hours._

---

### Week 6: Support Vector Machines

**Goal:** Understand SVMs and kernels.  
**Resources:**
- ISLR (Ch. 9).
- Pattern Recognition and Machine Learning by Bishop (Ch. 7).
- StatQuest YouTube: “SVM” (free).

**Day-Wise Plan:**

- **Day 1 (Mon):** Study hard-margin SVM, derive objective. Watch StatQuest (2.5 hours).
- **Day 2 (Tue):** Study soft-margin SVM, kernel trick. Derive dual form (2.5 hours).
- **Day 3 (Wed):** Implement SVM with linear kernel (scikit-learn) (2.5 hours).
- **Day 4 (Thu):** Apply SVM to downsampled MNIST (RBF kernel) (2.5 hours).
- **Day 5 (Fri):** Compare linear vs. RBF kernels. Analyze results (2.5 hours).
- **Day 6 (Sat):** Revisit Lagrange multipliers, document findings (2.5 hours).

_Total Time: 15 hours._

---

### Week 7: K-Means Clustering and KNN

**Goal:** Master unsupervised and instance-based learning.  
**Resources:**
- ISLR (Ch. 10).
- Géron’s book (Ch. 9).
- Scikit-learn clustering docs.

**Day-Wise Plan:**

- **Day 1 (Mon):** Study K-Means, derive objective. Read ISLR (2.5 hours).
- **Day 2 (Tue):** Implement K-Means from scratch (Python) (2.5 hours).
- **Day 3 (Wed):** Study KNN, distance metrics. Read Géron (2.5 hours).
- **Day 4 (Thu):** Implement KNN from scratch (Python) (2.5 hours).
- **Day 5 (Fri):** Apply K-Means to Mall Customers dataset (scikit-learn) (2.5 hours).
- **Day 6 (Sat):** Apply KNN to Iris dataset, compare with Naive Bayes (2.5 hours).

_Total Time: 15 hours._

---

### Week 8: Principal Component Analysis (PCA)

**Goal:** Understand dimensionality reduction.  
**Resources:**
- ISLR (Ch. 10).
- 3Blue1Brown YouTube: “Eigenvalues and Eigenvectors” (free).
- Scikit-learn PCA docs.

**Day-Wise Plan:**

- **Day 1 (Mon):** Study PCA, covariance matrix. Derive eigenvalue decomposition (2.5 hours).
- **Day 2 (Tue):** Watch 3Blue1Brown, revisit PCA math (2.5 hours).
- **Day 3 (Wed):** Implement PCA from scratch (NumPy) (2.5 hours).
- **Day 4 (Thu):** Apply PCA to MNIST for visualization (scikit-learn) (2.5 hours).
- **Day 5 (Fri):** Experiment with number of components, analyze variance (2.5 hours).
- **Day 6 (Sat):** Document PCA results, summarize math (2.5 hours).

_Total Time: 15 hours._

---

## Month 3: Advanced Algorithms and Projects

**Goal:** Master neural networks, ensemble methods, and apply skills to projects.

### Week 9: Neural Networks (Feedforward)

**Goal:** Understand neural networks and backpropagation.  
**Resources:**
- Deep Learning by Goodfellow et al. (Ch. 6, free PDF).
- Fast.ai: “Practical Deep Learning” (free).
- PyTorch tutorials (free).

**Day-Wise Plan:**

- **Day 1 (Mon):** Study neural network architecture, activations. Read Goodfellow (2.5 hours).
- **Day 2 (Tue):** Derive backpropagation, chain rule. Watch Fast.ai (2.5 hours).
- **Day 3 (Wed):** Implement neural network from scratch (NumPy) (2.5 hours).
- **Day 4 (Thu):** Apply PyTorch to Fashion MNIST dataset (2.5 hours).
- **Day 5 (Fri):** Tune layers, activation functions in PyTorch (2.5 hours).
- **Day 6 (Sat):** Document results, revisit backpropagation math (2.5 hours).

_Total Time: 15 hours._

---

### Week 10: Gradient Boosting (XGBoost, LightGBM)

**Goal:** Master ensemble boosting methods.  
**Resources:**
- XGBoost documentation (free).
- Géron’s book (Ch. 7).
- Kaggle: “XGBoost Tutorial” (free).

**Day-Wise Plan:**

- **Day 1 (Mon):** Study boosting, additive modeling. Read XGBoost docs (2.5 hours).
- **Day 2 (Tue):** Derive XGBoost objective, second-order gradients (2.5 hours).
- **Day 3 (Wed):** Apply XGBoost to House Prices dataset (Kaggle) (2.5 hours).
- **Day 4 (Thu):** Tune XGBoost (learning rate, max_depth) (2.5 hours).
- **Day 5 (Fri):** Compare XGBoost vs. Random Forest on same dataset (2.5 hours).
- **Day 6 (Sat):** Document results, summarize boosting math (2.5 hours).

_Total Time: 15 hours._

---

### Week 11: Capstone Project (Part 1)

**Goal:** Apply multiple algorithms to a real-world dataset.  
**Resources:**
- Kaggle datasets and notebooks (free).
- Scikit-learn, XGBoost, PyTorch docs.

**Day-Wise Plan:**

- **Day 1 (Mon):** Select Kaggle dataset, preprocess (handle missing values, encode) (2.5 hours).
- **Day 2 (Tue):** Apply Logistic Regression and Random Forest (2.5 hours).
- **Day 3 (Wed):** Apply XGBoost, evaluate with cross-validation (2.5 hours).
- **Day 4 (Thu):** Compare model performance (metrics, plots) (2.5 hours).
- **Day 5 (Fri):** Document initial results, identify best model (2.5 hours).
- **Day 6 (Sat):** Explore feature importance, refine preprocessing (2.5 hours).

_Total Time: 15 hours._

---

### Week 12: Capstone Project (Part 2) and Review

**Goal:** Optimize models and review all concepts.  
**Resources:**
- Kaggle tutorials on ensemble methods.
- ISLR, Géron’s book for review.

**Day-Wise Plan:**

- **Day 1 (Mon):** Optimize models (try stacking, feature engineering) (2.5 hours).
- **Day 2 (Tue):** Finalize project, improve best model (2.5 hours).
- **Day 3 (Wed):** Write project report or create presentation (2.5 hours).
- **Day 4 (Thu):** Review Linear Regression, Logistic, SVM math (2.5 hours).
- **Day 5 (Fri):** Review Neural Networks, XGBoost derivations (2.5 hours).
- **Day 6 (Sat):** Revisit weak areas, summarize learnings (2.5 hours).

_Total Time: 15 hours._

---

## Minimum Weekly Hours

**Total:** 15–20 hours/week (~2–3 hours/day, 5–6 days/week).

**Breakdown:**

- Theory/Mathematics: 6–8 hours.
- Coding: 5–7 hours.
- Practice/Projects: 3–5 hours.

**Flexibility:** If 15–20 hours is too intensive, extend to 4–5 months.  
If you have 25–30 hours/week, add more datasets or explore advanced topics (e.g., deep learning variants).
