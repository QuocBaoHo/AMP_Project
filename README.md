# 🔬 Antimicrobial Peptides (AMP) Classifier

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)

An advanced AI-powered system designed to classify and analyze the antimicrobial properties of Amino Acid sequences (Antimicrobial Peptides - AMP). 
The core highlight of this project is the integration of **Explainable AI (XAI)**, which not only provides highly accurate predictions but also "explains" which specific regions of the Amino Acid sequence contribute most to the antimicrobial properties.

## 🚀 Core Features

- **Rapid & Accurate Prediction**: Instantly determines whether a given peptide sequence is an AMP, accompanied by a confidence score.
- **Advanced AI Core**: Powered by a robust **Multi-branch Convolutional Neural Network (mCNN)** deep learning model and a state-of-the-art **ESM-2 Multitask Model** for simultaneous AMP and Toxicity prediction.
- **Explainable AI (XAI) - Feature Analysis**: 
  - Utilizes **Occlusion Sensitivity** to compute the relative importance of each individual Amino Acid in the sequence.
  - Visualizes the results via an interactive **HTML Heatmap**, highlighting the most critical Amino Acids in deep red.
- **Interactive Benchmark Dashboard**: A dedicated interface (`benchmark_app.py`) to run evaluations on the full `test_multitask` dataset (9.6k samples) and visually compare metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) across baseline models (1D CNN, Random Forest), the proposed mCNN model, and the SOTA ESM-2 Multitask model.
- **Modern Microservices Architecture**: Decouples a high-performance backend API (FastAPI) from a smooth, interactive frontend interface (Streamlit).

## 📊 Our Data

To teach our AI, we need good examples. We get these examples from trusted science databases. You can see how we clean this data in the `Data` folder. Here is the list of data we use:

### The Good Data
* **HemoPI-1:** 552 toxic examples (bad for red blood cells) and 552 safe examples.
* **HemoPI-2:** 552 toxic examples and 462 safe examples. The safe ones are special because they are weak, meaning they are safe AMPs.
* **HemoPI-3:** 885 toxic examples and 738 safe examples. Just like HemoPI-2, these are weak and safe.
* **RNN-Hem:** 1359 toxic examples and 1198 safe examples.
* **Hlppredfuse:** 1066 toxic examples and 2422 safe examples.
* **Viridiplantae:** A giant list of over 90,000 examples! We use this to show our big AI what AMPs look like. We tell the AI we don't know if they are toxic or not (`is_toxic = -1`).

### ⚠️ Why We Don't Use "AMP-Combined"

There is a big dataset called **AMP-Combined** (3007 toxic and 4172 safe examples) made by mixing some of the lists above. We **do not** use it because it has a big mistake (Data Bias):

* **Bad Mixing (Negative Sampling Flaw):** All the toxic examples are real AMPs. But all the safe examples are just random proteins that are not AMPs at all.
* **Confusing the AI (Spurious Correlation):** It makes the AI learn the wrong rule. The AI starts thinking: "If it looks like an AMP, it must be toxic! If it looks like a random protein, it must be safe!"
* **Missing the Goal (The Clinical Blindspot):** This bad mix completely removes the most important thing we want to find in medicine: **AMPs that kill bacteria but are safe for humans**. If we use this bad data, the AI will incorrectly think that safe, lab-made medicines are dangerous!

## 🛠️ Setup & Installation

### 1. System Requirements
Ensure you have the necessary dependencies installed:
```bash
pip install fastapi uvicorn pydantic tensorflow keras streamlit requests
```

### 2. Start the Backend (FastAPI Core)
Open a new terminal and run the following command to initialize the backend API:
```bash
python backend.py
```
*(The system will load the mCNN model and the API will listen at: `http://127.0.0.1:8000`)*

### 3. Start the Frontend (Streamlit UI)
Open another terminal (keeping the backend running) and launch the user interface:
```bash
streamlit run frontend.py
```
*(The web application will automatically open in your browser at: `http://localhost:8501`)*

### 4. Run the Benchmark Dashboard
Open another terminal and launch the benchmark interface:
```bash
streamlit run benchmark_app.py
```

## 🧬 Project Structure

```text
AMP_Project/
│
├── Data/                  # Data processing scripts and datasets
├── ModelTraining/         # Jupyter notebooks and scripts for training models
├── backend.py             # FastAPI Server: AI logic & XAI interpolation
├── frontend.py            # Streamlit Client: Web App UI & HTML Heatmap rendering
├── benchmark_app.py       # Streamlit Dashboard: Full benchmark comparison on multitask test set
├── mcnn_model.keras       # Multi-branch CNN model weights
├── cnn_model.keras        # Baseline 1D CNN model weights
├── tokenizer.pkl          # Tokenizer for mapping sequences to numerical vectors
├── vectorizer.pkl         # TF-IDF Vectorizer for Random Forest model
└── esm2_multitask_final/  # Saved ESM-2 Multitask Model for AMP & Toxicity prediction

**⚠️ NOTES:** 
* **Model Training:** If you want to train the models yourself, it is highly recommended to copy the contents of the `ModelTraining/` folder to a separate directory outside this project. This helps prevent clutter and keeps the main application directory clean.
* **Large Files:** The trained Random Forest model (`rf_model.pkl`) is >100MB and is NOT tracked by Git to avoid size limits. Please manually download/upload it and place it in the root directory before running the benchmark.
```

## 🔮 Future Roadmap

- [ ] **Phase 1: Explainable AI Deep Dive (Grad-CAM Integration)**
  - Implement Grad-CAM for the mCNN model to explicitly highlight hydrophobic amino acids, ensuring the model captures meaningful biological features rather than memorizing data.
- [x] **Phase 2: State-of-the-Art (SOTA) Model Integration**
  - Integrated Meta's ESM-2 Protein Language Model.
  - Applied transfer learning by freezing base layers and fine-tuning the model for multitask prediction (AMP & Toxicity).
- [ ] **Phase 3: Industrial MLOps Pipeline**
  - Consolidate all models under a unified FastAPI architecture.
  - Enhance the Streamlit dashboard for seamless API interaction.
  - Containerize the entire ecosystem using Docker for scalable, production-ready deployment.

---
*Project is currently in the Alpha / Demo development phase.*
