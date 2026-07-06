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
- **Interactive Benchmark Dashboard**: A dedicated interface (`benchmark_app.py`) to run evaluations on the full `viri_test` dataset (28.5k samples) and visually compare metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) across baseline models (1D CNN, Random Forest) and the proposed mCNN model.
- **Modern Microservices Architecture**: Decouples a high-performance backend API (FastAPI) from a smooth, interactive frontend interface (Streamlit).

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
├── backend.py             # FastAPI Server: AI logic & XAI interpolation
├── frontend.py            # Streamlit Client: Web App UI & HTML Heatmap rendering
├── benchmark_app.py       # Streamlit Dashboard: Full benchmark comparison on viri_test
├── mcnn_model.keras       # Multi-branch CNN model weights
├── cnn_model.keras        # Baseline 1D CNN model weights
├── tokenizer.pkl          # Tokenizer for mapping sequences to numerical vectors
├── vectorizer.pkl         # TF-IDF Vectorizer for Random Forest model
└── esm2_multitask_final/  # Saved ESM-2 Multitask Model for AMP & Toxicity prediction

**⚠️ NOTE FOR LARGE FILES:** 
The trained Random Forest model (`rf_model.pkl`) is >100MB and is NOT tracked by Git to avoid size limits. Please manually download/upload it and place it in the root directory before running the benchmark.
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
