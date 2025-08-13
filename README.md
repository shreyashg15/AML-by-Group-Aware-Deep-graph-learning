# Anti-Money Laundering by Group-Aware Deep Graph Learning (GAGNN)

This project uses Graph Neural Networks to detect individual and group-based suspicious money laundering activities.

# Anti-Money Laundering (AML) Detection with Group-Aware GNN (GAGNN)

## 📌 Overview
This project is a **final year engineering project** for detecting fraudulent transaction patterns in financial data using a **Group-Aware Graph Neural Network (GAGNN)**.  
It uses **PyTorch Geometric** for graph learning, **Streamlit** for the frontend, and **PyVis** for interactive transaction graph visualization.

The system:
- Loads transaction datasets
- Builds a transaction graph
- Trains a GAGNN model to predict fraud risk at **node** and **group** levels
- Displays results in an interactive dashboard

---
Features:
1.Unsupervised GAE + Clustering
Detect suspicious/fraudulent transaction groups without labeled data.

2.Model Training & Loading
->First run trains the model (unsupervised GAE).
->Saves predictions and model weights for instant reuse.
->Supports custom save names and fixed filenames.
->Dataset fingerprinting for reproducible saves.

3.Performance Features
->Early stopping with patience.
->Mini-batch GPU/CPU training.
->Live progress bar & epoch loss updates.
->Automatic model loading if data fingerprint already trained.

4.Graph Visualization
->Interactive Fraud Graph with hover tooltips and fraud probability color scale.
->Static Tree Graph layout for alternative network structure view.
->Directional arrows to show money flow (configurable).
->Node color intensity proportional to predicted fraud probability.

5.Analysis Tables
->Sortable summary table with fraud probabilities, group numbers, and statistics.
->High-risk nodes table filtered by a fraud probability threshold.

6.File Management
->All models saved in models/ folder.
->Dataset fingerprints saved in data_fingerprints/ for tracking.
->Download buttons for predictions and models.

Pipeline Architecture:
1.Data Input
->CSV file with transaction records (source, target, amount).
->Supports column renaming from nameOrig, nameDest.

2.Graph Construction
->Builds a directed graph (nx.DiGraph) from transactions.
->Computes node-level features:
    ->In-degree / Out-degree
    ->Transaction amount statistics (sum, mean, std, max, min)
    ->Ratios & log-scaled features

3.Model Training (GAE)
->Encoder: 2-layer GCN with batch norm, dropout, and LeakyReLU.
->Reconstruction: Link prediction loss + L2 regularization.
->Clustering: KMeans on node embeddings to detect groups.

4.Anomaly Scoring
->Distance from cluster centroids → normalized as fraud probability.

5.Visualization
->Plotly interactive scatter for graph visualization.
->Static network layout for overview.
->Node hover tooltips with fraud probability, group, total amount.

6.Saving & Loading
Saves:
    ->Fingerprinted model + predictions
    ->Fixed name files for quick access
    ->Optional custom-named files

->Loads saved model if fingerprint matches and retraining not forced.

4. App Controls
Upload CSV or use default dataset from data/converted01.csv.

Custom save-name for model and predictions.

Configure epochs, learning rate, embedding size, threshold.

Option to force retrain instead of loading saved model.
## 📂 Project Structure


## Structure
- Users = Nodes
- Transactions = Edges
- Trained using GAT (Graph Attention Network)
anti_money_laundering/
│
├── data_fingerprints
├── models_path
├── lib
├── app.py # Main Streamlit dashboard
├── app_gagnn.py #another advanced improved model and main file
├── data_gene_rand.py # random data generator for training if you need u can run.
├── convert.py # converting file of data to convert un ethical data to cleaned data **if only need u can run: no need**
├── graph_only_app.py # Streamlit app for only graph visualization
├── main.py # Model training entry point
│
├── data/
│ └── preprocess.py # Data loading and preprocessing
│
├── models/
│ ├── gat_encoder.py # Graph Attention Network encoder
│ ├── group_classifier.py # Group classification head
│ ├── group_layer.py # Custom GAGNN layer
│ ├── multitask_gnn.py # Main GAGNN architecture
│ ├── node_classifier.py # Node classification head
│
├── train/
│ └── train.py # Model training logic
│
├── utils/
│ ├── graph_utils.py # Graph building and manipulation
│ ├── visualize.py # Graph visualization utilities
│
└── requirements.txt # Python dependencies

---

** Prerequisites**
->Python 3.8+
->GPU (preferably with CUDA) for training
->Pytorch or TensorFlow
->Graph learning libraries: DGL, PyTorch Geometric (PyG), or Spektral

----------------------------------------------------------------------------

| Component           | Tool / Language / Framework                      |
| ------------------- | ------------------------------------------------ |
| Programming         | Python 3.8+                                      |
| Deep Learning       | PyTorch or TensorFlow                            |
| Graph Learning      | PyTorch Geometric or DGL                         |
| Visualization       | matplotlib / seaborn / NetworkX                  |
| Database (optional) | Redis / PostgreSQL (for in-memory graph storage) |
| Optimization        | Adam optimizer                                   |

------------------------------------------------------------------------------
**if any dependencies and libraies missed install then manually using cmmand prompt**

## 🚀 Installation

1. **Clone or extract the project**
```bash
git clone <repo_url>
cd anti_money_laundering

2. **Create a virtual environment**
conda create -n pyg-env python=3.10 -y
conda activate pyg-env # for activating environment and to use and run project on it

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
 
 ** to run in vs code **
 Step 1: Select the Right Python Interpreter in VS Code
Open Command Palette in VS Code:

Press Ctrl + Shift + P

Type: “Python: Select Interpreter” → Press Enter

Choose:Python 3.x.x ('gnn-gpu': conda)
or
Python 3.x.x ('pyg-env': conda)
 with this path : 
 C:\Users\--your sys name--\anaconda3\envs\gnn-gpu\python.exe

  Step 2: (Optional) Confirm settings.json Has the Correct Python Path
In VS Code, open:

.vscode/settings.json (create if it doesn't exist)

Add/verify this line:

{
  "python.pythonPath": "C:\\Users\\Shreyas.H.G\\anaconda3\\envs\\gnn-gpu\\python.exe"
}
**Replace gnn-gpu with pyg-env if you’re using that.**
restart vs code


3.**Install dependencies**

pip install -r requirements.txt

## Requirements
See `requirements.txt`

## How to Run
```bash
activate pyg env environment :
conda activate pyg-env
train the model :


python main.py

Run Streamlit Dashboard:
streamlit run app.py

📌 Notes
Requires GPU for faster training.

Dataset must be in the correct CSV format with columns:
source, target, amount, label

credits:

Developed for Final Year Engineering Project on Anti-Money Laundering detection using Group-Aware Graph Neural Networks by ShreyasHG & Team