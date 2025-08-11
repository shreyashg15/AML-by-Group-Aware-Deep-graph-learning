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

## 📂 Project Structure


## Structure
- Users = Nodes
- Transactions = Edges
- Trained using GAT (Graph Attention Network)
anti_money_laundering/
│
├── app.py # Main Streamlit dashboard
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