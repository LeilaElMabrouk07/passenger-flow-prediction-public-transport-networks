Passenger Flow Forecasting with STGCN

- Project Overview

This part focuses on forecasting passenger flow in public transportation networks using Spatio-Temporal Graph Convolutional Networks (STGCN).

The objective is to anticipate congestion peaks before they occur, enabling transport operators to adapt service supply proactively and maintain quality of service.

The model is trained on Transport for London (TfL) NUMBAT data (2016–2024) and operates at the level of inter-station links.

- Methodology

We model the transport network as a graph structure:

Nodes: inter-station links (e.g., Station A → Station B)
Edges: connectivity between consecutive links (line graph representation)
Features: passenger flow over time

The model combines:

Temporal Convolutions → capture time dynamics
Graph Convolutions → capture spatial dependencies

- Dataset
Source: Transport for London
Access: https://tfl.gov.uk/info-for/open-data-users/our-open-data
Data used:
NUMBAT dataset (2016–2024)
Focus: Link Loads tables
Time granularity: 15-minute intervals
Key variables:
From Station, To Station
Line, Direction
Time intervals
Passenger flow

- Installation

Python 3.11
Recommended: virtual environment

python3.11 -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

- Dependencies

pip install -r requirements.txt

- Project Structure

STGCN_MODEL/
│
├── src/
│   ├── build_table.py          # Data preprocessing (Excel → Parquet)
│   ├── graph_link.py           # Link-based graph construction
│   ├── graph_station.py        # Station graph utilities
│   ├── model_stgcn.py          # STGCN architecture
│   ├── link_dataset_stgcn.py   # Dataset & normalization
│   ├── train_link_stgcn.py     # Training pipeline
│   ├── app_link.py             # Streamlit visualization
│   ├── config.py
│   ├── frame_importance_stgcn.py
│   ├── plot_convergence.py
│   ├── preprocess.py
│
├── requirements.txt
└── README.md


- Data Processing Pipeline

Raw Excel files → Long format
Time normalization (continuous timeline across midnight)
Construction of:
Link IDs
Graph adjacency matrix
Conversion into tensors:
Shape: [T, E] (time × links)
Sliding window generation:
Input: N_HISTORY
Output: HORIZON

- Model Architecture

The STGCN model consists of:

Temporal Convolution Layer
Graph Convolution Layer
Temporal Convolution Layer
Output layer predicting multiple future steps
Input
[B, 1, N, T]
Output
[B, H, N]

Where:

B: batch size
N: number of links
T: history window
H: prediction horizon

- Training

Run:

python src/train_link_stgcn.py
Training setup:
Training years: 2016–2022
Validation year: 2023
Test year: 2024
Loss: MSE
Metrics:
MAE
RMSE

Model and normalization are saved in cache/

- Evaluation

Evaluation is performed on: Global metrics
Per-horizon metrics (e.g., +15min, +30min, ...)

- Visualization

Run the Streamlit app:

streamlit run src/app_link.py

Features:

Select:
Year
Line
Direction
Time
Compare:
Real vs predicted flows
Visualize:
Passenger flow per link
Forecast errors



