## Facebook Friends Recommendation System with Graph Attention Network

Friend recommendation systems are important in enhancing user engagement on social media platforms by suggesting potential new connections. Our project aims to develop a friend recommendation system by leveraging Graph Attention Network (GAT) to predict potential new connections within social networks. Traditional Graph Convolutional Networks (GCNs) treat all neighboring nodes equally during the aggregation process, which may not capture the varying significance of different relationships. In contrast, GAT introduce an attention mechanism that assigns learnable weights to neighboring nodes, allowing the model to focus more on influential relationships. This weighted consideration aligns well with social dynamics, where some friendships have a more substantial impact on user behavior than others. By employing GAT, we intend to capture these nuanced interactions, thereby enhancing the accuracy and relevance of friend suggestions on platforms like Facebook. This approach not only improves recommendation quality but also contributes to a deeper understanding of social network structures.

## Project Structure

```plaintext
project-root/
├── data/                           
├── notebooks/
│   └── data_exploration.ipynb      # Jupyter notebook for data exploration
├── environment.yml                 # Conda environment configuration file
├── experiment/                     # Experiment results
├── src/                            # Source code for training
└── README.md                       # Project documentation
```

## Usage
1. Clone the repo.
2. Install dependencies:
3. Run the training and evaluation script:
    python src/main.py

## Experiments
- **GCN Experiment**: Results can be found in `experiments/gcn_experiment/results.json`.

## Method
TBU
