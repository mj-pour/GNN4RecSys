# GNN4RecSys

A PyTorch framework for graph-based recommender systems, extending the original LightGCN architecture with attention mechanisms, hard negative sampling, and modular components for exploring improved GNN-powered recommendation models.

---

## Features

- Enhanced LightGCN architecture with:
  - Attention-based message aggregation
  - Hard negative sampling for stronger contrastive learning
  - Configurable depth, embedding size, and propagation strategies
- Modular design for experimenting with new GNN architectures
- Support for benchmark datasets: Movielens, Gowalla, Yelp2018, Amazon-Book, LastFM
- Clean training pipeline with logging, evaluation, and reproducibility

---

## Installation

```bash
pip install -r requirements.txt

## Enviroment Requirement

`pip install -r requirements.txt`

## An example to run a 3-layer attlgn

run **Attentive LightGCN** on **Movielens** dataset:

* command

` cd code && python main.py --model="attlgn" --dataset="movielens-100k" --layer=3 --recdim=64 --lr=0.001 --decay=1e-4 --topks="[10]" --testbatch=10 --bpr_batch=512 --epochs=500`

