# Delhivery Network Intelligence Dashboard

## Project Overview

This project analyzes Delhivery's logistics network using graph analytics, machine learning, Node2Vec embeddings, and GraphSAGE embeddings to identify bottlenecks, predict delivery delays, and support operational decision-making.

The objective is to improve ETA prediction accuracy while identifying high-risk hubs and delay-prone corridors within the transportation network.

## Problem Statement

Logistics networks contain thousands of interconnected hubs and transportation corridors. Traditional ETA estimates based solely on route distance and travel time often fail to capture operational realities such as:

- Hub congestion
- Network bottlenecks
- Corridor-level delays
- Route-type differences (FTL vs Carting)

This project develops a graph-based intelligence system capable of:

1. Detecting bottleneck hubs
2. Identifying chronic delay corridors
3. Predicting shipment ETAs
4. Comparing route strategies
5. Quantifying operational impact

## Project Pipeline

### Part 1: Data Understanding & EDA
- Shipment exploration
- Delay ratio analysis
- Route type profiling

### Part 2: Network Construction
- Directed logistics graph creation
- Edge-level metrics

### Part 3: Hub Analytics
- Centrality measures
- Hub performance indicators

### Part 4: Bottleneck Detection
- Bottleneck Risk Score (BRS)
- Critical hub identification

### Part 5: Delay Corridor Analysis
- Chronic corridor detection
- Severity ranking

### Part 6: ETA Prediction
Models evaluated:
- OSRM Baseline
- Random Forest
- XGBoost
- LightGBM

### Part 7: Graph Embeddings
- Node2Vec
- Feature importance analysis

### Part 8: GraphSAGE Enhancement
- Learned node embeddings
- ETA prediction improvement

### Part 9: FTL vs Carting Strategy
- Route-level operational recommendations



## Key Results

### ETA Prediction

| Model | MAE |
|---------|---------|
| OSRM | 206.56 sec |
| Random Forest | 37.29 sec |
| Node2Vec + RF | 36.56 sec |
| GraphSAGE + GBM | 39.32 sec |

### Operational KPIs

- Shipments analyzed: 143,418+
- Network hubs: 1,601+
- Chronic delay corridors: 20
- Revenue at risk identified: ₹2.45 Million
- Best operational accuracy: GraphSAGE + GBM
- Within ±15% prediction accuracy: 59.6%


## Business Value

The solution enables operations teams to:

- Prioritize infrastructure upgrades
- Identify high-risk hubs
- Reduce SLA breaches
- Improve ETA reliability
- Optimize route selection
- Improve customer experience


## Dashboard Features

- Executive Summary
- Network Overview
- Bottleneck Hub Analysis
- Delay Corridor Analysis
- ETA Prediction Results
- Node2Vec vs Baseline Comparison
- GraphSAGE Comparison
- FTL vs Carting Framework
- Downloadable Reports
- Final results


## Dashboard Link
https://delhiverynetworkintelligence.streamlit.app/
