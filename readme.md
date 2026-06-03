# AI-Based Active Queue Management (AQM) for 5G Networks 📡

## Project Overview

This project implements and compares different Active Queue Management (AQM) algorithms for congestion control in 5G networks using Python and Machine Learning techniques.

The system simulates packet traffic, queue behaviour, congestion formation, packet dropping, and latency control in a simplified 5G network environment.

The project progresses from traditional queue management algorithms to intelligent AI-based congestion prediction.

---

# Objectives

* Simulate packet flow in a network queue
* Study congestion behaviour under heavy traffic
* Compare FIFO, RED, and CoDel algorithms
* Analyze delay, throughput, and packet loss
* Build an ML model for congestion prediction
* Develop an intelligent ML-based AQM system

---

# Algorithms Implemented

## 1. FIFO (First In First Out)

FIFO is the baseline queue management algorithm.

### Characteristics

* Processes packets in arrival order
* Drops packets only when queue becomes full
* No congestion awareness
* High delay during heavy traffic

### What FIFO Shows

* Queue overflow behaviour
* Tail-drop packet loss
* High latency under congestion
* Worst-case queue performance

---

## 2. RED (Random Early Detection)

RED introduces early congestion control.

### Characteristics

* Uses average queue size
* Performs probabilistic packet dropping
* Prevents sudden queue overflow
* Uses EWMA averaging

### What RED Shows

* Early packet dropping reduces burst loss
* Smoother congestion handling
* Better throughput stability
* Reduced queue overflow compared to FIFO

### Limitation

RED controls queue size rather than actual packet delay.

---

## 3. CoDel (Controlled Delay)

CoDel is a delay-aware queue management algorithm.

### Characteristics

* Focuses on packet waiting time
* Ignores queue size
* Automatically controls latency
* Self-tuning behaviour

### What CoDel Shows

* Better latency control
* Bufferbloat prevention
* Adaptive packet dropping
* Improved QoS for 5G networks

---

# Machine Learning Based AQM

The final stage of the project integrates Machine Learning for predictive congestion control.

The ML model predicts network congestion using:

* Queue Length
* Arrival Rate
* Service Rate
* Delay
* Throughput
* Packet Drop Rate

The system then dynamically controls packet dropping before severe congestion occurs.

---

# Technologies Used

| Component            | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| Data Handling        | Pandas                        |
| Visualization        | Matplotlib                    |
| Machine Learning     | Scikit-learn                  |
| Simulation           | Python-based Queue Simulation |

---

# Folder Structure

```text id="n4a7tw"
AQM_Project/
│
├── packet.py
├── metrics.py
├── fifo.py
├── red.py
├── coDel.py
├── dataset_generator.py
├── processed_dataset.py
├── train_model.py
├── predictor.py
├── main.py
│
├── data/
│   ├── traffic_data.csv
│   └── processed_data.csv
│
├── models/
│   ├── congestion_model.pkl
│   └── scaler.pkl
│
└── graphs/
```

---

# Practical Workflow

```text id="82mjlwm"
Packet Generation
        ↓
FIFO Queue Simulation
        ↓
RED Congestion Control
        ↓
CoDel Delay Control
        ↓
Dataset Generation
        ↓
Machine Learning Training
        ↓
Congestion Prediction
        ↓
Smart Packet Management
```

---

# Metrics Evaluated

* Queue Length
* Packet Delay
* Throughput
* Packet Loss
* Congestion Probability

---

# Comparison Summary

| Algorithm | Queue Awareness | Delay Awareness | Congestion Control |
| --------- | --------------- | --------------- | ------------------ |
| FIFO      | Basic           | No              | Reactive           |
| RED       | Yes             | Partial         | Preventive         |
| CoDel     | No              | Yes             | Delay-Aware        |
| ML-AQM    | Intelligent     | Intelligent     | Predictive         |

---

# Key Learning Outcomes

* Understanding queue management techniques
* Congestion handling in 5G networks
* Delay optimization
* Packet scheduling
* Machine Learning integration in networking
* QoS optimization techniques

---

# Future Scope

* Reinforcement Learning based AQM
* NS-3 Integration
* Real-time traffic visualization
* Network slicing optimization
* 6G congestion prediction systems

---

# Conclusion

This project demonstrates the evolution of congestion control techniques from basic FIFO buffering to intelligent AI-driven Active Queue Management systems. By integrating Machine Learning with traditional AQM algorithms, the system improves congestion prediction, reduces packet delay, and enhances Quality of Service in modern 5G networks.
