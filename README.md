# Neural Network-Based DPD for 3-Way PA with Memory Effects

This project explores the design and optimization of Digital Pre-Distortion (DPD) models for a 3-way Power Amplifier (PA) with memory effects using various neural network architectures and optimization algorithms.

## 🧠 Models Evaluated
- **Fully Connected Network (FCN)**
- **LSTM**, **Bi-LSTM**, **GRU**
- **CNN-LSTM**, **Multi-layer FCN**
- **Temporal Convolutional Network (TCN)**
- **Transformer**
- **Time Delay Network**
- **Volterra Series**

## ⚙️ Optimization Algorithms
- Adam, SGDM, RMSProp
- Levenberg-Marquardt (trainlm)
- Scaled Conjugate Gradient (trainscg)
- Bayesian Regularization (trainbr)

## 📊 Key Findings
- **Multi-layer FCN** with Bayesian regularization achieved the best performance: **-49.75 dB NMSE** on the validation set.
- Complex models (e.g., Bi-LSTM, Transformer) suffered from **overfitting** despite high fitting accuracy.
- **Volterra models** underperformed compared to neural network approaches.

## 🛠️ Tools
- MATLAB (Deep Learning Toolbox, Neural Network Toolbox)
- SystemVue for simulation and validation

## 👥 Authors
- Yuan Zhong
- Yansen Jia

This project demonstrates the effectiveness of neural networks in PA linearization and highlights the importance of model simplicity and appropriate regularization for generalization performance.

---
**Tag**: `DPD` `Power Amplifier` `Neural Networks` `MATLAB` `Digital Pre-Distortion` `5G` `Wireless Communications`
