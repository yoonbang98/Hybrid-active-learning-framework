# Hybrid active learning strategy for optimizing experimental conditions in cell-free enzyme cascades 
Here, we propose a **Hybrid Active Learning (HAL)** framework to minimize experimental trials in cell-free enzyme cascades. Our method maximizes the efficiency of AI model learning by utilizing a hybrid query strategy that combines distance-based and model prediction-based information to select experimental conditions most useful for learning.

## 1. Dataset

1) Butanol
2) Lycopene
3) Limonene

## 2. Source code
1) Code for hyperparameter tuning and HAL function
   - Initialize_and_HAL_function.py
2) Code for training each dataset
   - HAL_Butanol.py
   - HAL_Lycopene.py
   - HAL_Limonene.py

Instruction to use the source code:
1) Download the code.
2) Make sure you are in the main directory where the code is.

## 3. Library
1) python==3.11.5
2) numpy==1.26.3
3) pandas==1.5.3
4) xgboost==2.0.3
5) sklearn==1.4.0
6) tqdm==4.66.1
