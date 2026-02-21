# Hybrid active learning strategy for optimizing experimental conditions in cell-free enzyme cascades 
Here, we propose a **Hybrid Active Learning (HAL)** framework for optimizing experimental conditions in cell-free enzyme cascades. Our method maximizes the efficiency of AI model learning by utilizing a hybrid query strategy that combines distance-based and model prediction-based information to select experimental conditions most useful for learning.

## 1. Dataset

1) Butanol - A. S. Karim, Q. M. Dudley, A. Juminaga, Y. B. Yuan, S. A. Crowe, J. T. Heggestad, S. Garg, T. Abdalla, W. S. Grubbe, B. J. Rasor, D. N. Coar, M. Torculas, M. Krein, F. Liew, A. Quattlebaum, R. O. Jensen, J. A. Stuart, S. D. Simpson, M. Köpke and M. C. Jewett (2020) In vitro prototyping and rapid optimization of biosynthetic enzymes for cell design. Nat Chem Biol. 16: 912-+.
2) Lycopene - This is unique data generated through experiments designed and conducted by the project team itself.
3) Limonene - Q. Dudley, A. Karim, C. Nash, M. Jewett (2020) In vitro prototyping of limonene biosynthesis using cell-free protein synthesis. Metabolic Engineering. 61: 251–260.

## 2. Source code
1) Code for hyperparameter tuning and HAL function
   - Initialize_and_HAL_function.py
2) Code for training each dataset
   - HAL_Butanol.py
   - HAL_Lycopene.py
   - HAL_Limonene.py
3) Code for lycopene experiment
   - Recommendation_function.py (Preprocessing, Hyperparameter tuning, METIS, HAL function)
   - Lycopene_recommendation.py (METIS & HAL, Day 1 recommendation example)

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
