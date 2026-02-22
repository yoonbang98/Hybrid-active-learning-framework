# Hybrid active learning strategy for optimizing experimental conditions in cell-free enzyme cascades 
Here, we propose a **Hybrid Active Learning (HAL)** framework for optimizing experimental conditions in cell-free enzyme cascades. Our method maximizes the efficiency of AI model learning by utilizing a hybrid query strategy that combines distance-based and model prediction-based information to select experimental conditions most useful for learning.

## 1. Dataset

| Dataset | Source / Reference |
| :--- | :--- |
| **Butanol** | A. S. Karim *et al.* (2020). In vitro prototyping and rapid optimization of biosynthetic enzymes for cell design. *Nat Chem Biol.* 16: 912-+. |
| **Lycopene** | **[Original Data]** This is unique data generated through experiments designed and conducted by the project team itself. |
| **Limonene** | Q. Dudley *et al.* (2020). In vitro prototyping of limonene biosynthesis using cell-free protein synthesis. *Metabolic Engineering.* 61: 251–260. |

## 2. Source Code

The source code is categorized into three main parts based on its function:

* **Core HAL Framework**
  * `Initialize_and_HAL_function.py` : Code for hyperparameter tuning and the main HAL function.
* **Google colab implementation of HAL Framework for lycopene**
  * `HAL_example_Lycopene.ipynb` : Active learning simulation for lycopene production using Hybrid Active Learning.
* **Dataset Training**
  * `HAL_Butanol.py` : Training script for the Butanol dataset.
  * `HAL_Lycopene.py` : Training script for the Lycopene dataset.
  * `HAL_Limonene.py` : Training script for the Limonene dataset.
* **Lycopene Experiment Application**
  * `Recommendation_function.py` : Contains functions for preprocessing, hyperparameter tuning, METIS, and HAL.
  * `Lycopene_recommendation.py` : Example script for Day 1 recommendations using METIS & HAL.

## 3. Getting Started

**Instructions to use the source code:**
1. Download or clone this repository.
2. Open your terminal and ensure you are in the main directory containing the downloaded code.
3. Run the desired scripts from this root directory.

## 4. Requirements

The project requires **Python 3.11.5**. The following libraries are required to run the code:

```text
numpy==1.26.3
pandas==1.5.3
xgboost==2.0.3
scikit-learn==1.4.0
tqdm==4.66.1
