# Facial Recognition for Arnold Schwarzenegger

![Facial Recognition](facialrecognition.jpg)

## Project Overview
You are a member of an elite group of data scientists, specialising in advanced facial recognition technology, this firm is dedicated to identifying and safeguarding prominent individuals from various spheres—ranging from entertainment and sports to politics and philanthropy. The team's mission is to deploy AI-driven solutions that can accurately distinguish between images of notable personalities and the general populace, enhancing the personal security of such high-profile individuals. You're to focus on Arnold Schwarzenegger, a figure whose accomplishments span from bodybuilding champion to Hollywood icon, and from philanthropist to the Governor of California.

## The Data
The `data/lfw_arnie_nonarnie.csv` dataset contains processed facial image data derived from the "Labeled Faces in the Wild" (LFW) dataset, focusing specifically on images of Arnold Schwarzenegger and other individuals not identified as him. This dataset has been prepared to aid in the development and evaluation of facial recognition models. There are 40 images of Arnold Schwarzenegger and 150 of other people.

| Column Name | Description |
|-------------|-------------|
| PC1, PC2, ... PCN | Principal components from PCA, capturing key image features. |
| Label | Binary indicator: `1` for Arnold Schwarzenegger, `0` for others. |

## Methodology
The project follows these steps:
1.  **Data Loading**: The preprocessed data is loaded from `data/lfw_arnie_nonarnie.csv`.
2.  **Data Splitting**: The dataset is split into an 80% training set and a 20% test set.
3.  **Model Pipelines**: Three machine learning models are defined, each within a scikit-learn `Pipeline` that includes a `StandardScaler` for feature normalization:
    *   K-Nearest Neighbors (KNN)
    *   Logistic Regression
    *   Random Forest
4.  **Model Training and Selection**: The models are trained and evaluated using 5-fold cross-validation on the training data. The model with the highest average cross-validation score is selected as the best model.
5.  **Final Evaluation**: The best model is trained on the entire training set and then evaluated on the test set. Performance is measured using accuracy, precision, recall, and the F1-score.
6.  **Visualization**: A confusion matrix is created to visualize the model's performance on the test set.

## Results
*   **Best Model**: Logistic Regression was identified as the best-performing model with a cross-validation score of **0.8222**.
*   **Test Set Performance**:
    *   Accuracy: **0.8158**
    *   Precision: **1.0000**
    *   Recall: **0.1250**
    *   F1 Score: **0.2222**

The confusion matrix for the test set is:
```
[[30  0]
 [ 7  1]]
```
This shows that the model correctly classified all 30 "Non-Arnie" images and one "Arnie" image, but it failed to identify 7 "Arnie" images.

## Conclusion
The model demonstrated a strong ability to correctly identify non-Arnold images, achieving perfect precision. This means that when the model predicts an image is of Arnold Schwarzenegger, it is highly likely to be correct. However, the model struggles to identify all of Arnold's images, as indicated by the low recall score. This suggests that while the model is cautious and accurate in its positive predictions, it misses a significant number of Arnold's images. The focus of this project was on identifying Arnold Schwarzenegger, and while the precision is high, the low recall indicates room for improvement in the model's sensitivity.

## How to Run
To run the analysis yourself, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    Ensure you have Python installed. You can install the necessary libraries using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Run the notebook:**
    Start Jupyter Notebook and open `notebook.ipynb`:
    ```bash
    jupyter notebook
    ```
