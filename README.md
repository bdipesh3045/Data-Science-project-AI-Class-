# Hand Sign Recognition

This project predicts an American Sign Language letter from a static hand image. We will compare classical machine learning models with a neural network and deploy the best-performing model as a usable app. All models will be trained from scratch without pre-trained models.

**Dataset**

We will use Sign Language MNIST, which contains 28×28 grayscale images of 24 letters. J and Z are excluded because their signs require movement.

**Approach**

For classical machine learning, we will use **SVM and XGBoost** because they learn in different ways. SVM learns boundaries that separate letter classes, while XGBoost builds decision trees sequentially to reduce prediction errors.

To explore neural networks and compare their efficiency, we will use **TensorFlow** to build a Multi-Layer Perceptron. We will experiment with different hidden layers, layer sizes, and activation functions.

**Evaluation**

We will compare accuracy, F1 scores, confusion matrices, training time, and prediction speed using consistent data splits and preprocessing. **Weights & Biases** will track experiments and neural network loss curves alongside the classical model benchmark results.

**Demo and Deliverables**

We will use **Flask** to build a simple web app that accepts a PNG upload and displays the predicted letter and confidence score. Flask will also provide a prediction API for the selected model.

The final deliverables will include a clear Jupyter notebook, reusable preprocessing and prediction code, an experiment dashboard, and a working demo with a presentation. The goal is a clean, versioned, reproducible pipeline alongside strong model performance.

**Resources**

- [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [SVM documentation](https://scikit-learn.org/stable/modules/svm.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [TensorFlow documentation](https://www.tensorflow.org/guide)
- [Weights & Biases documentation](https://docs.wandb.ai/)
- [Flask documentation](https://flask.palletsprojects.com/)


**Contributors**
**-Dipesh Sharma**
**-Palden Zimba Tamang**
**-Anjit Kafle**