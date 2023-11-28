import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from DetectDelimiter import detect_delimiter

def Classify(train_data_path, train_label_path, test_data_path, output_file_path):
    train_delimiter = detect_delimiter(train_data_path)
    test_delimiter = detect_delimiter(test_data_path)

    train_data = pd.read_csv(train_data_path, delimiter=train_delimiter, header=None, engine='python')
    train_labels = pd.read_csv(train_label_path, header=None, engine='python')
    test_data = pd.read_csv(test_data_path, delimiter=test_delimiter, header=None, engine='python')

    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    X = train_data
    Y = train_labels.squeeze()

    # Check the number of features in the training data
    num_features = X.shape[1]

    if num_features >= 5000:
        # Use Logistic Regression for trainData2
        print("Performing Logistic Regression...")
        pipeline = performLogisticRegression(X, Y)
    elif num_features < 5000 and num_features >= 3000:
        # Use PCA followed by SVM for trainData1
        print("Performing PCA followed by SVM...")
        pipeline = performPCASVC(X, Y)
    else:
        # Use Random Forest for trainData 3,4,5
        print("Performing randomforest...")
        pipeline = performRandomForest(X, Y)

    test_predictions = pipeline.predict(test_data)

    with open(output_file_path, 'w') as f:
        for label in test_predictions:
            f.write(str(label) + '\n')

    scores = cross_val_score(pipeline, X, Y, cv=5)
    print("Average cross-validation score: ", scores.mean())

# All the algorithms

def performLogisticRegression(X, Y):
    # Create a pipeline that first scales the data and then applies logistic regression
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)  # Increase max_iter from the default
    )
    pipeline.fit(X, Y)
    return pipeline

def performDecisionTree(X, Y):
    # Create a Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Create a pipeline with the classifier
    pipeline = Pipeline([
        ('classifier', clf)
    ])

    # Fit the classifier on the data
    pipeline.fit(X, Y)

    return pipeline

def knn_classify(X, Y, n_neighbors_list=[5, 7, 11, 15, 21, 31, 41, 51, 61, 71, 79, 81, 91]):
    X_clean, Y_clean = remove_outliers(X, Y)

    best_score = 0
    best_k = n_neighbors_list[0]
    best_classifier = None

    for k in n_neighbors_list:
        classifier = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(classifier, X_clean, Y_clean, cv=5)
        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_k = k
            best_classifier = classifier

    print(f"Best K: {best_k}")
    best_classifier.fit(X, Y)  # Fit the classifier with the best K
    return best_classifier

def remove_outliers(X, Y):
    df = pd.concat([X, Y], axis=1)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    X_out = df_out.iloc[:, :-1]
    Y_out = df_out.iloc[:, -1]
    return X_out, Y_out


def performNaiveBayes(X, Y):
    # Create a Gaussian Naive Bayes classifier
    gnb = GaussianNB()

    # Create a pipeline with the Naive Bayes classifier
    pipeline = Pipeline([
        ('gnb', gnb)
    ])

    # Train the classifier on the entire dataset
    pipeline.fit(X, Y)

    # Return the pipeline with the trained Naive Bayes classifier
    return pipeline

def performPCASVC(X, Y):
  pca = PCA(n_components=0.95)
  classifier = SVC(kernel='rbf')

  pipeline = Pipeline([
      ('pca', pca),
      ('svc', classifier)
  ])

  pipeline.fit(X, Y)
  return pipeline

def performRandomForest(X, Y):
    classifier = RandomForestClassifier()
    classifier.fit(X, Y)
    return classifier

def performSVM(X, Y):
    # Splitting the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize SVM classifier
    model = SVC(kernel='rbf')

    # Train the model
    model.fit(X_train_scaled, Y_train)

    return model