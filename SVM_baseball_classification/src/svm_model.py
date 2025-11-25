import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
import numpy as np

# 1. 데이터 로드
def data_load():

    # train, test 데이터 로드
    train = pd.read_csv("Data/processed/train_20251124_235812.csv")
    test = pd.read_csv("Data/processed/test_20251124_235812.csv")

    # 투표율 계산
    train["vote_rate"] = (train["votes"] / train["ballots"] * 100).round(2)
    test["vote_rate"] = (test["votes"] / test["ballots"] * 100).round(2)

    # 간편 사용을 위한 변수 생성
    x = 'votes'
    y = "vote_rate"

    X_train = train[[x, y]]
    y_train = train['inducted']
    X_test = test[[x, y]]
    y_test = test['inducted']

    return X_train, y_train, X_test, y_test, train


# 2. SVM 하이퍼파라미터 탐색 함수
def svc_param_selection(X_train, y_train):
    svm_parameters = [
                        {'kernel': ['rbf'],
                         'gamma': [0.00001,0.0001, 0.001, 0.01, 0.1, 1],
                         'C': [0.01, 0.1, 1, 10, 100, 1000]
                        }
                       ]
    
    clf = GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(X_train, y_train.values.ravel())
    print(clf.best_params_)
    
    return clf


def visuall(clf, train):

    # Data 준비
    train["vote_rate"] = (train["votes"] / train["ballots"] * 100).round(2)
    x, y = 'votes', 'vote_rate'

    X = train[[x, y]]
    Y = train['inducted'].apply(lambda v: 0 if v=='N' else 1)

    # Best hyperparameters 사용
    best_C = clf.best_params_['C']
    best_gamma = clf.best_params_['gamma']
    
    best_model = SVC(C=best_C, gamma=best_gamma)
    best_model.fit(X, Y)

    # Meshgrid (데이터 범위 기반)
    xx_min, xx_max = X[x].min(), X[x].max()
    yy_min, yy_max = X[y].min(), X[y].max()

    xx, yy = np.meshgrid(
        np.linspace(xx_min, xx_max, 200),
        np.linspace(yy_min, yy_max, 200)
    )

    # Decision boundary 계산
    Z = best_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 🔥 단일 그래프만 출력
    fig = plt.figure(figsize=(6, 6))
    plt.title(f"Best Model (C={best_C}, gamma={best_gamma})", size=14)

    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu, shading='auto')
    plt.scatter(X[x], X[y], c=Y, cmap=plt.cm.RdBu_r, edgecolors='k')
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()

    return fig


def model_test(X_test, y_test, clf):
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("accuracy : "+ str(accuracy_score(y_true, y_pred)) )
    comparison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_true.values.ravel()}) 
    print(comparison)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title("Confusion Matrix")
    return fig


