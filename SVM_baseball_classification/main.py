from src.svm_model import *
from src.visualize import *

def main():
    X_train, y_train, X_test, y_test, train = data_load()
    clf = svc_param_selection(X_train, y_train)
    png = visuall(clf, train)
    save_png(png, prefix="best_model")
    fig = model_test(X_test, y_test, clf)
    save_png(fig, prefix="confusion")

if __name__ =="__main__":
    main()