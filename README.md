
# EXPERIMENT REPORT


## 📅 Date: 2025-11-25

---

## 🧠 Model: SVM (RBF)

---

## 📌 Dataset Info:
- samples: 786
- class balance: N: 762 (96.9%), Y: 24 (3.1%) → ⚠ Imbalanced dataset!
  
---

## 📦 Hyperparameters
- C: 1000
- gamma: 1e-05
- kernel: RBF
  
---

## 📊 Performance (Summary)
Accuracy: 0.997

---

## 📄 Classification Report (Raw Text)
```bash
              precision    recall  f1-score   support

           N       1.00      1.00      1.00       762
           Y       0.92      1.00      0.96        24

    accuracy                           1.00       786
   macro avg       0.96      1.00      0.98       786
weighted avg       1.00      1.00      1.00       786
```
---

## 📈 Saved Visuals:
- confusion_matrix.png
- data.png
- best_model.png
  
---

## 🚀 실행 방법 
```bash
git clone https://github.com/Rohstar0613/SVM-imbalanced-data-learning
cd SVM_baseball_classification
pip install -r requirements.txt
python main.py
```
---

## 🧠 More Details & Reflection
자세한 실험 과정과 회고록은 아래 링크에서 확인할 수 있습니다.  
👉 https://rohstar.tistory.com/entry/1

---
