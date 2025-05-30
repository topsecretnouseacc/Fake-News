import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Veri yükleme
    df = pd.read_csv('data/processed/combined.csv')
    X = df['content'].tolist()
    y = df['label'].tolist()

    # Train/test ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Embedding modeli (MiniLM)
    print("Embedding hesaplanıyor (train)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    emb_train = embedder.encode(X_train, show_progress_bar=True, batch_size=64)

    # Lojistik Regresyon eğit
    print("Lojistik Regresyon eğitiliyor...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(emb_train, y_train)

    # Test verisi üzerinde embedding + tahmin
    print("Embedding hesaplanıyor (test)...")
    emb_test = embedder.encode(X_test, show_progress_bar=True, batch_size=64)
    y_pred = clf.predict(emb_test)
    y_prob = clf.predict_proba(emb_test)[:,1]

    # Performans metrikleri
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc = roc_auc_score(y_test, y_prob)
    print("\nPerformans Raporu:")
    print(f"  • Accuracy : {acc:.4f}")
    print(f"  • Precision: {prec:.4f}")
    print(f"  • Recall   : {rec:.4f}")
    print(f"  • F1-score : {f1:.4f}")
    print(f"  • ROC-AUC  : {roc:.4f}\n")
    print("Detaylı sınıflandırma raporu:")
    print(classification_report(y_test, y_pred, target_names=['Real','Fake']))

    # Confusion matrix görseli
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/confusion_matrix.png', bbox_inches='tight')
    print("Confusion matrix kaydedildi → reports/figures/confusion_matrix.png")

    # Modelleri kaydet
    os.makedirs('models', exist_ok=True)
    joblib.dump(embedder, 'models/embedder.joblib')
    joblib.dump(clf,      'models/classifier.joblib')
    print("Modeller kaydedildi → models/ klasörü")

if __name__ == '__main__':
    main()
