import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_processor import DataProcessor

def train():
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv("raw_data.csv")
    except FileNotFoundError:
        print("Error: raw_data.csv not found. Run data_generator.py first.")
        return

    # Process data
    print("Processing data...")
    processor = DataProcessor()
    
    # We need to split types correctly. 
    # DataProcessor handles X cleaning. y is separate.
    X = df.drop(columns=['machine_failure'])
    y = df['machine_failure']

    # Fit processor on X
    # Note: In a real pipeline we fit on train, transform on test.
    # Here we'll split first.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit on train, transform both
    processor.fit(X_train)
    X_train_processed = processor.transform(X_train)
    X_test_processed = processor.transform(X_test)
    
    # Train Model
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_processed, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test_processed)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    
    # Save Metrics
    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Failure'], 
                yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save Model and Processor
    joblib.dump(clf, 'model.joblib')
    joblib.dump(processor, 'processor.joblib')
    
    print("Training complete. Artifacts saved: model.joblib, processor.joblib, metrics.json, confusion_matrix.png")

if __name__ == "__main__":
    train()
