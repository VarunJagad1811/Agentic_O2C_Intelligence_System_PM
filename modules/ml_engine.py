import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@st.cache_resource
def load_engine():
    try:
        df = pd.read_csv("data/O2C_Dataset_10000_Cases_Enriched_50Features.csv")
    except FileNotFoundError:
        st.error("Error: CSV file not found. Ensure it is in the 'data' folder.")
        return None, None, None, None, None, None

    df['order_value'] = pd.to_numeric(df['order_value'], errors='coerce').fillna(0)
    df['is_international'] = pd.to_numeric(df['is_international'], errors='coerce').fillna(0)
    df['package_weight_kg'] = pd.to_numeric(df['package_weight_kg'], errors='coerce').fillna(0)
    df['vendor_reliability_score'] = pd.to_numeric(df['vendor_reliability_score'], errors='coerce').fillna(90)

    # Normalize core metrics
    np.random.seed(42)
    norm_value = df['order_value'].clip(upper=10000) / 10000.0  
    norm_weight = df['package_weight_kg'].clip(upper=100) / 100.0 

    # --- 1. ACADEMIC CAUSAL LOGIC INJECTION (UPDATED) ---
    def shipping_risk(mode):
        m = str(mode).lower()
        if 'air' in m: return 0.25       # Aviation security, strict weight/size limits
        if 'sea' in m: return 0.15       # Customs, port authority border checks
        if 'ground' in m: return 0.05    # Standard domestic transport
        return 0.0
        
    def staff_risk(level):
        l = str(level).lower()
        if 'low' in l or '0' in l: return 0.15     # Inexperienced staff = Error Risk
        if 'high' in l or '2' in l: return -0.15   # Senior staff = Risk Mitigator
        return 0.0                                 # Medium staff = Baseline
        
    ship_mod = df['shipping_mode'].apply(shipping_risk)
    staff_mod = df['staff_training_level'].apply(staff_risk)
    vendor_mod = (90 - df['vendor_reliability_score']) * 0.005 
    
    rule_score = (norm_value * 0.35) + \
                 (norm_weight * 0.15) + \
                 (df['is_international'] * 0.20) + \
                 (df['is_large_electronic'] * 0.15) + \
                 ship_mod + staff_mod + vendor_mod
    
    final_score = rule_score + np.random.normal(0, 0.02, len(df))
    threshold = final_score.quantile(0.70) 
    
    # FIX: BULLETPROOF ACADEMIC PROBABILITY ASSIGNMENT
    # If the score is above threshold, there is a 90% chance it gets reviewed (10% human error/exception).
    # If the score is below threshold, there is only a 5% chance of a random compliance audit.
    prob = np.where(final_score > threshold, 0.90, 0.05)
    
    # Use binomial distribution to assign final 0 or 1 label stochastically
    df['has_manual_review'] = np.random.binomial(1, prob)

    def get_base_time(mode):
        m = str(mode).lower()
        if 'air' in m: return np.random.normal(2, 0.5)  
        elif 'sea' in m: return np.random.normal(15, 3) 
        return np.random.normal(5, 1) 
        
    df['base_days'] = df['shipping_mode'].apply(get_base_time)
    risk_delay = df['has_manual_review'].apply(lambda x: np.random.normal(5, 1.5) if x==1 else 0)
    df['processing_days'] = (df['base_days'] + risk_delay).round(1).clip(lower=1.0)
    
    df['Variant_Name'] = df['shipping_mode'] + " (" + df['is_international'].apply(lambda x: 'Intl' if x==1 else 'Dom') + ")"
    df['Process_Path_Group'] = df.apply(lambda row: f"{row['Variant_Name']} ➡ {'⛔ Review' if row['has_manual_review'] == 1 else '✅ Low Risk'}", axis=1)

    features = ['order_value', 'package_weight_kg', 'is_international', 'is_large_electronic', 'staff_training_level', 'shipping_mode', 'product_type', 'vendor_reliability_score']
    X = df[features].copy()
    y = df['has_manual_review']
    
    encoders = {}
    for col in ['staff_training_level', 'shipping_mode', 'product_type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # FIX: IMPLEMENT TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    # Train ONLY on the training data
    model.fit(X_train, y_train)
    
    # FIX: CALCULATE TRUE METRICS
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Attach metrics to the model object so they can be accessed dynamically by app.py
    model.test_metrics = {
        "accuracy": test_acc,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }
    
    explainer = shap.TreeExplainer(model)
    
    # Returning test_acc instead of the previous fake training accuracy
    return df, X, model, explainer, encoders, test_acc

def get_flat_shap(explainer, input_row, expected_len):
    """Safely flattens SHAP arrays regardless of model output format."""
    try:
        shap_values = explainer.shap_values(input_row)
        if isinstance(shap_values, list): vals = shap_values[1][0]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3: vals = shap_values[0, :, 1]
        else: vals = shap_values[0]
        vals = np.array(vals).flatten()
        
    # FIX: FIX THE SHAP EXCEPTION (No silent swallowing)
    except Exception as e:
        print(f"SHAP Error during calculation: {e}")
        vals = np.zeros(expected_len)
        
    if len(vals) > expected_len: vals = vals[:expected_len]
    elif len(vals) < expected_len: vals = np.concatenate((vals, np.zeros(expected_len - len(vals))))
    
    return vals