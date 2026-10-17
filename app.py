import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import quote
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Page Configuration
st.set_page_config(page_title="Kepler Exoplanet Classifier", layout="wide")
st.title("🌌 NASA Kepler Exoplanet Classifier")
st.markdown("Predicting real exoplanets from false positives using Random Forest.")

# The physical and signal features
FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
    'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_srad'
]

@st.cache_data
def load_data():
    query = f"select kepid, koi_disposition, {', '.join(FEATURES)} from cumulative"
    url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=' + quote(query) + '&format=csv'
    return pd.read_csv(url)

with st.spinner("Fetching data from NASA Exoplanet Archive..."):
    df = load_data()

# Data Preprocessing
labeled = df[df.koi_disposition.isin(['CONFIRMED', 'FALSE POSITIVE'])].dropna()
X = labeled[FEATURES]
y = (labeled.koi_disposition == 'CONFIRMED').astype(int)

# Group Shuffle Split to prevent data leakage
split = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups=labeled.kepid))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Model Training
@st.cache_resource
def train_model(X_tr, y_tr):
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_tr, y_tr)
    return model

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

# --- Top Section: Metrics and Predictions ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    st.write(f"**Test F1 Score:** {f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

with col2:
    st.subheader("Top Unconfirmed Exoplanet Candidates")
    candidates = df[df.koi_disposition == 'CANDIDATE'].dropna(subset=FEATURES).copy()
    probabilities = model.predict_proba(candidates[FEATURES])[:, 1]
    candidates['Confidence Score'] = probabilities
    top_candidates = candidates.sort_values(by='Confidence Score', ascending=False)
    st.dataframe(top_candidates[['kepid', 'Confidence Score', 'koi_period', 'koi_prad']].head(10))

st.markdown("---")

# --- Bottom Section: Insights Required by the Guide ---
st.header("Model Insights")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Feature Importances")
    st.write("Which features does the model lean on most?[cite: 12]")
    
    # Extract and plot feature importances
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    importances.plot.barh(ax=ax1, color='#1f77b4')
    ax1.set_xlabel("Importance Score")
    ax1.set_ylabel("Features")
    st.pyplot(fig1)

with col4:
    st.subheader("Learning Curve")
    st.write("How does performance change as the training set shrinks?[cite: 12]")
    
    # Calculate learning curve data (cached to prevent slow reloads)
    @st.cache_data
    def get_learning_curve_data():
        train_fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
        f1_scores = []
        for frac in train_fractions:
            n_samples = int(frac * len(X_train))
            sub_model = RandomForestClassifier(n_estimators=300, random_state=42)
            sub_model.fit(X_train.iloc[:n_samples], y_train.iloc[:n_samples])
            f1_scores.append(f1_score(y_test, sub_model.predict(X_test)))
        return train_fractions, f1_scores
    
    fractions, f1_scores = get_learning_curve_data()
    
    # Plot learning curve
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([f * 100 for f in fractions], f1_scores, marker='o', linestyle='-', color='#ff7f0e')
    ax2.set_xlabel("Training Set Size (%)")
    ax2.set_ylabel("F1 Score")
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)