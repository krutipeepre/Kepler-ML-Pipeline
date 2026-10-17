# 🌌 NASA Kepler Exoplanet Classifier

![App Screenshot](image_341605.png) *(Note: You can replace this with a short GIF of you scrolling through the app)*

**Live Demo:** [Explore the Dashboard](https://krutipeepre-kepler-ml-pipeline-app-moun4n.streamlit.app/)

An interactive machine learning dashboard that processes Kepler space telescope data to distinguish confirmed exoplanets from false positives using a Random Forest architecture.

## 📊 The Data
Data is dynamically queried from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)'s Kepler Objects of Interest (KOI) cumulative table[cite: 11, 14].

*   **Features:** The model trains on physical and signal characteristics, including orbital period, transit duration, planet radius, and star temperature[cite: 11].
*   **Leakage Prevention:** Evaluation columns like `koi_score` and `koi_fpflag_` were intentionally excluded from the feature set. Because these are derived from the same vetting process that produced the target labels, including them would artificially inflate accuracy[cite: 12]. 

## 🧠 The Model
*   **Architecture:** Random Forest Classifier (`n_estimators=300`)[cite: 12].
*   **Validation Strategy:** To prevent data leakage, a `GroupShuffleSplit` was applied on the star ID (`kepid`)[cite: 11, 12]. This ensures that multiple planetary candidates orbiting the same star do not cross-contaminate the training and testing sets[cite: 11].
*   **Evaluation:** Performance is tracked beyond basic accuracy, utilizing Precision, Recall, and an F1-score learning curve plotted against training set sizes from 10% to 100%[cite: 11, 12].
*   **Discovery:** The final model predicts on unvetted `CANDIDATE` rows to output a ranked list of the most probable undiscovered exoplanets[cite: 11, 12].

## 🚀 Running Locally

Clone the repository:
   ```bash
   git clone [https://github.com/krutipeepre/](https://github.com/krutipeepre/)<your-repo-name>.git
   cd <your-repo-name>
   ```

1. Install the required dependencies:
```
pip install -r requirements.txt
```
2. Launch the Streamlit application:
```
streamlit run app.py
```

# 🔮 Future Improvements

Moving forward, I would swap the tabular feature dataset for the actual raw light curves using the lightkurve package to train a 1D Convolutional Neural Network directly on the transit dips. Additionally, implementing an asynchronous background job to cache the Exoplanet Archive query into a local CSV would significantly improve the initial load time for end users. Finally, integrating TESS mission data could help generalize the model's predictive power beyond the original Kepler field of view.
