"""Marketing response & customer segment predictor - Flask backend."""
import sys, warnings
from pathlib import Path
import joblib, pandas as pd
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")

# --- Compatibility shims: the response model was pickled with a different
# scikit-learn build. Remove these if you run the SAME sklearn version used for training.
try:
    from sklearn._loss import _loss as _L
    sys.modules.setdefault("_loss", _L)
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            def __init__(self, *a, **k): super().__init__(*a)
            def __setstate__(self, s): pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

M = Path(__file__).parent / "models"
imputer = joblib.load(M / "imputer.pkl")
scaler = joblib.load(M / "scaler.pkl")
kmeans = joblib.load(M / "kmeans_model.pkl")
km_features = joblib.load(M / "kmeans_features.pkl")
response_model = joblib.load(M / "marketing_response_model.pkl")

def _patch_imputers(obj):
    """Older pickles lack attributes newer sklearn imputers expect."""
    from sklearn.impute import SimpleImputer
    if isinstance(obj, SimpleImputer) and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = getattr(obj, "_fit_dtype", None) or obj.statistics_.dtype
    for sub in getattr(obj, "steps", []):
        _patch_imputers(sub[1])
    for _, t, _c in getattr(obj, "transformers_", []):
        _patch_imputers(t)

_patch_imputers(response_model)
_patch_imputers(imputer)

SEGMENTS = {
    0: {"name": "Value-Conscious Families",
        "desc": "Lower income and spend, more kids at home, frequent web visits and deal purchases."},
    1: {"name": "Premium Spenders",
        "desc": "Higher income and spend, many purchases, fewer deals, more past campaign acceptances."},
}
NUMERIC = ["Income", "Recency", "NumDealsPurchases", "NumWebVisitsMonth", "Complain",
           "Age_At_2014", "Customer_Tenure_Days", "Children_Home", "Total_Spending",
           "Total_Purchases", "Previous_Campaigns_Accepted"]

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    d = request.get_json(force=True)
    row = {k: pd.to_numeric(d.get(k), errors="coerce") for k in NUMERIC}
    row["Education"] = d.get("Education") or None
    row["Marital_Status"] = d.get("Marital_Status") or None
    df = pd.DataFrame([row])

    prob = float(response_model.predict_proba(df)[0][1])

    X = df[km_features].astype(float)
    Xs = scaler.transform(pd.DataFrame(imputer.transform(X), columns=km_features))
    seg = int(kmeans.predict(Xs)[0])

    tier = "High" if prob >= 0.5 else "Medium" if prob >= 0.25 else "Low"
    return jsonify(probability=prob, tier=tier, segment_id=seg, **SEGMENTS[seg])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
