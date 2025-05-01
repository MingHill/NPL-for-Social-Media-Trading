import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import shap
from xgboost import XGBRegressor  # type: ignore
from utils import _load_yaml, CONFIG_PATH
from typing import Sequence
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, embed_cfg: dict):
        self.embed_type = embed_cfg.get("type", "tfidf")
        if self.embed_type == "tfidf":
            tfidf_cfg = embed_cfg.get("tfidf", {})
            self.vectorizer = TfidfVectorizer(
                max_features=int(tfidf_cfg.get("max_features", 1000)),
                ngram_range=tuple(tfidf_cfg.get("ngram_range", (1, 1))),
                stop_words=tfidf_cfg.get("stop_words", None),
            )
        elif self.embed_type == "sentence_transformers":
            if SentenceTransformer is None:
                sys.exit("sentence_transformers not installed. Please install sentence-transformers or change embedding type.")
            st_cfg = embed_cfg.get("sentence_transformers", {})
            model_name = st_cfg.get("model_name", "all-MiniLM-L6-v2")
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
        elif self.embed_type == "none":
            # No text embedding; drop textual features
            pass
        else:
            sys.exit(f"Unsupported embedding type '{self.embed_type}'.")

    def fit(self, texts: Sequence[str]):
        if self.embed_type == "tfidf":
            self.vectorizer.fit(texts)
        # for 'sentence_transformers' and 'none', no fitting needed
        return self

    def transform(self, texts: Sequence[str]):
        if self.embed_type == "tfidf":
            return self.vectorizer.transform(texts)
        if self.embed_type == "sentence_transformers":
            return self.model.encode(texts, show_progress_bar=False)
        # 'none': return empty array with shape (n_samples, 0)
        # Use numpy array for empty features
        import numpy as _np
        n = len(texts)
        return _np.zeros((n, 0))

    def fit_transform(self, texts: Sequence[str]):
        if self.embed_type == "tfidf":
            return self.vectorizer.fit_transform(texts)
        if self.embed_type == "sentence_transformers":
            return self.model.encode(texts, show_progress_bar=False)
        # 'none': return empty array with shape (n_samples, 0)
        import numpy as _np
        n = len(texts)
        return _np.zeros((n, 0))

    def get_feature_names_out(self):
        if self.embed_type == "tfidf":
            return self.vectorizer.get_feature_names_out()
        if self.embed_type == "sentence_transformers":
            dim = self.model.get_sentence_embedding_dimension()
            return [f"embed_{i}" for i in range(dim)]
        # 'none': no text features
        return []

def _get_split_frames(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Path]:
    """Read train/val/test CSVs and return them along with the model_dir."""
    data_dir = Path(cfg["preprocess"].get("output_dir", "data/processed"))
    model_dir = Path(cfg["train_regressor"].get("model_dir", "models/regressor"))

    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        sys.exit(f"Processed data not found at {train_path}. Have you run preprocess.py yet?")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    return train_df, val_df, test_df, model_dir

def _split_xy(df: pd.DataFrame):
    """Return numeric‑only X, separate input_text series, and y."""
    y = df["target"].values
    numeric_X = df.drop(columns=[c for c in ("target", "input_text") if c in df.columns])
    text_series = (
        df["input_text"].fillna("") if "input_text" in df.columns else pd.Series("", index=df.index)
    )
    return numeric_X, text_series, y


def _combine_numeric_text(numeric_X: pd.DataFrame, text_matrix, *, dense: bool = True):
    """Horizontally stack numeric dataframe with sparse TF‑IDF matrix."""
    if dense:
        combined = hstack([numeric_X.values, text_matrix]).toarray()  # type: ignore
    else:
        combined = hstack([numeric_X.values, text_matrix])  # sparse
    return combined

def _evaluate(model, X, y, *, name: str) -> dict[str, float]:
    """Compute and print evaluation metrics, returning them in a dict."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<10} MSE: {mse:,.4f} | RMSE: {rmse:,.4f} | R²: {r2:,.4f}")
    return {"mse": mse, "rmse": rmse, "r2": r2}

def _make_model(model_name: str, params: dict, seed: int, *, use_gpu: bool):
    if model_name == "xgboost":
        return XGBRegressor(
            **params,
            device="cuda" if use_gpu else "cpu",
            objective="reg:squarederror",
            random_state=seed,
        )
    sys.exit(f"Unsupported model '{model_name}'.")


def main() -> None:
    cfg = _load_yaml(CONFIG_PATH)

    train_df, val_df, test_df, output_dir = _get_split_frames(cfg)

    X_train_num, X_train_text, y_train = _split_xy(train_df)
    X_val_num, X_val_text, y_val = _split_xy(val_df) if val_df is not None else (None, None, None)
    X_test_num, X_test_text, y_test = _split_xy(test_df) if test_df is not None else (None, None, None)

    train_cfg: dict = cfg.get("train_regressor", {})
    embedding_cfg = train_cfg.get("embedding", {})
    vectorizer = TextEmbedder(embedding_cfg)
    # If embedding type is 'none', skip text features and use numeric only
    if vectorizer.embed_type == "none":
        # Numeric features only
        X_train = X_train_num.values
        X_val = X_val_num.values if X_val_num is not None else None
        X_test = X_test_num.values if X_test_num is not None else None
    else:
        # Compute embeddings for text
        X_train_text_mat = vectorizer.fit_transform(X_train_text)
        X_val_text_mat = vectorizer.transform(X_val_text) if X_val_text is not None else None
        X_test_text_mat = vectorizer.transform(X_test_text) if X_test_text is not None else None
        # Combine numeric + text features
        if vectorizer.embed_type == "sentence_transformers":
            # Dense embeddings: numpy hstack
            import numpy as _np
            X_train = _np.hstack([X_train_num.values, X_train_text_mat])
            X_val = _np.hstack([X_val_num.values, X_val_text_mat]) if X_val_text_mat is not None else None
            X_test = _np.hstack([X_test_num.values, X_test_text_mat]) if X_test_text_mat is not None else None
        else:
            # Sparse/dense TF-IDF
            X_train = _combine_numeric_text(X_train_num, X_train_text_mat)
            X_val = _combine_numeric_text(X_val_num, X_val_text_mat) if X_val_text_mat is not None else None
            X_test = _combine_numeric_text(X_test_num, X_test_text_mat) if X_test_text_mat is not None else None

    seed: int = cfg["global"].get("seed", 42)
    model_name = str(train_cfg.get("model", "random_forest")).lower()
    use_gpu = bool(train_cfg.get("use_gpu", False))

    model_params = train_cfg.get(model_name, {})
    model = _make_model(model_name, model_params, seed, use_gpu=use_gpu)

    cv_folds = int(train_cfg.get("cv_folds", 0))  # 0/1 disables CV
    if cv_folds > 1:
        print(f"Running {cv_folds}-fold cross‑validation …")
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        print(f"CV RMSE: {(-cv_scores).mean():.4f} ± {(-cv_scores).std():.4f}")

    print("Fitting model …")
    model.fit(X_train, y_train)

    _evaluate(model, X_train, y_train, name="Train")
    if X_val is not None:
        _evaluate(model, X_val, y_val, name="Validation")
    if X_test is not None:
        _evaluate(model, X_test, y_test, name="Test")

    results_dir = Path(train_cfg.get("results_dir", "results/regressor"))
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"true": y_train, "pred": model.predict(X_train)}).to_csv(results_dir / "train_predictions.csv", index=False)
    if X_val is not None:
        pd.DataFrame({"true": y_val, "pred": model.predict(X_val)}).to_csv(results_dir / "validation_predictions.csv", index=False)
    if X_test is not None:
        pd.DataFrame({"true": y_test, "pred": model.predict(X_test)}).to_csv(results_dir / "test_predictions.csv", index=False)

    try:
        print("Computing SHAP feature importances …")
        feature_names = list(X_train_num.columns) + list(vectorizer.get_feature_names_out())
        if model_name in {"random_forest", "lightgbm", "xgboost"}:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
        sample_idx = np.random.choice(len(y_train), size=min(500, len(y_train)), replace=False)
        shap_vals = explainer.shap_values(X_train[sample_idx])
        importances = pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_names).sort_values(ascending=False)
        top_k = int(train_cfg.get("shap_top_k", 50))
        importances.head(top_k).to_csv(results_dir / "shap_top_importances.csv", header=["mean_abs_shap"], index_label="feature")
        print(importances.head(top_k))
    except Exception as e:  # pragma: no cover
        print(f"SHAP computation skipped/not supported for {model_name}: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    print(f"Saved model + vectorizer to '{output_dir}'.")


if __name__ == "__main__":
    main()
