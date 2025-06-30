import re
import os
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask, request, jsonify





# ── Sentiment extractor ──────────────────────────────────────────────────────
class EmojiAndSentimentFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text = X["original_text"].fillna("")
        emoji_count = text.apply(lambda t: sum(1 for c in t if c in __import__("emoji").EMOJI_DATA))
        vader_comp  = text.apply(lambda t: self.analyzer.polarity_scores(t)["compound"])
        return np.vstack([emoji_count, vader_comp]).T

# ── Two‐stage emotion ─────────────────────────────────────────────────────────
class EmotionPipeline:
    def __init__(self, emo2_pkl, emo_pkl, emo_le_pkl):
        self.emo2  = joblib.load(emo2_pkl)   # emotionality
        self.emo   = joblib.load(emo_pkl)    # fine‐emotion
        self.le    = joblib.load(emo_le_pkl) # label encoder

    def predict(self, X):
        Xdf = X[["original_text","english_keywords","primary_theme","sentiment"]].fillna("")
        is_em = self.emo2.predict(Xdf)
        out   = np.array(["neutral"]*len(Xdf), dtype=object)

        mask  = (is_em == 1)
        if mask.any():
            sub = Xdf.loc[mask]
            p   = self.emo.predict(sub)
            out[mask] = self.le.inverse_transform(p)


class ValidityPipeline:
    def __init__(self, cfg):
        # 1) Primary‐theme
        self.prim_model = joblib.load(cfg["PRIM_MODEL_PKL"])
        self.prim_scl   = joblib.load(cfg["PRIM_SCALER_PKL"])
        self.prim_le    = joblib.load(cfg["PRIM_LE_PKL"])      # <— load label encoder

        # 2) Secondary‐theme
        self.sec_models = joblib.load(cfg["SEC_MODELS_PKL"])
        self.sec_scl    = joblib.load(cfg["SEC_SCALER_PKL"])

        # 3) Sentiment
        self.sent_pipe  = joblib.load(cfg["SENT_PKL"])

        # 4) Full emotion pipeline
        full_emo_pipe   = joblib.load(cfg["FULL_EMO_PKL"])
        self.emo2_model = full_emo_pipe.emotionality_model
        self.emo_model  = full_emo_pipe.emotion_model
        self.emo_le     = joblib.load(cfg["EMO_LE_PKL"])
        self.emo_ohe    = OneHotEncoder(
            categories=[list(self.emo_le.classes_)],
            handle_unknown="ignore",
            sparse_output=False
        )

        # 5) Final validity classifier
        self.final_clf  = joblib.load(cfg["FINAL_MODEL_OUT"])

    def predict_primary_theme(self, X_embed, stats):
        """Scale stats, stack with embeddings, then predict and decode."""
        stats_s = self.prim_scl.transform(stats)
        Xp      = np.hstack([X_embed, stats_s])
        proba   = self.prim_model.predict_proba(Xp)
        idx     = np.argmax(proba, axis=1)
        return self.prim_le.inverse_transform(idx)

    def preprocess(self, texts, english_keywords=None):
        n = len(texts)
        ek = english_keywords or [""]*n
        df = pd.DataFrame({
            "original_text":    texts,
            "english_keywords": ek
        })

        # 1) get embeddings & text‐stats (exactly as before) …
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        X_embed = embedder.encode(
            df["original_text"] + " " + df["english_keywords"],
            batch_size=64, convert_to_numpy=True
        )

        df["combined"]   = df["original_text"] + " " + df["english_keywords"]
        df["kw_overlap"] = df["combined"].apply(
            lambda t: len(set(re.findall(r"\b[a-z]{3,}\b", t.lower())))
        )
        df["txt_len"] = df["original_text"].str.len()
        df["has_q"]   = df["original_text"].str.contains(r"\?").astype(int)
        df["caps"]    = df["original_text"].str.count(r"[A-Z]")
        df["excl"]    = df["original_text"].str.count(r"!")

        stats = df[["kw_overlap","txt_len","has_q","caps","excl"]].values

        # 2) PRIMARY THEME FIRST
        df["primary_theme"] = self.predict_primary_theme(X_embed, stats)

        # 3) DUMMY main_emotion so sentiment pipeline will accept it
        df["main_emotion"] = ""    # or "neutral" — just must exist

        # 4) NOW FILL SENTIMENT (requires 4 cols)
        Xs = df[[
            "english_keywords",
            "original_text",
            "primary_theme",
            "main_emotion"
        ]].fillna("")
        df["sentiment"] = self.sent_pipe.predict(Xs).astype(float)

        # 5) TRUE two‐stage EMOTION (overwrites dummy)
        emot_flag = self.emo2_model.predict(
            df[["original_text","english_keywords","primary_theme","sentiment"]]
        )
        df["main_emotion"] = "neutral"
        idxs = np.where(emot_flag == 1)[0]
        if len(idxs):
            sub   = df.iloc[idxs]
            fine  = self.emo_model.predict(
                sub[["original_text","english_keywords","primary_theme","sentiment"]]
            )
            df.loc[idxs, "main_emotion"] = self.emo_le.inverse_transform(fine)

        emo_feat = self.emo_ohe.fit_transform(df[["main_emotion"]])

        # 6) SECONDARY THEME PROBS, FINAL STACK …
        sec_stats   = np.hstack([stats, df["sentiment"].values.reshape(-1,1)])
        sec_stats_s = self.sec_scl.transform(sec_stats)

        sec_probs = []
        for m in self.sec_models:
            sec_probs.append(
                m.predict_proba(
                    np.hstack([X_embed, sec_stats_s, emo_feat])
                )[:,1]
            )

        X_final = np.hstack([
            self.prim_model.predict_proba(
                np.hstack([X_embed, self.prim_scl.transform(stats)])
            ),
            np.vstack(sec_probs).T,
            emo_feat,
            df["sentiment"].values.reshape(-1,1)
        ])

        return X_final
        return X_final
    def predict(self, texts, english_keywords=None):
        Xf     = self.preprocess(texts, english_keywords)
        preds  = self.final_clf.predict(Xf)
        probs  = self.final_clf.predict_proba(Xf)[:,1]
        return preds, probs



FULL_PIPE_PKL = os.path.join(BASE_DIR, "pipe", "validity_full_pipeline.pkl")
pipe = joblib.load(FULL_PIPE_PKL)  # now it can find ValidityPipeline, Embedder, etc.


# ─── Load/construct the keyword_bank ────────────────────
import pandas as pd

KW_BANK_CSV  = "/Users/twh/Desktop/validity_App/pipe/primary/theme_keyword_bank.csv"
kw_df        = pd.read_csv(KW_BANK_CSV).dropna(how="all")
keyword_bank = {
    theme: [str(w).lower() for w in kw_df[theme].dropna()]
    for theme in kw_df.columns
}


# inference_validity.py

import re
import csv

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

# ─── 1) CONFIG: point these to your saved files ───────────────────────────

BASE_DIR = os.path.dirname(__file__)

# Primary‐theme artifacts
PRIM_MODEL_PKL  = os.path.join(BASE_DIR, "pipe", "primary", "primary_theme_classifier_e5_fixed.pkl")
PRIM_SCALER_PKL = os.path.join(BASE_DIR, "pipe", "primary", "handcrafted_scaler.pkl")
LABEL_LE_PKL    = os.path.join(BASE_DIR, "pipe", "primary", "primary_theme_label_encoder_embeddings.pkl")

# Secondary‐theme artifacts
SEC_MODELS_PKL  = os.path.join(BASE_DIR, "pipe", "secondary_theme_classifier.pkl")
SEC_SCALER_PKL  = os.path.join(BASE_DIR, "pipe", "sec_scaler.pkl")

# Sentiment & emotion pipelines
SENT_PKL        = os.path.join(BASE_DIR, "pipe", "sentiment", "sentiment_model_with_emoji_vader.pkl")
FULL_EMO_PKL    = os.path.join(BASE_DIR, "pipe", "Emotion", "full_emotion_pipeline.pkl")
EMO_LE_PKL      = os.path.join(BASE_DIR, "pipe", "Emotion", "emotion_label_encoder_low_neutral.pkl")

# Final validity classifier
FINAL_MODEL_PKL = os.path.join(BASE_DIR, "pipe", "validity_classifier.pkl")

# Keyword bank CSV
KW_BANK_CSV     = os.path.join(BASE_DIR, "pipe", "primary", "theme_keyword_bank.csv")

# ─── 2) LOAD all your artifacts ────────────────────────────────────────────

# a) Primary‐theme model & scaler
prim_model   = joblib.load(PRIM_MODEL_PKL)
prim_scaler  = joblib.load(PRIM_SCALER_PKL)

# b) Secondary‐theme models & scaler
sec_models   = joblib.load(SEC_MODELS_PKL)  # list of one‐vs‐rest XGBs
sec_scaler   = joblib.load(SEC_SCALER_PKL)

# c) Sentiment‐fill pipeline
sent_pipe    = joblib.load(SENT_PKL)

# d) Emotion pipeline
emo_full     = joblib.load(FULL_EMO_PKL)
emo2_model   = emo_full.emotionality_model
emo_model    = emo_full.emotion_model
emo_le       = joblib.load(EMO_LE_PKL)
ohe_emotion  = OneHotEncoder(
    categories=[list(emo_le.classes_)],
    sparse_output=False,
    handle_unknown="ignore"
)
# “Prime” the OHE so it knows its categories
_ohe_dummy   = ohe_emotion.fit(np.array(emo_le.classes_).reshape(-1,1))

# e) Embedder & VADER
embedder     = SentenceTransformer("intfloat/e5-base-v2")
vader        = SentimentIntensityAnalyzer()

# f) Final validity classifier
valid_clf    = joblib.load(FINAL_MODEL_PKL)

# g) Keyword bank for text‐stats
kw_df        = pd.read_csv(KW_BANK_CSV).dropna(how="all")
keyword_bank = {
    theme: [str(w).lower() for w in kw_df[theme].dropna()]
    for theme in kw_df.columns
}


# ─── 3) UTILS to compute each block of features ────────────────────────────

def get_sentiment(text: str) -> float:
    """
    Run your saved sentiment pipeline on a single‐row DataFrame.
    Returns a float in [-1.0, +1.0].
    """
    df = pd.DataFrame({
        "original_text":    [text],
        "english_keywords": [""],      # no precomputed keywords in inference
        "primary_theme":    [""],
        "main_emotion":     [""]
    })
    score = sent_pipe.predict(df)[0]
    # If the output is a string label, map to numeric
    if isinstance(score, str):
        m = {"neg": -1, "negative": -1, "neu": 0, "neutral": 0, "pos": 1, "positive": 1}
        score = m.get(score, 0)
    return float(score)


def get_main_emotion(text: str) -> str:
    """
    First run the emotionality model (0 or 1). If 0, return "neutral".
    Otherwise, run the fine‐emotion model and map back via LabelEncoder.
    """
    row = {
        "original_text":    text,
        "english_keywords": "",
        "primary_theme":    "",
        "sentiment":        get_sentiment(text)
    }
    flag = emo2_model.predict(pd.DataFrame([row]))[0]
    if flag == 0:
        return "neutral"
    lbl = emo_model.predict(pd.DataFrame([row]))[0]
    return emo_le.inverse_transform([lbl])[0]


def get_emo_ohe(emotion_label: str) -> np.ndarray:
    """
    One‐hot encode the single emotion label (shape returned = (1, 26)).
    """
    return ohe_emotion.transform([[emotion_label]])


def text_stats(text: str, theme: str="") -> np.ndarray:
    """
    Compute the five handcrafted text‐stats:
     1) kw_overlap (words ∩ keyword_bank[theme])
     2) length of text
     3) has question mark?
     4) number of uppercase letters
     5) number of exclamation marks
    """
    txt    = text
    words  = set(re.findall(r"\b[a-z]{3,}\b", txt.lower()))
    kw_ov  = len(words & set(keyword_bank.get(theme, [])))
    return np.array([
        kw_ov,
        len(txt),
        int("?" in txt),
        len(re.findall(r"[A-Z]", txt)),
        len(re.findall(r"!", txt))
    ], dtype=float)


def get_primary_probs(text: str) -> np.ndarray:
    """
    1) Compute embedding of [text] → shape (1, D_embed)
    2) Compute text_stats(text) → shape (1, 5)
    3) Scale the 5 stats via prim_scaler → shape (1, 5)
    4) hstack: [emb (1×D) ‖ scaled_stats (1×5)] → shape (1, D+5)
    5) prim_model.predict_proba(...) → array of length = n_primary_classes
    """
    emb   = embedder.encode([text])                    # (1, D_embed)
    stats = text_stats(text).reshape(1, -1)             # (1, 5)
    Xp    = np.hstack([emb, prim_scaler.transform(stats)])  # (1, D+5)
    return prim_model.predict_proba(Xp)[0]              # (n_primary,)


def get_secondary_probs(text: str,
                        sentiment: float,
                        emo_ohe: np.ndarray) -> np.ndarray:
    """
    1) emb = embedder.encode([text])            → (1, D_embed)
    2) stats = text_stats(text).reshape(1,-1)   → (1, 5)
    3) sent_arr = [[sentiment]]                → (1, 1)
    4) sec_in = [stats (1×5) ‖ sent_arr (1×1)]  → (1, 6)
    5) sec_scaled = sec_scaler.transform(sec_in)   → (1, 6)
    6) Xb = [emb (1×D) ‖ sec_scaled (1×6) ‖ emo_ohe (1×26)] → (1, D+6+26)
    7) Run each of the sec_models on Xb, return array of their “prob( class = 1 )”
       → shape (n_secondary_models,)
    """
    emb      = embedder.encode([text])                # (1, D_embed)
    stats    = text_stats(text).reshape(1, -1)         # (1, 5)
    sent_arr = np.array([[sentiment]])                 # (1, 1)
    sec_in   = np.hstack([stats, sent_arr])            # (1, 6)
    sec_s    = sec_scaler.transform(sec_in)            # (1, 6)
    Xb       = np.hstack([emb, sec_s, emo_ohe])        # (1, D+6+26)
    return np.array([m.predict_proba(Xb)[0, 1] for m in sec_models])  # (n_sec_models,)


def infer_validity(text: str) -> tuple[int, float]:
    """
    1) sentiment = get_sentiment(text)
    2) main_emotion = get_main_emotion(text)
    3) emo_ohe = get_emo_ohe(main_emotion)       → shape (1,26)
    4) pp = get_primary_probs(text).reshape(1,-1) → shape (1, n_primary)
    5) sp = get_secondary_probs(text, sentiment, emo_ohe).reshape(1,-1)
    6) Xf = [pp (1×n_p) ‖ sp (1×n_s) ‖ emo_ohe (1×26) ‖ [[sentiment]] (1×1)]
         → shape (1, total_features)
    7) pred = valid_clf.predict(Xf)[0]
    8) conf = valid_clf.predict_proba(Xf)[0,1]   # prob “1” = REAL
    """
    s   = get_sentiment(text)
    me  = get_main_emotion(text)
    eo  = get_emo_ohe(me)                            # (1,26)

    pp  = get_primary_probs(text).reshape(1, -1)      # (1, n_primary)
    sp  = get_secondary_probs(text, s, eo).reshape(1, -1)  # (1, n_secondary)

    s_arr = np.array([[s]])                           # (1,1)
    Xf    = np.hstack([pp, sp, eo, s_arr])            # (1, total_features)

    pred = valid_clf.predict(Xf)[0]
    conf = valid_clf.predict_proba(Xf)[0, 1]
    return int(pred), float(conf)


def extract_features(text: str) -> np.ndarray:
    """
    Return exactly the 1D feature vector used by infer_validity:
      [primary_probs (n_p,) ‖ secondary_probs (n_s,) ‖ emo_onehot (26,) ‖ sentiment (1,)]
    """
    s   = get_sentiment(text)
    me  = get_main_emotion(text)
    eo  = get_emo_ohe(me).flatten()                  # (26,)
    pp  = get_primary_probs(text).flatten()          # (n_p,)
    sp  = get_secondary_probs(text, s, eo.reshape(1, -1)).flatten()  # (n_s,)
    return np.concatenate([pp, sp, eo, [s]])


# ─── 4) FEEDBACK LOGGER ─────────────────────────────────────────────────────

def record_feedback(text: str,
                    predicted: int,
                    correct: int,
                    filepath: str="feedback.csv") -> None:
    """
    Compute the feature vector for `text` and append:
      [text, predicted, correct, f0, f1, ..., fN]
    to `feedback.csv`.  Write header once if file doesn’t exist.
    """
    feats = extract_features(text).tolist()
    row   = [text, predicted, correct] + feats

    # Write header if the file doesn't exist yet
    try:
        with open(filepath, 'x', newline='') as f:
            writer = csv.writer(f)
            header = (
                ["original_text", "predicted", "correct"]
                + [f"f{i}" for i in range(len(feats))]
            )
            writer.writerow(header)
    except FileExistsError:
        pass

    # Append the data row
    with open(filepath, 'a', newline='') as f:
        csv.writer(f).writerow(row)


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    pred, conf = infer_validity(data["comment"])
    return jsonify(validity=pred, confidence=conf)

# ─── 5) CLI LOOP ───────────────────────────────────────────────────────────
"""
if __name__ == "__main__":
    print("Enter a comment, I’ll predict VALID (1) or FAKE (0).")
    print("Then type the correct label [0/1] to log feedback.")
    while True:
        txt = input("> ").strip()
        if not txt:
            continue

        pred, conf = infer_validity(txt)
        print(f"→ I predict {pred} (confidence: {conf:.1%})")

        corr = input("Correct label? [0/1 or Enter to skip] ").strip()
        if corr in ("0", "1"):
            record_feedback(txt, pred, int(corr))
            print("✅ Logged text, prediction, correction, + features.")
"""


# 
if __name__ == "__main__":
    # Read PORT from env (defaults to 5000)
    port = int(os.environ.get("PORT", 5000))
    # Start Flask’s built-in dev server; Docker or gunicorn 
    # will bypass this when you use `gunicorn app:app`
    app.run(host="0.0.0.0", port=port, debug=True)