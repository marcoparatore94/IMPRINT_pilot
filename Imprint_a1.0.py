# ============================================================
# IMPRINT analysis with real benign + malignant patients
# Constant features auto-removed
# Bootstrap AUC CIs, PR-AUC, calibration slope/intercept
# Risk stratification: fixed thresholds + quantile tertiles
# Boxplots split across pages (Figure 6.1, 6.2, 6.3)
# Methodological notes and reproducibility metadata on separate pages
# Clean visuals, constrained layout, histogram of CV probabilities with threshold lines
# Safe PDF overwrite with timestamp fallback
# ============================================================

import os, datetime, platform, textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, average_precision_score, brier_score_loss, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import sklearn

# ------------------------------------------------------------
# Rendering settings (clean visuals and stable layout)
# ------------------------------------------------------------
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.dpi'] = 150

# ------------------------------------------------------------
# Paths and safe report handling
# ------------------------------------------------------------
benign_path = r"C:\Users\marco\PycharmProjects\IMPRINT_pilot\DB Imprint_benign.xlsx"
malignant_path = r"C:\Users\marco\PycharmProjects\IMPRINT_pilot\DB_Imprint_malignant.xlsx"
report_path = r"C:\Users\marco\PycharmProjects\IMPRINT_pilot\IMPRINT_report.pdf"
export_path = r"C:\Users\marco\PycharmProjects\IMPRINT_pilot\IMPRINT_dataset.csv"

def get_safe_report_path(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            return path
        except PermissionError:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = os.path.join(os.path.dirname(path), f"IMPRINT_report_{ts}.pdf")
            print(f"⚠️ Report locked; saving to: {alt}")
            return alt
    return path

report_path = get_safe_report_path(report_path)

# ------------------------------------------------------------
# Data specification
# ------------------------------------------------------------
markers_all = ["NLR","PLR","SII","SIRI","LMR","MLR","ELR","PIV"]
markers_roc = ["NLR","PLR","SII","SIRI","PIV"]
features_model_all = ["NLR","SII","SIRI","PIV","Age","Diameter_gt8","Irregular_margins","Color4","Shadows_absent"]

# ------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------
df_ben = pd.read_excel(benign_path)
df_mal = pd.read_excel(malignant_path)

benign = df_ben[markers_all].copy()
malignant = df_mal[markers_all].copy()
benign["label"] = 0
malignant["label"] = 1

def add_ecography_vars(df, src):
    df["Age"] = src["Age at diagnosis"] if "Age at diagnosis" in src.columns else np.nan
    # Ultrasound flags fixed to "orange" scenario
    df["Diameter_gt8"] = 1
    df["Irregular_margins"] = 1
    df["Color4"] = 1
    df["Shadows_absent"] = 1
    return df

benign = add_ecography_vars(benign, df_ben)
malignant = add_ecography_vars(malignant, df_mal)

data = pd.concat([malignant, benign], ignore_index=True)
for col in features_model_all:
    data[col] = pd.to_numeric(data[col], errors="coerce")

n_total = len(data)
prevalence = float(np.mean(data["label"] == 1))

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def youden_cutoff(y_true, y_scores):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    J = tpr - fpr
    ix = int(np.argmax(J))
    return {
        "cutoff": float(thr[ix]),
        "sens": float(tpr[ix]),
        "spec": float(1 - fpr[ix]),
        "auc": float(auc(fpr, tpr)),
        "fpr": fpr,
        "tpr": tpr,
        "ix": ix
    }

def ppv_npv(sens, spec, prev):
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    npv = (spec * (1 - prev)) / ((1 - sens) * prev + spec * (1 - prev))
    return float(ppv), float(npv)

def classify_risk(prob):
    if prob < 0.05: return "Low"
    elif prob < 0.20: return "Intermediate"
    else: return "High"

def tight_save(pdf, fig):
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# ------------------------------------------------------------
# Single-marker ROC and PPV/NPV scenarios
# ------------------------------------------------------------
results = []
for m in markers_roc:
    vals = data[m].copy().astype(float).fillna(data[m].median())
    res = youden_cutoff(data["label"], vals)
    ppv5, npv5 = ppv_npv(res["sens"], res["spec"], 0.05)
    ppv10, npv10 = ppv_npv(res["sens"], res["spec"], 0.10)
    ppv20, npv20 = ppv_npv(res["sens"], res["spec"], 0.20)
    results.append([m, res["cutoff"], res["sens"], res["spec"], res["auc"],
                    ppv5, npv5, ppv10, npv10, ppv20, npv20])

# ------------------------------------------------------------
# Drop constant features
# ------------------------------------------------------------
features_model = [f for f in features_model_all if data[f].nunique(dropna=False) > 1]

# ------------------------------------------------------------
# Combined model with CV, PR-AUC, bootstrap AUC CI, calibration slope/intercept
# ------------------------------------------------------------
X = data[features_model].values
y = data["label"].values

pipeline = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500, solver="liblinear"))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
y_scores_cv_all = np.zeros_like(y, dtype=float)

for tr, te in cv.split(X, y):
    pipeline.fit(X[tr], y[tr])
    yhat = pipeline.predict_proba(X[te])[:, 1]
    y_scores_cv_all[te] = yhat
    fpr, tpr, _ = roc_curve(y[te], yhat)
    auc_scores.append(auc(fpr, tpr))

auc_cv_mean = float(np.mean(auc_scores))
auc_cv_std = float(np.std(auc_scores))

pipeline.fit(X, y)
y_score_full = pipeline.predict_proba(X)[:, 1]
fpr_m, tpr_m, _ = roc_curve(y, y_score_full)
auc_m_full = float(auc(fpr_m, tpr_m))

# PR-AUC and PR curve
pr_auc = float(average_precision_score(y, y_scores_cv_all))
precision, recall, pr_thr = precision_recall_curve(y, y_scores_cv_all)

# Bootstrap AUC CI
rng = np.random.default_rng(42)
boot_aucs = []
for i in range(1000):
    idx = rng.choice(len(y), len(y), replace=True)
    Xb, yb = X[idx], y[idx]
    pipeline.fit(Xb, yb)
    yhatb = pipeline.predict_proba(Xb)[:, 1]
    fpr_b, tpr_b, _ = roc_curve(yb, yhatb)
    boot_aucs.append(auc(fpr_b, tpr_b))
ci_low, ci_high = [float(x) for x in np.percentile(boot_aucs, [2.5, 97.5])]

# Calibration slope/intercept
eps = 1e-6
logit_p = np.log((y_scores_cv_all + eps) / (1 - y_scores_cv_all + eps)).reshape(-1, 1)
cal_lr = LogisticRegression(max_iter=500, solver="liblinear").fit(logit_p, y)
cal_slope = float(cal_lr.coef_[0][0])
cal_intercept = float(cal_lr.intercept_[0])

# Variable importance (signed coefficients)
coefs = pipeline.named_steps["lr"].coef_[0]
coef_series = pd.Series(coefs, index=features_model)
coef_sorted_abs = coef_series.abs().sort_values(ascending=False)

# Odds ratios (univariate approximation) with bootstrap CIs
or_ci = {}
for feat in features_model:
    vals = data[feat].copy().astype(float).fillna(data[feat].median()).values.reshape(-1, 1)
    boot_or = []
    for i in range(300):
        idx = rng.choice(len(y), len(y), replace=True)
        Xi, yi = vals[idx], y[idx]
        try:
            lr = LogisticRegression(max_iter=500, solver="liblinear").fit(Xi, yi)
            boot_or.append(np.exp(lr.coef_[0][0]))
        except Exception:
            continue
    if len(boot_or) >= 30:
        med = float(np.median(boot_or))
        lo = float(np.percentile(boot_or, 2.5))
        hi = float(np.percentile(boot_or, 97.5))
        # filter non-informative constant ORs (1.00, 1.00–1.00)
        if not (round(med, 2) == 1.00 and round(lo, 2) == 1.00 and round(hi, 2) == 1.00):
            or_ci[feat] = (med, lo, hi)

# Missing counts and percentages
missing_counts = data[features_model_all].isna().sum().reindex(features_model_all)
missing_pct = (missing_counts / n_total * 100).round(1)

# Risk stratification (fixed thresholds and tertiles)
risk_classes = pd.Series([classify_risk(p) for p in y_scores_cv_all])
risk_counts = risk_classes.value_counts().reindex(["Low","Intermediate","High"]).fillna(0).astype(int)

quant_labels = ["Low tertile", "Mid tertile", "High tertile"]
quant_strata = pd.qcut(y_scores_cv_all, 3, labels=quant_labels)
quant_counts = quant_strata.value_counts().reindex(quant_labels)

# Calibration curve and Brier score
prob_true, prob_pred = calibration_curve(y, y_scores_cv_all, n_bins=10, strategy="uniform")
brier = float(brier_score_loss(y, y_scores_cv_all))

# Decision curve analysis (focused 5–30% thresholds)
thresholds = np.linspace(0.05, 0.30, 30)
net_benefit_model, net_benefit_all, net_benefit_none = [], [], []
event_rate = float(np.mean(y == 1))
for thr in thresholds:
    pred_pos = y_scores_cv_all >= thr
    tp = np.sum((pred_pos == 1) & (y == 1))
    fp = np.sum((pred_pos == 1) & (y == 0))
    n = len(y)
    nb_model = (tp / n) - (fp / n) * (thr / (1 - thr))
    nb_all = (event_rate) - (1 - event_rate) * (thr / (1 - thr))
    net_benefit_model.append(nb_model)
    net_benefit_all.append(nb_all)
    net_benefit_none.append(0.0)

# ------------------------------------------------------------
# PDF report (figures/tables only; notes and metadata split)
# ------------------------------------------------------------
with PdfPages(report_path) as pdf:
    # Table 1: Youden cut-offs and PPV/NPV
    fig_tbl, ax_tbl = plt.subplots(figsize=(12, 5))
    ax_tbl.axis("off")
    header = ["Marker","Cut-off","Sensitivity","Specificity","AUC",
              "PPV 5%","NPV 5%","PPV 10%","NPV 10%","PPV 20%","NPV 20%"]
    rows = [[r[0], f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.3f}",
             f"{r[5]:.2f}", f"{r[6]:.2f}", f"{r[7]:.2f}", f"{r[8]:.2f}", f"{r[9]:.2f}", f"{r[10]:.2f}"]
            for r in results]
    table = ax_tbl.table(cellText=[header] + rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.05, 1.25)
    ax_tbl.set_title("Table 1. Youden cut-offs and PPV/NPV at different prevalences", fontsize=12, pad=10)
    tight_save(pdf, fig_tbl)

    # Figure 1: ROC curves of single markers
    fig_roc, ax_roc = plt.subplots(figsize=(8.5, 7.5))
    for m in markers_roc:
        vals = data[m].copy().astype(float).fillna(data[m].median())
        r = youden_cutoff(data["label"], vals)
        ax_roc.plot(r["fpr"], r["tpr"], label=f"{m} (AUC={r['auc']:.3f})")
        ax_roc.scatter(r["fpr"][r["ix"]], r["tpr"][r["ix"]], s=40)
    ax_roc.plot([0,1],[0,1],"k--")
    ax_roc.set_xlabel("1 - Specificity"); ax_roc.set_ylabel("Sensitivity")
    ax_roc.set_title("Figure 1. ROC curves of single markers\nwith Youden point")
    ax_roc.legend(loc="lower right")
    tight_save(pdf, fig_roc)

    # Figure 2: Combined ROC with AUC CI and PR-AUC
    fig_model, ax_model = plt.subplots(figsize=(8.5, 7.5))
    ax_model.plot(fpr_m, tpr_m, label=f"Combined (in-sample AUC={auc_m_full:.3f})", color="crimson", linewidth=2)
    ax_model.plot([0,1],[0,1],"k--")
    ax_model.set_xlabel("1 - Specificity"); ax_model.set_ylabel("Sensitivity")
    ax_model.set_title(
        f"Figure 2. Combined ROC\n(CV AUC={auc_cv_mean:.3f} ± {auc_cv_std:.3f}; "
        f"bootstrap 95% CI [{ci_low:.3f}, {ci_high:.3f}]; PR-AUC={pr_auc:.3f})"
    )
    ax_model.legend(loc="lower right")
    tight_save(pdf, fig_model)

    # Figure 2b: Precision–Recall curve
    fig_pr, ax_pr = plt.subplots(figsize=(8.5, 6.5))
    ax_pr.plot(recall, precision, color="navy", label=f"Model (PR-AUC={pr_auc:.3f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Figure 2b. Precision–Recall curve\n(CV probabilities)")
    ax_pr.legend(loc="lower left")
    tight_save(pdf, fig_pr)

    # Figure 3: Variable importance (signed standardized coefficients)
    fig_coef, ax_coef = plt.subplots(figsize=(8, 6))
    coef_series.sort_values().plot(kind="barh", ax=ax_coef, color="steelblue")
    ax_coef.set_title("Figure 3. Variable importance\n(signed standardized coefficients)")
    ax_coef.set_xlabel("Coefficient (standardized)")
    tight_save(pdf, fig_coef)

    # Figure 3b: Distribution of cross-validated probabilities (with fixed-threshold lines)
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(y_scores_cv_all, bins=20, color="steelblue", edgecolor="white")
    ax_hist.axvline(0.05, color="gray", linestyle="--", linewidth=1)
    ax_hist.axvline(0.20, color="gray", linestyle="--", linewidth=1)
    ax_hist.set_title("Figure 3b. Distribution of cross-validated predicted probabilities")
    ax_hist.set_xlabel("Predicted probability"); ax_hist.set_ylabel("Count")
    tight_save(pdf, fig_hist)

    # Table 2: Missing values imputed per feature (counts + %)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.axis("off")
    imp_header = ["Feature","NaN imputed (median)","Missing %"]
    imp_rows = [[var, int(missing_counts[var]), f"{missing_pct[var]:.1f}%"] for var in features_model_all]
    imp_table = ax_imp.table(cellText=[imp_header] + imp_rows, loc="center", cellLoc="center")
    imp_table.auto_set_font_size(False); imp_table.set_fontsize(9); imp_table.scale(1.05, 1.25)
    ax_imp.set_title("Table 2. Missing values imputed per feature", fontsize=12, pad=10)
    tight_save(pdf, fig_imp)

    # Table 3a: Risk stratification (fixed thresholds)
    fig_risk, ax_risk = plt.subplots(figsize=(8, 4.8))
    ax_risk.axis("off")
    risk_header = ["Risk class (fixed thresholds)","Count"]
    risk_rows = [["Low (<5%)", int(risk_counts.get("Low",0))],
                 ["Intermediate (5–20%)", int(risk_counts.get("Intermediate",0))],
                 ["High (>20%)", int(risk_counts.get("High",0))]]
    risk_table = ax_risk.table(cellText=[risk_header] + risk_rows, loc="center", cellLoc="center")
    risk_table.auto_set_font_size(False); risk_table.set_fontsize(9); risk_table.scale(1.05, 1.25)
    ax_risk.set_title("Table 3a. Risk stratification\n(CV probabilities; fixed thresholds)", fontsize=12, pad=10)
    tight_save(pdf, fig_risk)

    # Table 3b: Risk stratification (quantile tertiles)
    fig_risk_q, ax_risk_q = plt.subplots(figsize=(8, 4.8))
    ax_risk_q.axis("off")
    risk_q_header = ["Risk class (tertiles)","Count"]
    risk_q_rows = [[lbl, int(quant_counts.get(lbl,0))] for lbl in quant_labels]
    risk_q_table = ax_risk_q.table(cellText=[risk_q_header] + risk_q_rows, loc="center", cellLoc="center")
    risk_q_table.auto_set_font_size(False); risk_q_table.set_fontsize(9); risk_q_table.scale(1.05, 1.25)
    ax_risk_q.set_title("Table 3b. Risk stratification\n(CV probabilities; quantile tertiles)", fontsize=12, pad=10)
    tight_save(pdf, fig_risk_q)

    # Figure 4: Calibration plot
    fig_cal, ax_cal = plt.subplots(figsize=(7.5, 6.5))
    ax_cal.plot([0,1],[0,1],"k--", label="Perfect calibration")
    ax_cal.plot(prob_pred, prob_true, marker="o", label="Model (CV)")
    ax_cal.set_xlabel("Predicted probability"); ax_cal.set_ylabel("Observed event rate")
    ax_cal.set_title(
        f"Figure 4. Calibration plot\n(bins=10 uniform; Brier={brier:.3f}; slope={cal_slope:.2f}; intercept={cal_intercept:.2f})"
    )
    ax_cal.legend(loc="upper left")
    tight_save(pdf, fig_cal)

    # Figure 5: Decision curve analysis (clean legend labels)
    fig_dca, ax_dca = plt.subplots(figsize=(8.5, 6.5))
    ax_dca.plot(thresholds, net_benefit_model, label="Model", color="crimson")
    ax_dca.plot(thresholds, net_benefit_all, label="Treat all", color="gray")
    ax_dca.plot(thresholds, [0.0]*len(thresholds), label="Treat none", color="black")
    ax_dca.set_xlabel("Risk threshold"); ax_dca.set_ylabel("Net benefit")
    ax_dca.set_title("Figure 5. Decision curve analysis\n(focused 5–30%)")
    ax_dca.legend(loc="upper right")
    tight_save(pdf, fig_dca)

    # Figure 6: Boxplots of NLR, SII, SIRI (all in one page, without outliers)
    fig_box, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, marker in zip(axes, ["NLR", "SII", "SIRI"]):
        ben_vals = pd.to_numeric(data[marker][data["label"] == 0], errors="coerce").fillna(data[marker].median())
        mal_vals = pd.to_numeric(data[marker][data["label"] == 1], errors="coerce").fillna(data[marker].median())

        bp = ax.boxplot([ben_vals, mal_vals],
                        widths=0.8,
                        patch_artist=True,
                        showfliers=False)  # <-- esclude i valori estremi

        bp['boxes'][0].set_facecolor('#89CFF0')
        bp['boxes'][1].set_facecolor('#F4A3A3')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Benign", "Malignant"])
        ax.set_title(marker)
        ax.set_ylabel("Value")

    fig_box.suptitle("Figure 6. Boxplots of NLR, SII, and SIRI (Benign vs Malignant, outliers excluded)")
    tight_save(pdf, fig_box)

    # Figure 7: Signed coefficients
    fig_coef2, ax_coef2 = plt.subplots(figsize=(9, 6))
    coef_series.sort_values().plot(kind="barh", ax=ax_coef2, color="steelblue")
    ax_coef2.set_title("Figure 7. Signed coefficients\n(standardized)")
    ax_coef2.set_xlabel("Coefficient")
    tight_save(pdf, fig_coef2)

    # Table 4: Odds ratios (bootstrap CIs), filtered for non-informative rows
    fig_or, ax_or = plt.subplots(figsize=(10, 5))
    ax_or.axis("off")
    or_header = ["Feature","Median OR","95% CI low","95% CI high"]
    or_rows = []
    for feat in features_model:
        if feat in or_ci:
            med, lo, hi = or_ci[feat]
            or_rows.append([feat, f"{med:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
    if not or_rows:
        or_rows = [["(No informative features)", "-", "-", "-"]]
    or_table = ax_or.table(cellText=[or_header] + or_rows, loc="center", cellLoc="center")
    or_table.auto_set_font_size(False); or_table.set_fontsize(9); or_table.scale(1.05, 1.25)
    ax_or.set_title("Table 4. Odds ratios (bootstrap CIs)\n(univariate approximation)", fontsize=12, pad=10)
    tight_save(pdf, fig_or)

    # Final page 1: Methodological Notes
    fig_notes = plt.figure(figsize=(8.5, 11))
    ax_notes = fig_notes.add_subplot(111); ax_notes.axis("off")
    notes = [
        "Table 1: Cut-offs via Youden index. PPV/NPV shown at 5%, 10%, 20% prevalence scenarios.",
        "Figure 1: ROC curves for single markers; Youden points mark the balanced threshold.",
        f"Figure 2: Combined ROC. CV AUC mean={auc_cv_mean:.3f} (SD={auc_cv_std:.3f}); bootstrap 95% CI [{ci_low:.3f}, {ci_high:.3f}].",
        f"Figure 2b: Precision–Recall curve; PR-AUC={pr_auc:.3f}. Prefer PR-AUC when class imbalance is relevant.",
        "Figure 3 & 7: Variable importance shown via signed standardized coefficients; interpret multivariately.",
        "Figure 3b: The probability histogram shows risk distribution and explains fixed-threshold skew (lines at 0.05 and 0.20).",
        "Table 2: Median imputation applied. Missing shown as counts and percentages.",
        "Tables 3a–3b: Risk stratification via fixed clinical thresholds and quantile tertiles.",
        f"Figure 4: Calibration (10 uniform bins). Brier={brier:.3f}. Calibration slope={cal_slope:.2f}, intercept={cal_intercept:.2f}.",
        "Figure 5: Decision curve analysis focused on 5–30% thresholds. Compare model vs 'Treat all' and 'Treat none'.",
        "Figure 6: Boxplots show distribution differences; medians/IQRs highlight central tendency and dispersion.",
        "Constant ultrasound flags with no variance are excluded from modeling."
    ]
    y = 0.95
    ax_notes.set_title("Methodological Notes", fontsize=12, pad=16)
    for i, note in enumerate(notes, start=1):
        wrapped = textwrap.fill(note, width=98)
        ax_notes.text(0.0, y, f"{i}. {wrapped}", ha="left", va="top", fontsize=9)
        y -= 0.08
    pdf.savefig(fig_notes); plt.close(fig_notes)

    # Final page 2: Reproducibility metadata
    fig_meta = plt.figure(figsize=(8.5, 11))
    ax_meta = fig_meta.add_subplot(111); ax_meta.axis("off")
    meta = [
        f"Dataset size: n={n_total}, prevalence (malignant)={prevalence:.3f}",
        f"Features used (non-constant): {', '.join(features_model)}",
        "Pipeline: median imputation + standardization + logistic regression",
        "CV: 5-fold stratified, random_state=42",
        f"Software: Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"matplotlib {mpl.__version__}, pandas {pd.__version__}",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    y = 0.95
    ax_meta.set_title("Reproducibility metadata", fontsize=12, pad=16)
    for m in meta:
        wrapped = textwrap.fill(m, width=98)
        ax_meta.text(0.0, y, f"- {wrapped}", ha="left", va="top", fontsize=9)
        y -= 0.06
    pdf.savefig(fig_meta); plt.close(fig_meta)

print(f"✅ PDF report generated: {report_path}")
data.to_csv(export_path, index=False)
print(f"✅ Unified dataset saved to: {export_path}")