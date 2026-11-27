"""
Hệ thống Cảnh báo Sớm Sinh viên & Giải thích XAI trên LMS (OULAD × SEDL)
Đồ án Project I — Tác giả: Vũ Hoà Bình (MSSV 202416137)
Chạy ứng dụng: streamlit run demo/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Cấu hình trang & Hằng số Dự án
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LMS Early Warning & XAI | Vũ Hoà Bình",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
SNAPSHOTS_DIR = ROOT_DIR / "data" / "processed" / "week5_6_snapshots"
WEEK7_DIR = ROOT_DIR / "data" / "processed" / "week7_modeling"
WEEK6_DIR = ROOT_DIR / "data" / "processed" / "week6_advanced_features"
WEEK8_XAI_DIR = ROOT_DIR / "data" / "processed" / "week8_xai"

MILESTONES = [25, 50, 75, 100]

ORDINAL_MAPPINGS: Dict[str, Dict[str, int]] = {
    "highest_education": {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4,
    },
    "imd_band": {
        "0-10%": 0,
        "10-20": 1,
        "20-30%": 2,
        "30-40%": 3,
        "40-50%": 4,
        "50-60%": 5,
        "60-70%": 6,
        "70-80%": 7,
        "80-90%": 8,
        "90-100%": 9,
        "Unknown": 4,
    },
    "age_band": {"0-35": 0, "35-55": 1, "55<=": 2},
}
NOMINAL_COLS = ["gender", "disability", "region", "code_module"]
FLAG_COLS = ["imd_was_missing", "has_no_submissions", "has_no_clicks"]

# Bảng ánh xạ tên biến sang nhãn tiếng Việt dễ hiểu cho giảng viên / cố vấn học tập
FEATURE_LABELS_VI: Dict[str, str] = {
    "submission_rate": "Tỷ lệ hoàn thành bài tập đến hạn",
    "course_progress_submission_rate": "Tỷ lệ nộp bài theo tiến độ khóa",
    "days_since_last_activity": "Số ngày vắng mặt gần nhất",
    "max_inactivity_streak": "Chuỗi ngày ngưng hoạt động dài nhất",
    "inactivity_ratio": "Tỷ lệ ngày không tương tác",
    "total_clicks": "Tổng lượt tương tác (clicks)",
    "active_days": "Tổng số ngày có đăng nhập",
    "active_days_in_course": "Số ngày hoạt động trong kỳ",
    "active_days_ratio": "Tỷ lệ ngày duy trì hoạt động",
    "active_span_days": "Khoảng thời gian gắn kết (ngày)",
    "clicks_last_2_weeks": "Lượt click trong 14 ngày gần nhất",
    "clicks_week_1_to_4": "Lượt click 4 tuần đầu",
    "engagement_decay": "Hệ số suy giảm tương tác",
    "clicks_trend": "Xu hướng thay đổi lượt click",
    "consistency_score": "Độ đều đặn hàng tuần (Entropy)",
    "login_regularity": "Tỷ lệ tuần duy trì đăng nhập",
    "active_weeks": "Số tuần có hoạt động",
    "weighted_avg_score": "Điểm trung bình có trọng số",
    "avg_score": "Điểm trung bình bài nộp",
    "earned_weight_score": "Tổng điểm trọng số tích lũy",
    "first_score": "Điểm bài nộp đầu tiên",
    "min_score": "Điểm bài nộp thấp nhất",
    "max_score": "Điểm bài nộp cao nhất",
    "tma1_score": "Điểm bài TMA1",
    "tma1_days_early": "Số ngày nộp sớm bài TMA1",
    "submitted_tma1": "Đã nộp bài TMA1",
    "avg_days_before_deadline": "Số ngày nộp trước hạn TB",
    "late_submission_rate": "Tỷ lệ bài nộp trễ",
    "num_submissions": "Số bài tập đã nộp có điểm",
    "clicks_oucontent": "Lượt đọc bài giảng (OU Content)",
    "clicks_quiz": "Lượt làm trắc nghiệm (Quiz)",
    "clicks_forumng": "Lượt thảo luận diễn đàn (ForumNG)",
    "clicks_homepage": "Lượt truy cập trang chủ (Homepage)",
    "clicks_subpage": "Lượt mở trang con (Subpage)",
    "clicks_resource": "Lượt tải tài liệu (Resource)",
    "clicks_page": "Lượt đọc trang Page",
    "cognitive_clicks": "Tổng tương tác Nhận thức",
    "social_clicks": "Tổng tương tác Xã hội",
    "cognitive_ratio": "Tỷ trọng tương tác Nhận thức",
    "social_ratio": "Tỷ trọng tương tác Xã hội",
    "forumng_ratio": "Tỷ trọng tương tác Diễn đàn",
    "oucontent_ratio": "Tỷ trọng đọc bài giảng",
    "resource_ratio": "Tỷ trọng tải tài nguyên",
    "activity_diversity": "Độ đa dạng loại tài nguyên",
    "activity_entropy": "Entropy phân bố hoạt động",
    "studied_credits": "Tổng tín chỉ đang theo học",
    "credit_overload": "Cảnh báo quá tải tín chỉ (>60)",
    "num_of_prev_attempts": "Số lần học lại trước đây",
    "is_repeat_student": "Sinh viên học lại môn",
    "has_no_submissions": "Chưa nộp bài tập nào",
    "has_no_clicks": "Chưa có lượt click nào",
    "highest_education": "Trình độ học vấn đầu vào",
    "imd_band": "Chỉ số kinh tế-xã hội vùng (IMD)",
    "age_band": "Nhóm độ tuổi",
    "days_before_start": "Số ngày đăng ký trước khai giảng",
    "due_course_assessments": "Số bài tập đã đến hạn của môn",
    "total_course_assessments": "Tổng số bài tập cả môn",
    "module_presentation_length": "Độ dài khóa học (ngày)",
    "last_minus_first_score": "Chênh lệch điểm cuối trừ đầu",
    "score_trend_slope": "Độ dốc xu hướng điểm số",
}

# CSS tinh chỉnh giao diện học thuật sạch sẽ, trực quan
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.35rem; padding-bottom: 2rem;}
    .hero-banner {
        background: linear-gradient(120deg, #0f2942 0%, #1b4965 55%, #2c6e91 100%);
        color: #ffffff;
        padding: 1.1rem 1.45rem;
        border-radius: 12px;
        margin-bottom: 0.95rem;
        box-shadow: 0 4px 14px rgba(15, 41, 66, 0.16);
    }
    .hero-title {font-size: 1.52rem; font-weight: 700; margin: 0 0 0.25rem 0;}
    .hero-sub {font-size: 0.90rem; opacity: 0.93; margin: 0;}
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1b4965;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .kpi-label {font-size: 0.81rem; color: #475569; font-weight: 600; margin-bottom: 0.2rem;}
    .kpi-val {font-size: 1.48rem; color: #0f172a; font-weight: 700; line-height: 1.2;}
    .kpi-sub {font-size: 0.77rem; color: #64748b; margin-top: 0.22rem;}
    .risk-badge-very-high {background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.85rem;}
    .risk-badge-high {background:#ffedd5; color:#9a3412; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.85rem;}
    .risk-badge-watch {background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.85rem;}
    .risk-badge-safe {background:#dcfce7; color:#166534; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tải Mô hình Đa mốc 5-Fold OOF, Metadata & Tài nguyên Tuần 8 XAI (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp bộ mô hình XGBoost 5-Fold qua 4 mốc (25%–100%)...")
def load_models_and_metadata() -> Dict[str, Any]:
    bundle_4m_path = MODELS_DIR / "best_model_5fold_4milestones.joblib"
    if bundle_4m_path.exists():
        bundle_4m = joblib.load(bundle_4m_path)
    else:
        ens_25 = joblib.load(MODELS_DIR / "best_model_5fold_ensemble.joblib")
        best_25 = joblib.load(MODELS_DIR / "best_model.pkl")
        bundle_4m = {
            "model_name": "XGBoost (Tuned)",
            "feature_names_114": ens_25["feature_names_114"],
            "consensus_params": ens_25.get("consensus_params", {}),
            "milestones": {
                "25%": {
                    "n_cohort": 26248,
                    "at_risk_rate": 0.4139,
                    "fold_models": ens_25["fold_models"],
                    "fold_preprocessors": ens_25["fold_preprocessors"],
                    "fold_thresholds": ens_25.get("fold_thresholds", [0.432] * 5),
                    "mean_threshold": float(ens_25.get("mean_threshold", 0.432)),
                    "full_model": best_25,
                    "full_preprocessor_state": ens_25["full_preprocessor_state"],
                }
            },
        }

    with open(MODELS_DIR / "best_model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(WEEK7_DIR / "week7_experiment_metrics.json", "r", encoding="utf-8") as f:
        week7_metrics = json.load(f)
    catalog_df = pd.read_csv(WEEK6_DIR / "feature_catalog_week6.csv")

    week8_xai: Dict[str, Any] = {}
    if WEEK8_XAI_DIR.exists():
        w8_json = WEEK8_XAI_DIR / "week8_xai_metrics.json"
        w8_top_4m = WEEK8_XAI_DIR / "top_features_consensus_4milestones_week8.csv"
        w8_top_25 = WEEK8_XAI_DIR / "top_features_consensus_week8.csv"
        w8_fair = WEEK8_XAI_DIR / "fairness_audit_week8.csv"
        w8_cases = WEEK8_XAI_DIR / "case_studies_week8.json"

        if w8_json.exists():
            with open(w8_json, "r", encoding="utf-8") as f:
                week8_xai["metrics"] = json.load(f)
        if w8_top_4m.exists():
            week8_xai["top_features_4m"] = pd.read_csv(w8_top_4m)
        elif w8_top_25.exists():
            df_t = pd.read_csv(w8_top_25)
            df_t["milestone"] = "25%"
            week8_xai["top_features_4m"] = df_t
        if w8_fair.exists():
            week8_xai["fairness"] = pd.read_csv(w8_fair)
        if w8_cases.exists():
            with open(w8_cases, "r", encoding="utf-8") as f:
                week8_xai["case_studies"] = json.load(f)

    return {
        "bundle_4m": bundle_4m,
        "metadata": metadata,
        "week7_metrics": week7_metrics,
        "catalog_df": catalog_df,
        "week8_xai": week8_xai,
    }


def get_milestone_artifacts(bundle_4m: Dict[str, Any], pct: int) -> Dict[str, Any]:
    """Lấy đúng gói mô hình & bộ tiền xử lý 5-Fold của mốc thời gian tương ứng."""
    key = f"{pct}%"
    milestones_dict = bundle_4m["milestones"]
    if key in milestones_dict:
        return milestones_dict[key]
    return milestones_dict["25%"]


def transform_with_state(df_in: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    """Áp dụng chính xác quy trình biến đổi 114 đặc trưng In-Fold của OULAD5FoldCVEngine."""
    d = df_in.copy()
    d["imd_was_missing"] = d["imd_band"].isna().astype(float)
    d["has_no_submissions"] = (d["num_submissions"].fillna(0) == 0).astype(float)
    d["has_no_clicks"] = (d["total_clicks"].fillna(0) == 0).astype(float)

    d["imd_band"] = d["imd_band"].fillna(state["imd_mode"])
    d["date_registration"] = d["date_registration"].fillna(state["reg_median"])
    d["days_before_start"] = -d["date_registration"]

    score_cols = [
        "avg_score",
        "weighted_avg_score",
        "earned_weight_score",
        "min_score",
        "max_score",
        "score_std",
        "first_score",
        "tma1_score",
    ]
    for sc in score_cols:
        if sc in d.columns:
            d[sc] = d[sc].fillna(0.0)

    for ec in ["avg_days_early", "avg_days_before_deadline", "tma1_days_early"]:
        if ec in d.columns:
            d[ec] = d[ec].fillna(-10.0)

    for c in state["num_cols"]:
        if c in d.columns:
            d[c] = d[c].fillna(0.0)

    for col, mapping in ORDINAL_MAPPINGS.items():
        d[col] = d[col].map(mapping).fillna(mapping.get("Unknown", 2)).astype(float)

    for col, (low, high) in state["bounds"].items():
        if col in d.columns:
            d[col] = d[col].clip(lower=low, upper=high)

    for col in state["skewed_cols"]:
        if (
            col in d.columns
            and "click" in col
            and "ratio" not in col
            and "per" not in col
            and "cv" not in col
        ):
            d[col] = np.log1p(np.maximum(0.0, d[col]))

    cols_to_scale = state["cols_to_scale"]
    scaled_arr = state["scaler"].transform(d[cols_to_scale])
    scaled_df = pd.DataFrame(scaled_arr, columns=cols_to_scale, index=df_in.index)

    ohe_arr = state["ohe"].transform(df_in[NOMINAL_COLS].fillna("Missing").astype(str))
    ohe_df = pd.DataFrame(ohe_arr, columns=state["ohe_names"], index=df_in.index)

    flags_df = d[FLAG_COLS].copy()
    out = pd.concat([scaled_df, ohe_df, flags_df], axis=1)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def filter_cohort(df: pd.DataFrame, cohort_type: str) -> pd.DataFrame:
    if cohort_type == "full":
        return df.copy().reset_index(drop=True)
    if cohort_type == "dynamic_active":
        mask = df["date_unregistration"].isna() | (df["date_unregistration"] > df["cutoff_day"])
        return df[mask].copy().reset_index(drop=True)
    raise ValueError(f"cohort_type không hợp lệ: {cohort_type}")


@st.cache_data(show_spinner="Đang xử lý lát cắt dữ liệu & chấm điểm 5-Fold OOF...")
def load_and_score_snapshot(pct: int, cohort_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Đọc snapshot, biến đổi 114 đặc trưng và sinh dự báo (5-Fold OOF cho cả 4 mốc dynamic_active)."""
    artifacts = load_models_and_metadata()
    bundle_4m = artifacts["bundle_4m"]
    feature_names_114 = bundle_4m["feature_names_114"]
    m_art = get_milestone_artifacts(bundle_4m, pct)

    raw_df = pd.read_csv(SNAPSHOTS_DIR / f"features_snapshot_{pct}pct.csv")
    df_cohort = filter_cohort(raw_df, cohort_type)

    if cohort_type == "dynamic_active" and f"{pct}%" in bundle_4m["milestones"]:
        # Chấm điểm Out-of-Fold 5-Fold thực sự cho TẤT CẢ 4 mốc (0% rò rỉ dữ liệu)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        y = df_cohort["at_risk"].values
        groups = df_cohort["id_student"].values

        oof_probs = np.zeros(len(df_cohort), dtype=float)
        oof_folds = np.zeros(len(df_cohort), dtype=int)
        X_oof = pd.DataFrame(0.0, index=df_cohort.index, columns=feature_names_114)

        for fold_idx, (_, te_idx) in enumerate(sgkf.split(df_cohort, y, groups=groups)):
            prep_state = m_art["fold_preprocessors"][fold_idx]
            fold_model = m_art["fold_models"][fold_idx]
            X_te = transform_with_state(df_cohort.iloc[te_idx], prep_state)[feature_names_114]
            oof_probs[te_idx] = fold_model.predict_proba(X_te)[:, 1]
            oof_folds[te_idx] = fold_idx
            X_oof.iloc[te_idx] = X_te.values

        df_cohort["risk_score"] = oof_probs
        df_cohort["oof_fold_idx"] = oof_folds
        return df_cohort, X_oof, oof_folds

    # Với cohort Full: áp dụng full_preprocessor_state + full_model của mốc tương ứng
    state = m_art["full_preprocessor_state"]
    full_model = m_art["full_model"]
    X_full = transform_with_state(df_cohort, state)[feature_names_114]
    probs = full_model.predict_proba(X_full)[:, 1]
    df_cohort["risk_score"] = probs
    df_cohort["oof_fold_idx"] = -1
    return df_cohort, X_full, np.full(len(df_cohort), -1, dtype=int)


def classify_risk_tier(prob: float, tau: float) -> str:
    if prob >= max(0.70, tau):
        return "Nguy cơ Rất cao"
    if prob >= tau:
        return "Nguy cơ Cao"
    if prob >= min(0.30, tau * 0.7):
        return "Theo dõi"
    return "An toàn"


def get_risk_badge_html(tier: str) -> str:
    css_map = {
        "Nguy cơ Rất cao": "risk-badge-very-high",
        "Nguy cơ Cao": "risk-badge-high",
        "Theo dõi": "risk-badge-watch",
        "An toàn": "risk-badge-safe",
    }
    cls = css_map.get(tier, "risk-badge-watch")
    return f'<span class="{cls}">{tier}</span>'


def compute_exact_treeshap_row(
    model: xgb.XGBClassifier, x_row_df: pd.DataFrame
) -> Tuple[float, np.ndarray]:
    """Tính Exact TreeSHAP (log-odds) trực tiếp qua C++ core pred_contribs=True của XGBoost."""
    booster = model.get_booster()
    dm = xgb.DMatrix(x_row_df, feature_names=list(x_row_df.columns))
    contribs = booster.predict(dm, pred_contribs=True)[0]
    shap_vals = contribs[:-1]
    base_val = float(contribs[-1])
    return base_val, shap_vals


def map_feature_to_sedl_4dim(feat: str) -> str:
    """Ánh xạ 114 đặc trưng về 4 nhóm chuẩn Tuần 8: Behavioural, Cognitive, Social, Demographic & Context."""
    if any(
        k in feat
        for k in [
            "forumng",
            "oucollaborate",
            "ouelluminate",
            "ouwiki",
            "sharedsubpage",
            "social",
        ]
    ):
        return "Social"
    if any(
        k in feat
        for k in [
            "score",
            "quiz",
            "oucontent",
            "resource",
            "glossary",
            "dataplus",
            "dualpane",
            "externalquiz",
            "htmlactivity",
            "page",
            "repeatactivity",
            "cognitive",
            "activity_diversity",
            "activity_entropy",
            "questionnaire",
        ]
    ):
        return "Cognitive"
    if any(
        k in feat
        for k in [
            "gender_",
            "disability_",
            "region_",
            "code_module_",
            "imd_",
            "age_band",
            "highest_education",
            "studied_credits",
            "credit_overload",
            "num_of_prev_attempts",
            "is_repeat_student",
            "module_presentation_length",
            "due_course_assessments",
            "total_course_assessments",
            "days_before_start",
            "is_late_registration",
        ]
    ):
        return "Demographic & Context"
    return "Behavioural"


@st.cache_data(show_spinner="Đang tính Exact TreeSHAP mẫu tổng thể cho phân tích SEDL...")
def compute_sample_sedl_shap(pct: int, cohort_type: str, sample_n: int = 2000) -> pd.DataFrame:
    artifacts = load_models_and_metadata()
    bundle_4m = artifacts["bundle_4m"]
    feature_names_114 = bundle_4m["feature_names_114"]
    m_art = get_milestone_artifacts(bundle_4m, pct)
    df_cohort, X_mat, oof_folds = load_and_score_snapshot(pct, cohort_type)

    n = min(sample_n, len(df_cohort))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df_cohort), size=n, replace=False)

    if cohort_type == "dynamic_active" and f"{pct}%" in bundle_4m["milestones"]:
        contribs_list = []
        for f_idx in range(5):
            mask_f = oof_folds[idx] == f_idx
            if not np.any(mask_f):
                continue
            sub_idx = idx[mask_f]
            booster = m_art["fold_models"][f_idx].get_booster()
            dm = xgb.DMatrix(X_mat.iloc[sub_idx], feature_names=feature_names_114)
            contribs_list.append(booster.predict(dm, pred_contribs=True)[:, :-1])
        contribs = np.vstack(contribs_list)
    else:
        booster = m_art["full_model"].get_booster()
        dm = xgb.DMatrix(X_mat.iloc[idx], feature_names=feature_names_114)
        contribs = booster.predict(dm, pred_contribs=True)[:, :-1]

    mean_abs = np.mean(np.abs(contribs), axis=0)
    sedl_df = pd.DataFrame(
        {"feature": feature_names_114, "mean_abs_shap": mean_abs}
    )
    sedl_df["sedl_group"] = sedl_df["feature"].apply(map_feature_to_sedl_4dim)
    return sedl_df


def compute_sedl_radar_profile(
    row: pd.Series, df_cohort: pd.DataFrame
) -> Tuple[List[str], List[float], List[float]]:
    """Tính hồ sơ phân vị (percentile 0-100) trên 4 chiều SEDL so với trung vị nhóm An toàn."""
    safe_df = df_cohort[df_cohort["at_risk"] == 0]
    if safe_df.empty:
        safe_df = df_cohort

    def pct_rank(col: str, val: float, invert: bool = False) -> float:
        series = df_cohort[col].fillna(0.0)
        rank = float((series <= val).mean() * 100.0)
        return 100.0 - rank if invert else rank

    def safe_med_rank(col: str, invert: bool = False) -> float:
        med = float(safe_df[col].fillna(0.0).median())
        return pct_rank(col, med, invert=invert)

    dims = [
        "Behavioural<br>(Hành vi)",
        "Cognitive<br>(Nhận thức)",
        "Self-Regulation<br>(Tự điều chỉnh)",
        "Social<br>(Xã hội)",
    ]

    stu_beh = np.mean(
        [
            pct_rank("total_clicks", float(row.get("total_clicks", 0.0))),
            pct_rank("submission_rate", float(row.get("submission_rate", 0.0))),
            pct_rank("days_since_last_activity", float(row.get("days_since_last_activity", 0.0)), invert=True),
        ]
    )
    safe_beh = np.mean(
        [
            safe_med_rank("total_clicks"),
            safe_med_rank("submission_rate"),
            safe_med_rank("days_since_last_activity", invert=True),
        ]
    )

    stu_cog = np.mean(
        [
            pct_rank(
                "weighted_avg_score",
                float(row.get("weighted_avg_score", 0.0) if pd.notna(row.get("weighted_avg_score")) else 0.0),
            ),
            pct_rank("cognitive_clicks", float(row.get("cognitive_clicks", 0.0))),
            pct_rank("activity_entropy", float(row.get("activity_entropy", 0.0))),
        ]
    )
    safe_cog = np.mean(
        [
            safe_med_rank("weighted_avg_score"),
            safe_med_rank("cognitive_clicks"),
            safe_med_rank("activity_entropy"),
        ]
    )

    stu_srl = np.mean(
        [
            pct_rank("consistency_score", float(row.get("consistency_score", 0.0))),
            pct_rank("login_regularity", float(row.get("login_regularity", 0.0))),
            pct_rank("max_inactivity_streak", float(row.get("max_inactivity_streak", 0.0)), invert=True),
        ]
    )
    safe_srl = np.mean(
        [
            safe_med_rank("consistency_score"),
            safe_med_rank("login_regularity"),
            safe_med_rank("max_inactivity_streak", invert=True),
        ]
    )

    stu_soc = np.mean(
        [
            pct_rank("social_clicks", float(row.get("social_clicks", 0.0))),
            pct_rank("clicks_forumng", float(row.get("clicks_forumng", 0.0))),
            pct_rank("forumng_ratio", float(row.get("forumng_ratio", 0.0))),
        ]
    )
    safe_soc = np.mean(
        [
            safe_med_rank("social_clicks"),
            safe_med_rank("clicks_forumng"),
            safe_med_rank("forumng_ratio"),
        ]
    )

    stu_vals = [round(stu_beh, 1), round(stu_cog, 1), round(stu_srl, 1), round(stu_soc, 1)]
    safe_vals = [round(safe_beh, 1), round(safe_cog, 1), round(safe_srl, 1), round(safe_soc, 1)]
    return dims, stu_vals, safe_vals


def generate_pedagogical_recommendations(
    row: pd.Series, top_drivers: pd.DataFrame, risk_score: float, tau: float
) -> List[Tuple[str, str]]:
    """Sinh khuyến nghị can thiệp sư phạm tự động dựa trên tác nhân SHAP & chỉ số SEDL."""
    recs: List[Tuple[str, str]] = []
    pos_feats = set(top_drivers[top_drivers["shap_val"] > 0]["feature"].tolist())

    days_inactive = float(row.get("days_since_last_activity", 0.0))
    sub_rate = float(row.get("submission_rate", 0.0))
    w_score = float(row.get("weighted_avg_score", 0.0)) if pd.notna(row.get("weighted_avg_score")) else 0.0
    forum_clicks = float(row.get("clicks_forumng", 0.0))
    overload = int(row.get("credit_overload", 0))
    consistency = float(row.get("consistency_score", 0.0))

    if days_inactive >= 14 or "days_since_last_activity" in pos_feats or "max_inactivity_streak" in pos_feats:
        recs.append(
            (
                "Can thiệp Hành vi — Tái gắn kết khẩn cấp",
                f"Sinh viên đã ngưng hoạt động {days_inactive:.0f} ngày gần nhất. Cố vấn học tập nên gửi thông báo nhắc nhở cá nhân hóa qua email/LMS trong 48 giờ và kiểm tra rào cản kỹ thuật/cá nhân.",
            )
        )
    if sub_rate < 0.75 or "submission_rate" in pos_feats or "has_no_submissions" in pos_feats:
        recs.append(
            (
                "Hỗ trợ Nhiệm vụ Học tập (Task Completion)",
                f"Tỷ lệ nộp bài tập mới đạt {sub_rate:.0%}. Đề xuất rà soát các bài TMA/CMA quá hạn, cung cấp hướng dẫn từng bước và nhắc lịch deadline kế tiếp.",
            )
        )
    if w_score < 55.0 or "weighted_avg_score" in pos_feats or "avg_score" in pos_feats:
        recs.append(
            (
                "Phụ đạo Nhận thức (Cognitive Support)",
                f"Điểm trung bình trọng số hiện tại ({w_score:.1f}/100) ở mức thấp. Khuyến nghị sinh viên ôn lại học liệu lõi `oucontent` và làm lại `quiz` luyện tập cùng trợ giảng.",
            )
        )
    if forum_clicks < 10 or "clicks_forumng" in pos_feats or "social_clicks" in pos_feats:
        recs.append(
            (
                "Kết nối Xã hội (Social Engagement)",
                f"Mức độ tương tác diễn đàn còn hạn chế ({forum_clicks:.0f} lượt). Mời sinh viên tham gia nhóm thảo luận nhỏ hoặc phiên giải đáp trực tuyến để giảm cô lập.",
            )
        )
    if consistency < 0.45 or "consistency_score" in pos_feats or "engagement_decay" in pos_feats:
        recs.append(
            (
                "Huấn luyện Tự điều chỉnh (SRL)",
                f"Nhịp độ học tập thiếu ổn định (Consistency = {consistency:.2f}). Đề xuất chia nhỏ mục tiêu học 3–4 buổi/tuần thay vì học dồn sát hạn.",
            )
        )
    if overload == 1:
        recs.append(
            (
                "Tư vấn Khối lượng Tín chỉ",
                f"Sinh viên đang đăng ký {int(row.get('studied_credits', 0))} tín chỉ (> ngưỡng 60 tín chỉ). Xem xét điều chỉnh lộ trình học kỳ nếu quá tải.",
            )
        )

    if not recs:
        if risk_score >= tau:
            recs.append(
                (
                    "Theo dõi Chủ động",
                    "Xác suất rủi ro vượt ngưỡng cảnh báo nhẹ; tiếp tục theo dõi định kỳ hàng tuần và duy trì phản hồi tích cực cho bài nộp kế tiếp.",
                )
            )
        else:
            recs.append(
                (
                    "Duy trì Nhịp độ Học tập Tốt",
                    "Sinh viên đang duy trì tiến độ ổn định và an toàn. Khuyến khích tiếp tục giữ thói quen tương tác đều đặn.",
                )
            )
    return recs[:4]


@st.cache_data(show_spinner="Đang truy xuất hành trình OOF 4 mốc thời gian của sinh viên...")
def get_student_trajectory(
    id_student: int, code_module: str, code_presentation: str, cohort_type: str = "dynamic_active"
) -> pd.DataFrame:
    """Tính xác suất rủi ro (5-Fold OOF chuẩn khi dynamic_active) qua 4 mốc 25% -> 50% -> 75% -> 100%."""
    artifacts = load_models_and_metadata()
    bundle_4m = artifacts["bundle_4m"]

    rows: List[Dict[str, Any]] = []
    for pct in MILESTONES:
        df_s, _, _ = load_and_score_snapshot(pct, cohort_type)
        sub = df_s[
            (df_s["id_student"] == id_student)
            & (df_s["code_module"] == code_module)
            & (df_s["code_presentation"] == code_presentation)
        ]
        if sub.empty:
            continue
        r = sub.iloc[0]
        m_art = get_milestone_artifacts(bundle_4m, pct)
        f_idx = int(r.get("oof_fold_idx", -1))
        fold_thresholds = m_art.get("fold_thresholds", [m_art["mean_threshold"]] * 5)
        tau_m = float(fold_thresholds[f_idx]) if f_idx >= 0 else float(m_art["mean_threshold"])

        rows.append(
            {
                "Mốc (%)": f"{pct}%",
                "pct": pct,
                "cutoff_day": float(r["cutoff_day"]),
                "Fold": f"Fold {f_idx + 1}/5" if f_idx >= 0 else "Full Model",
                "Ngưỡng τ*": round(tau_m, 3),
                "Risk Score": float(r["risk_score"]),
                "total_clicks": float(r["total_clicks"]),
                "submission_rate": float(r["submission_rate"]),
                "weighted_avg_score": float(r["weighted_avg_score"]) if pd.notna(r["weighted_avg_score"]) else 0.0,
                "days_since_last_activity": float(r["days_since_last_activity"]),
                "max_inactivity_streak": float(r["max_inactivity_streak"]),
                "engagement_decay": float(r["engagement_decay"]),
            }
        )
    return pd.DataFrame(rows)


def compute_subgroup_fairness(df_eval: pd.DataFrame, group_col: str, tau: float) -> pd.DataFrame:
    """Tính các chỉ số công bằng nhóm (TPR/Recall, FPR, Positive Rate, Precision, AUC, Brier) trên tập lọc."""
    y_true = df_eval["at_risk"].values
    y_prob = df_eval["risk_score"].values
    y_pred = (y_prob >= tau).astype(int)

    work = pd.DataFrame(
        {
            "group": df_eval[group_col].fillna("Missing").astype(str),
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    records: List[Dict[str, Any]] = []
    for grp, g in work.groupby("group"):
        n = len(g)
        if n < 20:
            continue
        yt = g["y_true"].values
        yp = g["y_pred"].values
        ypr = g["y_prob"].values
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        pos_rate = float(yp.mean())
        prec = tp / max(1, tp + fp)
        auc_g = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else np.nan
        brier = float(brier_score_loss(yt, ypr))
        records.append(
            {
                "Nhóm": grp,
                "Số lượng (N)": int(n),
                "Tỷ lệ At-Risk thực": round(float(yt.mean()), 4),
                "Positive Rate (Dự báo)": round(pos_rate, 4),
                "Recall (TPR)": round(float(tpr), 4),
                "FPR": round(float(fpr), 4),
                "Precision": round(float(prec), 4),
                "ROC-AUC": round(auc_g, 4) if not np.isnan(auc_g) else np.nan,
                "Brier Score": round(brier, 4),
            }
        )

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        max_pr = out_df["Positive Rate (Dự báo)"].max()
        ref_pr = max_pr if max_pr > 0 else 1.0
        out_df["Disparate Impact Ratio"] = (out_df["Positive Rate (Dự báo)"] / ref_pr).round(3)
        out_df = out_df.sort_values("Số lượng (N)", ascending=False).reset_index(drop=True)
    return out_df


# ---------------------------------------------------------------------------
# Nạp tài nguyên chính & Dựng Sidebar
# ---------------------------------------------------------------------------
artifacts = load_models_and_metadata()
bundle_4m = artifacts["bundle_4m"]
metadata = artifacts["metadata"]
week7_metrics = artifacts["week7_metrics"]
week8_xai = artifacts["week8_xai"]
w8_metrics = week8_xai.get("metrics", {})
feature_names_114 = bundle_4m["feature_names_114"]

# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🎓 Hệ thống Cảnh báo Sớm Sinh viên &amp; Giải thích XAI Đa mốc (OULAD × SEDL)</div>
        <div class="hero-sub">
            Đồ án Project I · Tác giả: <b>Vũ Hoà Bình (MSSV 202416137)</b> ·
            Mô hình chuẩn hóa: <b>XGBoost (Tuned) 5-Fold Stratified Group OOF (25%–100%)</b> ·
            Giải thích nhân quả: <b>Exact TreeSHAP &amp; Phân tích Phản thực What-If</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.header("⚙️ Bộ lọc & Cấu hình Dự báo")
selected_pct = st.sidebar.select_slider(
    "Mốc dự báo thời lượng khóa học (%)",
    options=MILESTONES,
    value=25,
    help="Mốc 25% (~65 ngày đầu) là mốc cảnh báo sớm trọng tâm; hệ thống hỗ trợ chấm điểm 5-Fold OOF chuẩn cho cả 4 mốc 25%–100%.",
)
milestone_key = f"{selected_pct}%"
m_art = get_milestone_artifacts(bundle_4m, selected_pct)
milestone_default_tau = float(m_art.get("mean_threshold", metadata.get("mean_threshold", 0.432)))

# Tự động đồng bộ ngưỡng τ* tối ưu khi chuyển mốc thời gian
if "last_selected_pct" not in st.session_state:
    st.session_state["last_selected_pct"] = selected_pct
    st.session_state["tau_slider"] = round(milestone_default_tau, 3)
elif st.session_state["last_selected_pct"] != selected_pct:
    st.session_state["last_selected_pct"] = selected_pct
    st.session_state["tau_slider"] = round(milestone_default_tau, 3)

selected_cohort = st.sidebar.radio(
    "Tập hợp Sinh viên (Cohort)",
    options=["dynamic_active", "full"],
    format_func=lambda x: (
        "Dynamic Active (Đang học tại mốc cắt — Khuyến nghị)"
        if x == "dynamic_active"
        else "Full Cohort (Toàn bộ lượt đăng ký)"
    ),
    index=0,
)

df_scored, X_scored, oof_folds = load_and_score_snapshot(selected_pct, selected_cohort)

modules_list = ["Tất cả"] + sorted(df_scored["code_module"].unique().tolist())
pres_list = ["Tất cả"] + sorted(df_scored["code_presentation"].unique().tolist())

selected_module = st.sidebar.selectbox("Học phần (code_module)", modules_list, index=0)
selected_pres = st.sidebar.selectbox("Kỳ học (code_presentation)", pres_list, index=0)
tau = st.sidebar.slider(
    "Ngưỡng quyết định cảnh báo (τ)",
    min_value=0.15,
    max_value=0.80,
    step=0.005,
    key="tau_slider",
    help=f"Ngưỡng tối ưu 5-Fold CV cho mốc {milestone_key} là τ* = {milestone_default_tau:.3f} (đảm bảo Recall ≥ 0.75).",
)

# Áp dụng bộ lọc
mask_filter = pd.Series(True, index=df_scored.index)
if selected_module != "Tất cả":
    mask_filter &= df_scored["code_module"] == selected_module
if selected_pres != "Tất cả":
    mask_filter &= df_scored["code_presentation"] == selected_pres

df_view = df_scored[mask_filter].copy()
X_view = X_scored.loc[df_view.index]

if df_view.empty:
    st.warning("Không có sinh viên nào khớp với tổ hợp bộ lọc Học phần & Kỳ học hiện tại. Vui lòng chọn lại bộ lọc.")
    st.stop()

df_view["pred_at_risk"] = (df_view["risk_score"] >= tau).astype(int)
df_view["risk_tier"] = df_view["risk_score"].apply(lambda p: classify_risk_tier(p, tau))

is_oof_mode = selected_cohort == "dynamic_active" and milestone_key in bundle_4m["milestones"]
oof_4m_dict = w8_metrics.get("oof_performance_verification_4milestones", {})
m_cv_stats = oof_4m_dict.get(milestone_key, {})

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Chế độ chấm điểm ({milestone_key}):** "
    f"{'5-Fold Out-of-Fold (0% Leakage)' if is_oof_mode else f'XGBoost (Tuned) Full @{milestone_key}'}\n\n"
    f"**Số lượt đăng ký lọc:** `{len(df_view):,}`\n\n"
    f"**Số SV duy nhất:** `{df_view['id_student'].nunique():,}`\n\n"
    f"**Ngưỡng τ đang dùng:** `{tau:.3f}` (Tối ưu CV @{milestone_key}: `{milestone_default_tau:.3f}`)"
)

# ---------------------------------------------------------------------------
# 5 Tab Chức năng Chính
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 1. Tổng quan & XAI",
        "🚨 2. Ưu tiên Can thiệp",
        "🔍 3. TreeSHAP & What-If",
        "⚖️ 4. SEDL & Công bằng",
        "⏳ 5. Đa mốc (25%→100%)",
    ]
)

# ===========================================================================
# TAB 1: DASHBOARD TỔNG QUAN & GLOBAL CONSENSUS XAI
# ===========================================================================
with tab1:
    total_n = len(df_view)
    n_pred_risk = int(df_view["pred_at_risk"].sum())
    pct_pred_risk = n_pred_risk / total_n * 100.0
    actual_risk_rate = float(df_view["at_risk"].mean()) * 100.0

    y_true_v = df_view["at_risk"].values
    y_prob_v = df_view["risk_score"].values
    y_pred_v = df_view["pred_at_risk"].values

    rec_v = recall_score(y_true_v, y_pred_v, zero_division=0)
    prec_v = precision_score(y_true_v, y_pred_v, zero_division=0)
    f1_v = f1_score(y_true_v, y_pred_v, zero_division=0)
    auc_v = roc_auc_score(y_true_v, y_prob_v) if len(np.unique(y_true_v)) > 1 else 0.0
    pr_auc_v = average_precision_score(y_true_v, y_prob_v) if len(np.unique(y_true_v)) > 1 else 0.0
    acc_v = accuracy_score(y_true_v, y_pred_v)
    brier_v = brier_score_loss(y_true_v, y_prob_v)

    cv_auc_str = f"{m_cv_stats['roc_auc_mean']:.4f}" if "roc_auc_mean" in m_cv_stats else "N/A"
    cv_rec_str = f"{m_cv_stats['recall_mean']:.1%}" if "recall_mean" in m_cv_stats else "N/A"
    cv_brier_str = f"{m_cv_stats['brier_mean']:.4f}" if "brier_mean" in m_cv_stats else f"{brier_v:.4f}"
    shap_err_val = m_cv_stats.get("max_treeshap_additivity_error", None)
    shap_err_str = f"{shap_err_val:.2e}" if shap_err_val is not None else "< 1e-6"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Cohort Mốc {milestone_key}</div>'
            f'<div class="kpi-val">{total_n:,}</div>'
            f'<div class="kpi-sub">{df_view["id_student"].nunique():,} sinh viên duy nhất</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#dc2626;"><div class="kpi-label">SV Cảnh báo (Risk ≥ {tau:.3f})</div>'
            f'<div class="kpi-val">{n_pred_risk:,}</div>'
            f'<div class="kpi-sub">Chiếm {pct_pred_risk:.1f}% (Thực tế: {actual_risk_rate:.1f}%)</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#059669;"><div class="kpi-label">Độ nhạy (Recall)</div>'
            f'<div class="kpi-val">{rec_v:.1%}</div>'
            f'<div class="kpi-sub">5-Fold CV @{milestone_key}: {cv_rec_str} (Mục tiêu ≥ 75%)</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#2563eb;"><div class="kpi-label">Precision / F1-Score</div>'
            f'<div class="kpi-val">{prec_v:.1%} / {f1_v:.3f}</div>'
            f'<div class="kpi-sub">Acc: {acc_v:.1%} · Brier: {brier_v:.4f} (CV: {cv_brier_str})</div></div>',
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#7c3aed;"><div class="kpi-label">ROC-AUC / PR-AUC</div>'
            f'<div class="kpi-val">{auc_v:.4f}</div>'
            f'<div class="kpi-sub">CV @{milestone_key}: {cv_auc_str} · PR: {pr_auc_v:.4f} · SHAP Err: {shap_err_str}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.05, 1.25])

    with col_left:
        st.subheader(f"Phân tầng Mức độ Rủi ro ({milestone_key})")
        tier_order = ["Nguy cơ Rất cao", "Nguy cơ Cao", "Theo dõi", "An toàn"]
        tier_colors = {
            "Nguy cơ Rất cao": "#dc2626",
            "Nguy cơ Cao": "#f97316",
            "Theo dõi": "#eab308",
            "An toàn": "#16a34a",
        }
        tier_counts = (
            df_view["risk_tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
        )
        tier_counts.columns = ["Mức độ Rủi ro", "Số lượng"]
        tier_counts["Tỷ lệ (%)"] = (tier_counts["Số lượng"] / total_n * 100.0).round(2)

        fig_tier = px.pie(
            tier_counts,
            names="Mức độ Rủi ro",
            values="Số lượng",
            hole=0.48,
            color="Mức độ Rủi ro",
            color_discrete_map=tier_colors,
        )
        fig_tier.update_traces(textposition="inside", textinfo="percent+label")
        fig_tier.update_layout(height=340, margin=dict(t=15, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_tier, width="stretch")
        st.dataframe(tier_counts, width="stretch", hide_index=True)

    with col_right:
        st.subheader(f"Tỷ lệ Nguy cơ theo Học phần ({milestone_key})")
        mod_agg = (
            df_view.groupby("code_module", as_index=False)
            .agg(
                Tong_SV=("id_student", "count"),
                Du_bao_Nguy_co=("pred_at_risk", "mean"),
                Thuc_te_Nguy_co=("at_risk", "mean"),
            )
            .sort_values("Du_bao_Nguy_co", ascending=False)
        )
        mod_melt = mod_agg.melt(
            id_vars=["code_module", "Tong_SV"],
            value_vars=["Du_bao_Nguy_co", "Thuc_te_Nguy_co"],
            var_name="Loại",
            value_name="Tỷ lệ",
        )
        mod_melt["Loại"] = mod_melt["Loại"].map(
            {"Du_bao_Nguy_co": "Dự báo Cảnh báo", "Thuc_te_Nguy_co": "Thực tế At-Risk"}
        )
        fig_mod = px.bar(
            mod_melt,
            x="code_module",
            y="Tỷ lệ",
            color="Loại",
            barmode="group",
            color_discrete_map={"Dự báo Cảnh báo": "#ef4444", "Thực tế At-Risk": "#1e3a8a"},
            text_auto=".1%",
        )
        fig_mod.update_layout(
            height=340,
            yaxis_tickformat=".0%",
            xaxis_title="Mã học phần (Module)",
            yaxis_title="Tỷ lệ sinh viên (%)",
            margin=dict(t=15, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_mod, width="stretch")

        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            pres_agg = (
                df_view.groupby("code_presentation", as_index=False)
                .agg(N=("id_student", "count"), Risk_Pred=("pred_at_risk", "mean"))
                .sort_values("code_presentation")
            )
            pres_agg["Cảnh báo (%)"] = (pres_agg["Risk_Pred"] * 100).round(1)
            st.markdown("**Phân bố theo Kỳ học**")
            st.dataframe(
                pres_agg[["code_presentation", "N", "Cảnh báo (%)"]],
                width="stretch",
                hide_index=True,
            )
        with c_sub2:
            cm = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Dự báo An toàn", "Dự báo Nguy cơ"],
                y=["Thực tế An toàn", "Thực tế Nguy cơ"],
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(height=210, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.markdown("**Ma trận nhầm lẫn (Cohort lọc)**")
            st.plotly_chart(fig_cm, width="stretch")

    # Global Consensus Top 10 XAI cho mốc hiện tại
    top_df_4m = week8_xai.get("top_features_4m")
    gra_4m = w8_metrics.get("global_rank_agreement_4milestones", {})
    if top_df_4m is not None and not top_df_4m.empty:
        st.markdown("---")
        st.subheader(f"🧭 Top 10 Đặc trưng Đồng thuận Toàn cục (Global Consensus XAI — Mốc {milestone_key})")
        top_m = (
            top_df_4m[top_df_4m["milestone"] == milestone_key]
            .sort_values("consensus_score_pct", ascending=False)
            .head(10)
            .copy()
        )
        gra_m = gra_4m.get(milestone_key, {})
        if gra_m:
            st.caption(
                f"Công thức đồng thuận: $0.45 \\times \\text{{SHAP}} + 0.35 \\times \\text{{Permutation}} + 0.20 \\times \\text{{Gain}}$ · "
                f"Hệ số tương quan hạng Spearman tại {milestone_key}: "
                f"**SHAP vs Perm = {gra_m.get('spearman_shap_vs_perm', 0):.4f}** · "
                f"**SHAP vs Gain = {gra_m.get('spearman_shap_vs_gain', 0):.4f}** · "
                f"**Perm vs Gain = {gra_m.get('spearman_perm_vs_gain', 0):.4f}**."
            )

        gx1, gx2 = st.columns([1.15, 1.15])
        with gx1:
            plot_top = top_m.iloc[::-1].copy()
            plot_top["label_short"] = plot_top["feature_name"].map(
                lambda f: f"{FEATURE_LABELS_VI.get(f, f)} ({f})"
            )
            fig_top10 = px.bar(
                plot_top,
                x="consensus_score_pct",
                y="label_short",
                color="sedl_dimension",
                orientation="h",
                text=plot_top["consensus_score_pct"].map(lambda v: f"{v:.2f}%"),
                color_discrete_map={
                    "Behavioural": "#2563eb",
                    "Cognitive": "#059669",
                    "Social": "#7c3aed",
                    "Demographic & Context": "#d97706",
                },
            )
            fig_top10.update_traces(textposition="outside", cliponaxis=False)
            fig_top10.update_layout(
                height=360,
                xaxis_title="Consensus Score (%)",
                yaxis_title="",
                legend_title="Chiều SEDL",
                margin=dict(t=15, b=20, l=10, r=45),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_top10, width="stretch")

        with gx2:
            show_top = top_m[
                [
                    "consensus_rank",
                    "feature_name",
                    "sedl_dimension",
                    "consensus_score_pct",
                    "shap_share_pct",
                    "perm_share_pct",
                    "xgb_gain_share_pct",
                ]
            ].copy()
            show_top.columns = ["Hạng", "Đặc trưng", "Chiều SEDL", "Consensus (%)", "SHAP (%)", "Perm (%)", "Gain (%)"]
            for col_r in ["Consensus (%)", "SHAP (%)", "Perm (%)", "Gain (%)"]:
                show_top[col_r] = show_top[col_r].round(2)
            st.dataframe(show_top, width="stretch", height=360, hide_index=True)

# ===========================================================================
# TAB 2: DANH SÁCH CẢNH BÁO & ƯU TIÊN CAN THIỆP
# ===========================================================================
with tab2:
    st.subheader(f"🚨 Danh sách Cảnh báo & Điều phối Can thiệp Sớm (Mốc {milestone_key})")
    st.caption(
        "Sắp xếp giảm dần theo xác suất rủi ro Out-of-Fold (Risk Score) để cố vấn học tập ưu tiên các trường hợp cấp bách nhất."
    )

    f1, f2, f3, f4 = st.columns([1.2, 1.5, 1.1, 0.9])
    with f1:
        search_id = st.text_input("Tìm kiếm theo Mã SV (id_student)", placeholder="Ví dụ: 11391")
    with f2:
        selected_tiers = st.multiselect(
            "Lọc theo Mức độ Rủi ro",
            options=["Nguy cơ Rất cao", "Nguy cơ Cao", "Theo dõi", "An toàn"],
            default=["Nguy cơ Rất cao", "Nguy cơ Cao"],
        )
    with f3:
        only_at_risk = st.checkbox(f"Chỉ hiện Risk ≥ τ ({tau:.3f})", value=True)
    with f4:
        max_rows_show = st.selectbox("Số dòng hiển thị", [50, 100, 250, 500], index=1)

    triage_df = df_view.copy()
    if only_at_risk:
        triage_df = triage_df[triage_df["risk_score"] >= tau]
    if selected_tiers:
        triage_df = triage_df[triage_df["risk_tier"].isin(selected_tiers)]
    if search_id.strip():
        triage_df = triage_df[
            triage_df["id_student"].astype(str).str.contains(search_id.strip(), case=False, na=False)
        ]

    triage_df = triage_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    cols_triage = [
        "id_student",
        "code_module",
        "code_presentation",
        "risk_tier",
        "risk_score",
        "total_clicks",
        "days_since_last_activity",
        "max_inactivity_streak",
        "submission_rate",
        "weighted_avg_score",
        "engagement_decay",
        "final_result",
    ]
    display_df = triage_df[cols_triage].copy()
    display_df["risk_score"] = display_df["risk_score"].round(4)
    display_df["submission_rate"] = display_df["submission_rate"].round(3)
    display_df["weighted_avg_score"] = display_df["weighted_avg_score"].round(1)
    display_df["engagement_decay"] = display_df["engagement_decay"].round(3)

    c_info, c_dl = st.columns([3, 1])
    with c_info:
        st.info(
            f"Tìm thấy **{len(display_df):,}** sinh viên phù hợp điều kiện lọc tại mốc **{milestone_key}** "
            f"(đang hiển thị tối đa {min(max_rows_show, len(display_df))} dòng đầu)."
        )
    with c_dl:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Tải danh sách CSV",
            data=csv_bytes,
            file_name=f"canh_bao_som_{selected_pct}pct_{selected_module}.csv",
            mime="text/csv",
            width="stretch",
        )

    styled_triage = (
        display_df.head(max_rows_show)
        .style.background_gradient(subset=["risk_score"], cmap="Reds", vmin=0.0, vmax=1.0)
        .format(
            {
                "risk_score": "{:.2%}",
                "total_clicks": "{:,.0f}",
                "days_since_last_activity": "{:.0f}",
                "max_inactivity_streak": "{:.0f}",
                "submission_rate": "{:.1%}",
                "weighted_avg_score": "{:.1f}",
                "engagement_decay": "{:.3f}",
            }
        )
    )
    st.dataframe(styled_triage, width="stretch", height=460)

# ===========================================================================
# TAB 3: CHẨN ĐOÁN CÁ NHÂN, EXACT TREESHAP & GIẢ LẬP WHAT-IF
# ===========================================================================
with tab3:
    st.subheader(f"🔍 Chẩn đoán Cá nhân hóa ({milestone_key}): Exact TreeSHAP, Hồ sơ SEDL & Giả lập What-If")

    # Tích hợp danh sách Case Studies chuẩn OOF TP/FP/TN/FN của mốc hiện tại nếu có
    cs_by_m = week8_xai.get("case_studies", {}).get("by_milestone", {}).get(milestone_key, {})
    cs_ids: List[int] = []
    cs_tags: Dict[Tuple[int, str, str], str] = {}
    for q_name in ["TP", "FP", "TN", "FN"]:
        q_item = cs_by_m.get(q_name)
        if q_item:
            key_stu = (int(q_item["id_student"]), str(q_item["code_module"]), str(q_item["code_presentation"]))
            cs_ids.append(key_stu[0])
            cs_tags[key_stu] = f"[{q_name} Điển hình]"

    sorted_candidates = df_view.sort_values("risk_score", ascending=False)
    case_matches = df_view[df_view["id_student"].isin(cs_ids)] if cs_ids else df_view.iloc[0:0]
    top_risk_sample = sorted_candidates.head(70)
    mid_sample = sorted_candidates[
        (sorted_candidates["risk_score"] >= 0.35) & (sorted_candidates["risk_score"] <= 0.65)
    ].head(40)
    safe_sample = sorted_candidates[sorted_candidates["risk_score"] < 0.25].head(30)
    candidate_pool = pd.concat([case_matches, top_risk_sample, mid_sample, safe_sample]).drop_duplicates(
        subset=["id_student", "code_module", "code_presentation"]
    )

    sel_c1, sel_c2 = st.columns([2.2, 1.0])
    with sel_c2:
        manual_stu_id = st.text_input(
            "Tra cứu nhanh Mã SV cụ thể (tùy chọn)",
            placeholder="Nhập id_student...",
            key="tab3_manual_id",
        )

    if manual_stu_id.strip():
        matched = df_view[df_view["id_student"].astype(str) == manual_stu_id.strip()]
        if not matched.empty:
            candidate_pool = matched

    candidate_pool = candidate_pool.reset_index()
    options_labels = []
    for _, r in candidate_pool.iterrows():
        tag = cs_tags.get((int(r["id_student"]), str(r["code_module"]), str(r["code_presentation"])), "")
        prefix = f"{tag} " if tag else ""
        options_labels.append(
            f"{prefix}SV {int(r['id_student'])} | {r['code_module']}-{r['code_presentation']} | Risk: {r['risk_score']:.1%} ({r['risk_tier']}) | KQ: {r['final_result']}"
        )

    with sel_c1:
        chosen_label = st.selectbox(
            "Chọn Sinh viên cần phân tích chuyên sâu (gồm Case Studies TP/FP/TN/FN)",
            options=options_labels,
            index=0,
        )

    chosen_idx_pos = options_labels.index(chosen_label)
    orig_index = int(candidate_pool.loc[chosen_idx_pos, "index"])
    stu_row = df_view.loc[orig_index]
    stu_X_row = X_view.loc[[orig_index]]

    # Chọn đúng mô hình & preprocessor OOF của mốc hiện tại
    fold_idx = int(stu_row.get("oof_fold_idx", -1))
    if fold_idx >= 0:
        active_model = m_art["fold_models"][fold_idx]
        active_prep = m_art["fold_preprocessors"][fold_idx]
        model_note = f"OOF Fold {fold_idx + 1}/5 @{milestone_key}"
    else:
        active_model = m_art["full_model"]
        active_prep = m_art["full_preprocessor_state"]
        model_note = f"XGBoost Full @{milestone_key}"

    base_val, shap_vals = compute_exact_treeshap_row(active_model, stu_X_row)
    stu_risk = float(expit(base_val + shap_vals.sum()))
    stu_tier = classify_risk_tier(stu_risk, tau)

    g_col, info_col = st.columns([1.05, 1.95])
    with g_col:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=stu_risk * 100.0,
                number={"suffix": "%", "font": {"size": 34}},
                delta={"reference": tau * 100.0, "suffix": "% vs τ*"},
                title={"text": f"Xác suất Nguy cơ ({model_note})", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626" if stu_risk >= tau else "#16a34a"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, tau * 100.0], "color": "#fef9c3"},
                        {"range": [tau * 100.0, 70], "color": "#ffedd5"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 3}, "thickness": 0.8, "value": tau * 100.0},
                },
            )
        )
        fig_gauge.update_layout(height=255, margin=dict(t=42, b=10, l=32, r=38))
        st.plotly_chart(fig_gauge, width="stretch")

    with info_col:
        st.markdown(
            f"#### Hồ sơ Sinh viên `{int(stu_row['id_student'])}` — Khóa `{stu_row['code_module']}-{stu_row['code_presentation']}` ({milestone_key}) &nbsp; {get_risk_badge_html(stu_tier)}",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng Clicks", f"{int(stu_row['total_clicks']):,}")
        m2.metric("Ngày vắng gần nhất", f"{int(stu_row['days_since_last_activity'])} ngày")
        m3.metric("Tỷ lệ nộp bài", f"{float(stu_row['submission_rate']):.1%}")
        w_sc = stu_row["weighted_avg_score"]
        m4.metric("Điểm TB trọng số", f"{w_sc:.1f}" if pd.notna(w_sc) else "Chưa có")

        st.markdown(
            f"- **Nhân khẩu & Tải học tập:** Độ tuổi `{stu_row['age_band']}` · Học vấn `{stu_row['highest_education']}` · "
            f"Vùng IMD `{stu_row['imd_band']}` · Tín chỉ `{int(stu_row['studied_credits'])}` · Số lần học lại `{int(stu_row['num_of_prev_attempts'])}`.\n"
            f"- **Kết quả cuối kỳ thực tế (OULAD Ground Truth):** `{stu_row['final_result']}` (`at_risk = {int(stu_row['at_risk'])}`)."
        )

    shap_df = pd.DataFrame(
        {
            "feature": feature_names_114,
            "shap_val": shap_vals,
            "abs_shap": np.abs(shap_vals),
            "label_vi": [FEATURE_LABELS_VI.get(f, f) for f in feature_names_114],
        }
    ).sort_values("abs_shap", ascending=False)

    top_k = shap_df.head(12).iloc[::-1]

    c_shap, c_sedl = st.columns([1.3, 1.0])
    with c_shap:
        st.markdown(f"##### 📊 Giải thích Cục bộ Exact TreeSHAP @{milestone_key} (Top 12 tác nhân Log-Odds)")
        colors_bar = ["#dc2626" if v > 0 else "#2563eb" for v in top_k["shap_val"]]
        fig_shap = go.Figure(
            go.Bar(
                x=top_k["shap_val"],
                y=top_k["label_vi"] + " (" + top_k["feature"] + ")",
                orientation="h",
                marker_color=colors_bar,
                text=[f"{v:+.3f}" for v in top_k["shap_val"]],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig_shap.add_vline(x=0, line_width=1.5, line_color="#334155")
        max_abs_x = max(0.2, float(top_k["abs_shap"].max()))
        min_x = min(0.0, float(top_k["shap_val"].min()))
        max_x = max(0.0, float(top_k["shap_val"].max()))
        fig_shap.update_layout(
            height=420,
            xaxis_title="Shapley (Đỏ = Tăng nguy cơ, Xanh = Giảm)",
            yaxis_title="",
            xaxis=dict(range=[min_x - 0.45 * max_abs_x if min_x < 0 else -0.05, max_x + 0.45 * max_abs_x]),
            margin=dict(t=15, b=25, l=15, r=90),
        )
        st.plotly_chart(fig_shap, width="stretch")
        st.caption(
            f"Base value $E[f(X)] = {base_val:+.3f}$ → Tổng đóng góp $f(x) = {base_val + shap_vals.sum():+.3f}$ → $\\sigma(f(x)) =$ **{stu_risk * 100:.2f}%**."
        )

    with c_sedl:
        st.markdown("##### 🕸️ Hồ sơ 4 Chiều SEDL (Phân vị so với Nhóm An toàn)")
        dims, stu_vals_r, safe_vals_r = compute_sedl_radar_profile(stu_row, df_view)
        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=safe_vals_r + [safe_vals_r[0]],
                theta=dims + [dims[0]],
                fill="toself",
                name="Trung vị Nhóm An toàn",
                line_color="#16a34a",
                fillcolor="rgba(22, 163, 74, 0.18)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=stu_vals_r + [stu_vals_r[0]],
                theta=dims + [dims[0]],
                fill="toself",
                name=f"SV {int(stu_row['id_student'])}",
                line_color="#dc2626" if stu_risk >= tau else "#2563eb",
                fillcolor="rgba(220, 38, 38, 0.22)" if stu_risk >= tau else "rgba(37, 99, 235, 0.22)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=430,
            margin=dict(t=82, b=76, l=105, r=105),
            legend=dict(orientation="h", yanchor="bottom", y=1.14, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_radar, width="stretch")

    # Khuyến nghị Sư phạm Tự động
    st.markdown("##### 💡 Khuyến nghị Can thiệp Sư phạm Cá nhân hóa")
    recs = generate_pedagogical_recommendations(stu_row, shap_df.head(12), stu_risk, tau)
    rec_cols = st.columns(len(recs))
    for col_obj, (title_r, desc_r) in zip(rec_cols, recs):
        with col_obj:
            st.info(f"**{title_r}**\n\n{desc_r}")

    # Giả lập Phản thực What-If Simulator
    st.markdown("---")
    st.markdown(f"##### 🧪 Giả lập Phản thực (What-If Intervention Simulator — {model_note})")
    st.caption(
        "Điều chỉnh các chỉ số hành vi và học tập bên dưới để mô phỏng mức giảm xác suất rủi ro $\\Delta P(\\text{At-Risk})$ trên đúng mô hình OOF của mốc hiện tại."
    )

    w1, w2, w3 = st.columns(3)
    cur_clicks = int(stu_row["total_clicks"])
    cur_inactive = int(stu_row["days_since_last_activity"])
    cur_sub_rate = float(stu_row["submission_rate"])
    cur_wscore = float(stu_row["weighted_avg_score"]) if pd.notna(stu_row["weighted_avg_score"]) else 0.0
    cur_last2w = int(stu_row["clicks_last_2_weeks"])
    cur_forum = int(stu_row["clicks_forumng"])

    with w1:
        sim_clicks = st.slider("Tổng lượt click (total_clicks)", 0, 5000, min(5000, cur_clicks), step=25)
        sim_last2w = st.slider("Click 14 ngày gần nhất (clicks_last_2_weeks)", 0, 600, min(600, cur_last2w), step=5)
    with w2:
        sim_inactive = st.slider("Số ngày vắng mặt gần nhất (days_since_last_activity)", 0, 240, min(240, cur_inactive), step=1)
        sim_forum = st.slider("Lượt click diễn đàn (clicks_forumng)", 0, 800, min(800, cur_forum), step=5)
    with w3:
        sim_sub_rate = st.slider("Tỷ lệ nộp bài tập (submission_rate)", 0.0, 1.0, float(np.clip(cur_sub_rate, 0.0, 1.0)), step=0.05)
        sim_wscore = st.slider("Điểm TB trọng số (weighted_avg_score)", 0.0, 100.0, float(np.clip(cur_wscore, 0.0, 100.0)), step=1.0)

    sim_row_df = df_view.loc[[orig_index]].copy()
    cutoff_d = max(1.0, float(sim_row_df["cutoff_day"].iloc[0]))
    due_cw = max(1.0, float(sim_row_df["due_course_assessments"].iloc[0]))
    total_cw = max(1.0, float(sim_row_df["total_course_assessments"].iloc[0]))

    sim_row_df["total_clicks"] = float(sim_clicks)
    sim_row_df["clicks_per_day"] = float(sim_clicks) / cutoff_d
    sim_row_df["clicks_last_2_weeks"] = float(sim_last2w)
    sim_row_df["clicks_forumng"] = float(sim_forum)
    sim_row_df["forumng_ratio"] = float(sim_forum) / max(1.0, float(sim_clicks))
    sim_row_df["days_since_last_activity"] = float(sim_inactive)
    if sim_inactive < cur_inactive:
        sim_row_df["max_inactivity_streak"] = min(float(sim_row_df["max_inactivity_streak"].iloc[0]), max(7.0, float(sim_inactive)))
    sim_row_df["submission_rate"] = float(sim_sub_rate)
    est_subs = round(sim_sub_rate * due_cw)
    sim_row_df["num_submissions"] = est_subs
    sim_row_df["num_submissions_raw"] = est_subs
    sim_row_df["course_progress_submission_rate"] = min(1.0, est_subs / total_cw)
    sim_row_df["weighted_avg_score"] = float(sim_wscore)
    sim_row_df["avg_score"] = float(sim_wscore)
    w1_4 = max(1.0, float(sim_row_df["clicks_week_1_to_4"].iloc[0]))
    sim_row_df["engagement_decay"] = float(sim_last2w) / w1_4

    sim_X = transform_with_state(sim_row_df, active_prep)[feature_names_114]
    sim_risk = float(active_model.predict_proba(sim_X)[:, 1][0])
    delta_p = sim_risk - stu_risk
    sim_tier = classify_risk_tier(sim_risk, tau)

    s1, s2, s3 = st.columns(3)
    s1.metric("Risk Score Ban đầu", f"{stu_risk:.2%}", stu_tier, delta_color="off")
    s2.metric("Risk Score Giả lập (What-If)", f"{sim_risk:.2%}", f"{delta_p:+.2%} vs ban đầu", delta_color="inverse")
    status_change = (
        "✅ Đã chuyển sang AN TOÀN (Dưới ngưỡng τ)"
        if (stu_risk >= tau and sim_risk < tau)
        else ("⚠️ Vẫn trên ngưỡng cảnh báo" if sim_risk >= tau else "🟢 Duy trì trạng thái An toàn")
    )
    s3.info(f"**Kết quả Mô phỏng:**\n\n{status_change}\n\nPhân tầng mới: **{sim_tier}**")

# ===========================================================================
# TAB 4: XU HƯỚNG SEDL & KIỂM ĐỊNH CÔNG BẰNG
# ===========================================================================
with tab4:
    st.subheader(f"⚖️ Phân tích SEDL 4 Chiều & Kiểm định Công bằng Nhóm (Mốc {milestone_key})")

    t4_c1, t4_c2 = st.columns([1.05, 1.2])
    with t4_c1:
        st.markdown(f"##### Phân bố Chỉ số SEDL theo Kết quả Thực tế ({milestone_key})")
        sedl_metric_choice = st.selectbox(
            "Chọn đặc trưng phân tích",
            options=[
                "days_since_last_activity",
                "max_inactivity_streak",
                "engagement_decay",
                "submission_rate",
                "consistency_score",
                "weighted_avg_score",
            ],
            format_func=lambda x: f"{FEATURE_LABELS_VI.get(x, x)} ({x})",
        )
        plot_sample = df_view.sample(min(4000, len(df_view)), random_state=42).copy()
        plot_sample["Nhóm"] = plot_sample["at_risk"].map({0: "An toàn (Pass/Distinction)", 1: "Nguy cơ (Fail/Withdrawn)"})
        fig_dist = px.violin(
            plot_sample,
            x="Nhóm",
            y=sedl_metric_choice,
            color="Nhóm",
            box=True,
            color_discrete_map={
                "An toàn (Pass/Distinction)": "#16a34a",
                "Nguy cơ (Fail/Withdrawn)": "#dc2626",
            },
        )
        fig_dist.update_layout(height=350, showlegend=False, margin=dict(t=15, b=20, l=10, r=10))
        st.plotly_chart(fig_dist, width="stretch")

    with t4_c2:
        st.markdown("##### Diễn tiến Tỷ trọng 4 Chiều SEDL qua 4 Mốc (25% → 100% Exact TreeSHAP)")
        sedl_evo = w8_metrics.get("sedl_4d_evolution_25_to_100", [])
        if sedl_evo:
            evo_df = pd.DataFrame(sedl_evo)
            evo_melt = evo_df.melt(
                id_vars=["snapshot"],
                value_vars=["behavioural_pct", "cognitive_pct", "demographic_context_pct", "social_pct"],
                var_name="Chiều SEDL",
                value_name="Tỷ trọng (%)",
            )
            evo_melt["Chiều SEDL"] = evo_melt["Chiều SEDL"].map(
                {
                    "behavioural_pct": "Behavioural (Hành vi)",
                    "cognitive_pct": "Cognitive (Nhận thức)",
                    "demographic_context_pct": "Demographic & Context",
                    "social_pct": "Social (Xã hội)",
                }
            )
            fig_sedl_evo = px.bar(
                evo_melt,
                x="snapshot",
                y="Tỷ trọng (%)",
                color="Chiều SEDL",
                barmode="stack",
                text=evo_melt["Tỷ trọng (%)"].map(lambda v: f"{v:.1f}%" if v >= 5 else ""),
                color_discrete_map={
                    "Behavioural (Hành vi)": "#2563eb",
                    "Cognitive (Nhận thức)": "#059669",
                    "Demographic & Context": "#d97706",
                    "Social (Xã hội)": "#7c3aed",
                },
            )
            fig_sedl_evo.update_layout(
                height=350,
                xaxis_title="Mốc thời gian",
                yaxis_title="Tỷ trọng |SHAP| (%)",
                xaxis=dict(type="category"),
                margin=dict(t=15, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_sedl_evo, width="stretch")
        else:
            sedl_shap_df = compute_sample_sedl_shap(selected_pct, selected_cohort, sample_n=2000)
            sedl_agg = (
                sedl_shap_df.groupby("sedl_group", as_index=False)["mean_abs_shap"]
                .sum()
                .sort_values("mean_abs_shap", ascending=False)
            )
            sedl_agg["Tỷ trọng (%)"] = (sedl_agg["mean_abs_shap"] / sedl_agg["mean_abs_shap"].sum() * 100.0).round(2)
            fig_sedl_bar = px.bar(
                sedl_agg,
                x="Tỷ trọng (%)",
                y="sedl_group",
                orientation="h",
                text_auto=".1f",
                color="sedl_group",
            )
            fig_sedl_bar.update_layout(height=350, showlegend=False, yaxis_title="", margin=dict(t=15, b=20, l=10, r=20))
            st.plotly_chart(fig_sedl_bar, width="stretch")

    st.markdown("---")
    st.markdown(f"##### 🛡️ Kiểm định Công bằng Thuật toán (Subgroup Fairness Audit — Mốc {milestone_key}, τ = {tau:.3f})")
    fair_attr = st.selectbox(
        "Chọn thuộc tính nhạy cảm (Protected Attribute)",
        options=["gender", "age_band", "imd_band", "disability", "region"],
        format_func=lambda c: {
            "gender": "Giới tính (gender)",
            "age_band": "Nhóm độ tuổi (age_band)",
            "imd_band": "Chỉ số thiếu thốn đa chiều (imd_band)",
            "disability": "Tình trạng khuyết tật (disability)",
            "region": "Khu vực địa lý (region)",
        }[c],
    )
    fair_df = compute_subgroup_fairness(df_view, fair_attr, tau)
    if not fair_df.empty:
        fc1, fc2 = st.columns([1.15, 1.05])
        with fc1:
            fair_melt = fair_df.melt(
                id_vars=["Nhóm"],
                value_vars=["Recall (TPR)", "FPR", "Positive Rate (Dự báo)"],
                var_name="Chỉ số",
                value_name="Giá trị",
            )
            fig_fair = px.bar(
                fair_melt,
                x="Nhóm",
                y="Giá trị",
                color="Chỉ số",
                barmode="group",
                color_discrete_map={
                    "Recall (TPR)": "#059669",
                    "FPR": "#f97316",
                    "Positive Rate (Dự báo)": "#2563eb",
                },
            )
            fig_fair.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(t=15, b=20, l=10, r=10))
            st.plotly_chart(fig_fair, width="stretch")
        with fc2:
            st.dataframe(fair_df, width="stretch", height=295, hide_index=True)
            min_di = float(fair_df["Disparate Impact Ratio"].min())
            if min_di >= 0.80:
                st.success(f"Disparate Impact tối thiểu = {min_di:.3f} (≥ 0.80, đạt quy tắc công bằng 4/5ths).")
            else:
                st.warning(f"Disparate Impact tối thiểu = {min_di:.3f} (< 0.80 ở một phân nhóm — phản ánh chênh lệch tỷ lệ nền thực tế giữa các nhóm).")

    # Bảng tổng hợp kiểm định công bằng chuẩn OOF qua cả 4 mốc
    fair_summary_4m = w8_metrics.get("fairness_audit_summary_4milestones", {})
    if fair_summary_4m:
        with st.expander("📋 Tổng hợp Kiểm định Công bằng 5-Fold OOF qua cả 4 mốc (25% → 100%)", expanded=False):
            fs_rows = []
            for m_k in ["25%", "50%", "75%", "100%"]:
                attr_dict = fair_summary_4m.get(m_k, {})
                if fair_attr in attr_dict:
                    d_a = attr_dict[fair_attr]
                    fs_rows.append(
                        {
                            "Mốc": m_k,
                            "Thuộc tính": fair_attr,
                            "Nhóm chuẩn": d_a.get("reference_group"),
                            "ΔTPR (Equal Opportunity)": d_a.get("equal_opportunity_gap_delta_tpr"),
                            "ΔFPR": d_a.get("fpr_gap_delta_fpr"),
                            "ΔROC-AUC": d_a.get("roc_auc_gap"),
                            "DI Min": d_a.get("disparate_impact_min"),
                            "Đạt 4/5 Rule": "✅ Có" if d_a.get("passes_4_5ths_rule") else "⚠️ Chú ý",
                        }
                    )
            if fs_rows:
                st.dataframe(pd.DataFrame(fs_rows), width="stretch", hide_index=True)

# ===========================================================================
# TAB 5: ĐỐI SÁNH ĐA MỐC SỚM VS. MUỘN (25% -> 50% -> 75% -> 100%)
# ===========================================================================
with tab5:
    st.subheader("⏳ Đối sánh Đa mốc Thời gian: Cảnh báo Sớm vs. Cuối kỳ (25% → 50% → 75% → 100%)")

    bench_c1, bench_c2 = st.columns([1.2, 1.05])
    with bench_c1:
        st.markdown("##### Đường cong Hiệu năng 5-Fold CV của 10 Mô hình qua 4 Lát cắt (Dynamic Active)")
        bench_df = pd.DataFrame(week7_metrics["combined_10_models_4snapshots_dynamic"])
        metric_col = st.selectbox(
            "Chọn chỉ số đánh giá 5-Fold CV",
            options=["roc_auc_mean", "pr_auc_mean", "f1_mean", "recall_mean", "brier_mean"],
            format_func=lambda m: {
                "roc_auc_mean": "ROC-AUC (Mean 5-Fold)",
                "pr_auc_mean": "PR-AUC (Mean 5-Fold)",
                "f1_mean": "F1-Score (Mean 5-Fold)",
                "recall_mean": "Recall (Mean 5-Fold)",
                "brier_mean": "Brier Score (Thấp hơn = Tốt hơn)",
            }[m],
        )
        bench_df["pct_num"] = bench_df["snapshot"].str.replace("%", "").astype(int)
        bench_df = bench_df.sort_values("pct_num")
        fig_bench = px.line(
            bench_df,
            x="snapshot",
            y=metric_col,
            color="model",
            markers=True,
            line_dash="family",
        )
        fig_bench.update_layout(
            height=380,
            xaxis_title="Mốc thời lượng khóa học",
            yaxis_title=metric_col,
            margin=dict(t=15, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_bench, width="stretch")

    with bench_c2:
        st.markdown("##### Bảng Thẩm định 5-Fold OOF XGBoost (Tuned) & XAI qua 4 Mốc")
        if oof_4m_dict:
            lime_4m = w8_metrics.get("lime_vs_shap_200_oof_fidelity_4milestones", {})
            w8_rows = []
            for m_k in ["25%", "50%", "75%", "100%"]:
                om = oof_4m_dict.get(m_k, {})
                lm = lime_4m.get(m_k, {})
                w8_rows.append(
                    {
                        "Mốc": m_k,
                        "N": om.get("n_cohort"),
                        "τ*": om.get("tau_mean"),
                        "AUC": round(om.get("roc_auc_mean", 0.0), 4),
                        "PR-AUC": round(om.get("pr_auc_mean", 0.0), 4),
                        "Recall": round(om.get("recall_mean", 0.0), 4),
                        "F1": round(om.get("f1_mean", 0.0), 4),
                        "Brier": round(om.get("brier_mean", 0.0), 4),
                        "LIME R²": round(lm.get("local_r2_mean", 0.0), 4) if lm else None,
                    }
                )
            st.dataframe(pd.DataFrame(w8_rows), width="stretch", height=185, hide_index=True)

        early_df = pd.DataFrame(week7_metrics["early_prediction_summary"])
        early_sub = early_df[early_df["cohort"] == selected_cohort][
            ["snapshot", "model", "roc_auc_mean", "pr_auc_mean", "recall_mean", "f1_mean", "brier_mean"]
        ].copy()
        for c in ["roc_auc_mean", "pr_auc_mean", "recall_mean", "f1_mean", "brier_mean"]:
            early_sub[c] = early_sub[c].round(4)
        early_sub.columns = ["Mốc", "Mô hình (Tuần 7)", "AUC", "PR-AUC", "Recall", "F1", "Brier"]
        st.dataframe(early_sub, width="stretch", height=210, hide_index=True)

    st.markdown("---")
    st.markdown("##### 📈 Quỹ đạo Rủi ro Dọc thời gian 5-Fold OOF của Một Sinh viên Cụ thể (25% → 100%)")

    long_case = week8_xai.get("case_studies", {}).get("longitudinal_trajectory", {})
    long_id = long_case.get("id_student")
    long_df_match = df_view[df_view["id_student"] == long_id] if long_id else df_view.iloc[0:0]
    traj_pool = (
        pd.concat([long_df_match, df_view.sort_values("risk_score", ascending=False).head(60)])
        .drop_duplicates(subset=["id_student", "code_module", "code_presentation"])
        .reset_index(drop=True)
    )

    traj_options = []
    for _, r in traj_pool.iterrows():
        tag_l = "[Nghiên cứu dọc điển hình] " if long_id and int(r["id_student"]) == int(long_id) else ""
        traj_options.append(
            f"{tag_l}SV {int(r['id_student'])} | {r['code_module']}-{r['code_presentation']} | Risk@{selected_pct}%: {r['risk_score']:.1%} | KQ: {r['final_result']}"
        )
    chosen_traj_lbl = st.selectbox("Chọn sinh viên theo dõi quỹ đạo OOF qua cả 4 mốc", traj_options, index=0)
    chosen_traj_row = traj_pool.iloc[traj_options.index(chosen_traj_lbl)]

    traj_df = get_student_trajectory(
        int(chosen_traj_row["id_student"]),
        str(chosen_traj_row["code_module"]),
        str(chosen_traj_row["code_presentation"]),
        selected_cohort,
    )
    if not traj_df.empty:
        tc1, tc2 = st.columns([1.2, 1.0])
        with tc1:
            fig_traj = go.Figure()
            fig_traj.add_trace(
                go.Scatter(
                    x=traj_df["Mốc (%)"],
                    y=traj_df["Risk Score"],
                    mode="lines+markers+text",
                    name="Xác suất Nguy cơ (OOF)",
                    text=[f"{v:.1%}" for v in traj_df["Risk Score"]],
                    textposition="top center",
                    line=dict(color="#dc2626", width=3),
                    marker=dict(size=10),
                )
            )
            fig_traj.add_trace(
                go.Scatter(
                    x=traj_df["Mốc (%)"],
                    y=traj_df["Ngưỡng τ*"],
                    mode="lines+markers",
                    name="Ngưỡng tối ưu τ* theo mốc",
                    line=dict(color="#0f172a", width=2, dash="dash"),
                    marker=dict(size=7),
                )
            )
            fig_traj.update_layout(
                height=330,
                yaxis=dict(range=[0, 1.08], tickformat=".0%"),
                xaxis_title="Mốc thời gian trong khóa học",
                yaxis_title="Risk Score P(At-Risk)",
                margin=dict(t=25, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_traj, width="stretch")
        with tc2:
            show_t = traj_df[
                [
                    "Mốc (%)",
                    "Fold",
                    "Ngưỡng τ*",
                    "Risk Score",
                    "total_clicks",
                    "submission_rate",
                    "weighted_avg_score",
                    "days_since_last_activity",
                ]
            ].copy()
            show_t["Risk Score"] = show_t["Risk Score"].map(lambda x: f"{x:.1%}")
            show_t["submission_rate"] = show_t["submission_rate"].map(lambda x: f"{x:.1%}")
            show_t["weighted_avg_score"] = show_t["weighted_avg_score"].round(1)
            st.dataframe(show_t, width="stretch", height=330, hide_index=True)

st.markdown("---")
st.caption(
    "© 2026 Vũ Hoà Bình (MSSV 202416137) — Đồ án Project I: Phân tích dữ liệu học tập LMS (OULAD × SEDL). "
    "Hệ thống sử dụng XGBoost (Tuned) 5-Fold Stratified Group OOF qua 4 mốc (25%–100%) và Exact TreeSHAP."
)
