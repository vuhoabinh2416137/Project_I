"""
Module: src/preprocessing.py
Dự án: LMS Tiên tiến & Learning Analytics (OULAD)
Nhiệm vụ: Tuần 4 - Tiền xử lý dữ liệu (Data Preprocessing Pipeline)

Mục tiêu chính:
1. Xử lý missing values (studentAssessment, date_submitted, imd_band, first_click_day).
2. Xử lý và phân tích sinh viên Withdrawn sớm (date_unregistration < 0, 0 clicks).
3. Phát hiện và xử lý ngoại lai (IQR capping, Winsorization, Log1p transform).
4. Mã hóa biến phân loại (Ordinal encoding cho thứ bậc, One-Hot cho danh mục).
5. Chuẩn hóa dữ liệu số (RobustScaler / StandardScaler) đảm bảo KHÔNG rò rỉ dữ liệu (Zero Data Leakage).
6. Chia tập Train/Val/Test (70/15/15) theo Stratified Random Split và Stratified Group Split (id_student).
7. Xử lý mất cân bằng lớp (SMOTE, Random Under-sampling, Class Weights).
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Cấu hình encoding stdout/stderr cho Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OULAD_Preprocessing")

# Thiết lập đường dẫn mặc định
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
MODELS_DIR = PROJECT_DIR / "models"


class OULADPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline Tiền xử lý Dữ liệu Toàn diện cho Tập dữ liệu OULAD.
    Đảm bảo 100% Zero Data Leakage: mọi thống kê suy luận (Imputation, Outlier bounds,
    Encoders, Scalers) được tính toán ĐỘC LẬP trên tập huấn luyện (Train set) và áp dụng
    nhất quán cho tập Xác thực (Val set) và tập Kiểm thử (Test set).
    """

    ORDINAL_MAPPINGS = {
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
            "Unknown": 4,  # Điền giá trị trung vị mặc định cho missing
        },
        "age_band": {
            "0-35": 0,
            "35-55": 1,
            "55<=": 2,
        },
    }

    NOMINAL_CATEGORICAL_COLS = ["gender", "disability", "region", "code_module"]
    ORDINAL_CATEGORICAL_COLS = ["highest_education", "imd_band", "age_band"]

    SKEWED_NUMERICAL_COLS = [
        "total_clicks",
        "active_days",
        "daily_click_mean",
        "daily_click_std",
        "studied_credits",
        "clicks_oucontent",
        "clicks_forumng",
        "clicks_quiz",
        "clicks_homepage",
        "clicks_subpage",
        "clicks_resource",
        "clicks_ouwiki",
        "clicks_url",
    ]

    BASE_NUMERICAL_COLS = [
        "num_of_prev_attempts",
        "module_presentation_length",
        "date_registration",
        "clicks_per_active_day",
        "has_pre_course_clicks",
        "oucontent_ratio",
        "forumng_ratio",
        "quiz_ratio",
        "total_course_assessments",
        "avg_score",
        "min_score",
        "max_score",
        "num_submissions",
        "avg_days_early",
        "late_submissions",
        "tma1_score",
        "tma1_days_early",
        "submission_rate",
        "submitted_tma1",
        "late_submission_rate",
    ]

    def __init__(
        self,
        scaling_method: str = "robust",
        outlier_method: str = "capping",
        iqr_multiplier: float = 3.0,
        apply_log1p: bool = True,
        filter_pre_course_unreg: bool = False,
        random_state: int = 42,
    ):
        """
        Khởi tạo Bộ tiền xử lý OULAD.

        Parameters:
        -----------
        scaling_method : str
            Phương pháp chuẩn hóa: 'robust' (RobustScaler), 'standard' (StandardScaler), 'minmax' (MinMaxScaler).
        outlier_method : str
            Xử lý ngoại lai: 'capping' (Winsorization/Capping theo IQR hoặc percentile), 'none'.
        iqr_multiplier : float
            Hệ số IQR dùng cho capping (mặc định 3.0 tương ứng extreme outlier boundary).
        apply_log1p : bool
            Áp dụng hàm log(1 + x) cho các biến tương tác click bị lệch nặng.
        filter_pre_course_unreg : bool
            Nếu True, loại bỏ các sinh viên rút môn trước ngày khai giảng (date_unregistration < 0) và có 0 click.
        random_state : int
            Seed ngẫu nhiên đảm bảo tính tái lập.
        """
        self.scaling_method = scaling_method
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier
        self.apply_log1p = apply_log1p
        self.filter_pre_course_unreg = filter_pre_course_unreg
        self.random_state = random_state

        # Thuộc tính trạng thái học từ Train set
        self.impute_values_: Dict[str, Any] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.one_hot_encoder_: Optional[OneHotEncoder] = None
        self.one_hot_feature_names_: List[str] = []
        self.scaler_: Optional[Any] = None
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def load_dataset(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """Đọc bảng dữ liệu đặc trưng cấp độ Sinh viên - Khóa học."""
        path = csv_path or (PROCESSED_DIR / "student_course_eda_features.csv")
        logger.info(f"Đang đọc dữ liệu từ: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {path}")
        df = pd.read_csv(path)
        logger.info(f"Kích thước dữ liệu gốc: {df.shape}")
        return df

    def filter_early_withdrawals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Phân tích và tùy chọn lọc bỏ các sinh viên rút môn trước khai giảng (date_unregistration < 0)
        và không có bất kỳ tương tác LMS nào (total_clicks == 0).
        """
        pre_start_mask = (df["date_unregistration"] < 0)
        pre_start_zero_click_mask = pre_start_mask & (df["total_clicks"] == 0)

        stats = {
            "total_rows": len(df),
            "pre_start_unreg_count": int(pre_start_mask.sum()),
            "pre_start_unreg_pct": float(pre_start_mask.sum() / len(df) * 100),
            "pre_start_zero_clicks_count": int(pre_start_zero_click_mask.sum()),
            "pre_start_zero_clicks_pct": float(pre_start_zero_click_mask.sum() / len(df) * 100),
        }

        if self.filter_pre_course_unreg:
            logger.info(f"Lọc bỏ {stats['pre_start_zero_clicks_count']} sinh viên rút môn trước khai giảng không tương tác.")
            df_filtered = df[~pre_start_zero_click_mask].copy().reset_index(drop=True)
            return df_filtered, stats
        return df.copy(), stats

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        split_strategy: str = "stratified",
        stratify_col: str = "at_risk",
        group_col: str = "id_student",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chia tập dữ liệu Train / Val / Test (70% / 15% / 15%).
        Hỗ trợ 2 chiến lược:
        1. 'stratified': Phân tầng ngẫu nhiên (Stratified Random Split).
        2. 'group_stratified': Phân tầng có nhóm (Stratified Group Split) theo id_student,
           ngăn ngừa sinh viên học nhiều môn bị rò rỉ giữa Train và Test.
        """
        logger.info(f"Thực hiện chia Train/Val/Test theo chiến lược '{split_strategy}'...")

        if split_strategy == "stratified":
            # Chia 70% Train, 30% Temp (Val + Test)
            train_df, temp_df = train_test_split(
                df,
                test_size=(test_size + val_size),
                stratify=df[stratify_col],
                random_state=self.random_state,
            )
            # Chia đều Temp thành 15% Val và 15% Test
            relative_test_ratio = test_size / (test_size + val_size)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_ratio,
                stratify=temp_df[stratify_col],
                random_state=self.random_state,
            )
        elif split_strategy == "group_stratified":
            # Sử dụng StratifiedGroupKFold với n_splits=7 (1/7 ~ 14.3% mỗi fold)
            sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=self.random_state)
            splits = list(sgkf.split(df, df[stratify_col], groups=df[group_col]))

            # Lấy fold 0 làm Test (~14.3%), fold 1 làm Val (~14.3%), các fold còn lại làm Train (~71.4%)
            test_idx = splits[0][1]
            val_idx = splits[1][1]
            train_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([test_idx, val_idx]))

            train_df = df.iloc[train_idx].copy().reset_index(drop=True)
            val_df = df.iloc[val_idx].copy().reset_index(drop=True)
            test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        else:
            raise ValueError(f"Chiến lược split không hợp lệ: {split_strategy}")

        logger.info(f"Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                    f"Val size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                    f"Test size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

        # Kiểm tra tỷ lệ nhãn
        logger.info(f"Tỷ lệ At-Risk -> Train: {train_df[stratify_col].mean():.4f}, "
                    f"Val: {val_df[stratify_col].mean():.4f}, "
                    f"Test: {test_df[stratify_col].mean():.4f}")

        # Kiểm tra rò rỉ id_student
        train_students = set(train_df[group_col])
        val_students = set(val_df[group_col])
        test_students = set(test_df[group_col])

        overlap_tv = len(train_students.intersection(val_students))
        overlap_tt = len(train_students.intersection(test_students))
        overlap_vt = len(val_students.intersection(test_students))

        logger.info(f"Kiểm tra trùng lặp sinh viên: Train-Val: {overlap_tv}, "
                    f"Train-Test: {overlap_tt}, Val-Test: {overlap_vt}")

        return train_df, val_df, test_df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OULADPreprocessor":
        """
        Tính toán và học toàn bộ tham số thống kê CHỈ TỪ tập huấn luyện (Train set).
        """
        logger.info("Bắt đầu fit bộ tiền xử lý trên tập huấn luyện...")
        df_train = X.copy()

        # 1. Tính toán giá trị Imputation từ Train
        # imd_band missing -> Mode
        self.impute_values_["imd_band_mode"] = df_train["imd_band"].mode().iloc[0] if not df_train["imd_band"].mode().empty else "20-30%"
        # date_registration missing -> Median
        self.impute_values_["date_registration_median"] = float(df_train["date_registration"].median())
        # Assessments
        self.impute_values_["avg_score"] = 0.0
        self.impute_values_["min_score"] = 0.0
        self.impute_values_["max_score"] = 0.0
        self.impute_values_["tma1_score"] = 0.0
        # Days early missing -> sentinel value phản ánh trễ/không nộp (-10.0 ngày)
        self.impute_values_["avg_days_early"] = -10.0
        self.impute_values_["tma1_days_early"] = -10.0

        # 2. Tính toán ngưỡng Capping Outlier từ Train cho các cột skewed
        all_numeric_cols = self.SKEWED_NUMERICAL_COLS + self.BASE_NUMERICAL_COLS
        for col in all_numeric_cols:
            if col in df_train.columns:
                series = df_train[col].dropna()
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25
                p99 = float(series.quantile(0.995))
                p01 = float(series.quantile(0.005))

                if self.outlier_method == "capping":
                    # Capping chặn trên theo min(q75 + iqr_multiplier * iqr, p99)
                    upper_bound = max(q75 + self.iqr_multiplier * iqr, q75)
                    lower_bound = max(q25 - self.iqr_multiplier * iqr, p01)
                    # Không để chặn dưới của các biến click/score bị âm bất hợp lý
                    if "clicks" in col or "score" in col or "credits" in col or "rate" in col:
                        lower_bound = max(0.0, lower_bound)
                    self.outlier_bounds_[col] = (lower_bound, upper_bound)

        # 3. Fit OneHotEncoder cho biến danh mục (Nominal)
        nominal_data = df_train[self.NOMINAL_CATEGORICAL_COLS].fillna("Missing").astype(str)
        self.one_hot_encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.one_hot_encoder_.fit(nominal_data)
        self.one_hot_feature_names_ = list(
            self.one_hot_encoder_.get_feature_names_out(self.NOMINAL_CATEGORICAL_COLS)
        )

        # 4. Xác định danh sách các biến số sau khi biến đổi để fit Scaler
        # Chuẩn bị một bản sao biến đổi tạm thời của Train để fit Scaler
        transformed_temp = self._prepare_features(df_train, is_train=True)
        numeric_features_to_scale = self.SKEWED_NUMERICAL_COLS + self.BASE_NUMERICAL_COLS + self.ORDINAL_CATEGORICAL_COLS

        # Lọc các cột có mặt thực tế
        self.cols_to_scale_ = [c for c in numeric_features_to_scale if c in transformed_temp.columns]

        # Khởi tạo và fit Scaler
        if self.scaling_method == "robust":
            self.scaler_ = RobustScaler()
        elif self.scaling_method == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Scaling method không hợp lệ: {self.scaling_method}")

        self.scaler_.fit(transformed_temp[self.cols_to_scale_])

        # Danh sách feature đầy đủ đầu ra
        self.feature_names_out_ = self.cols_to_scale_ + self.one_hot_feature_names_ + ["imd_was_missing", "has_no_submissions", "has_no_clicks"]
        self.is_fitted_ = True
        logger.info(f"Đã hoàn thành fit preprocessor. Tổng số features đầu ra: {len(self.feature_names_out_)}")
        return self

    def _prepare_features(self, df_in: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Thực hiện Imputation, Capping và biến đổi số học trên DataFrame đầu vào."""
        df = df_in.copy()

        # Indicator flags trước khi điền missing
        df["imd_was_missing"] = df["imd_band"].isna().astype(float)
        df["has_no_submissions"] = (df["num_submissions"].fillna(0) == 0).astype(float)
        df["has_no_clicks"] = (df["total_clicks"].fillna(0) == 0).astype(float)

        # 1. Imputation
        df["imd_band"] = df["imd_band"].fillna(self.impute_values_["imd_band_mode"])
        df["date_registration"] = df["date_registration"].fillna(self.impute_values_["date_registration_median"])

        for col in ["avg_score", "min_score", "max_score", "tma1_score"]:
            if col in df.columns:
                df[col] = df[col].fillna(self.impute_values_[col])

        for col in ["avg_days_early", "tma1_days_early"]:
            if col in df.columns:
                df[col] = df[col].fillna(self.impute_values_[col])

        for col in self.SKEWED_NUMERICAL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 2. Ordinal Mapping
        for col, mapping in self.ORDINAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(mapping.get("Unknown", 2)).astype(float)

        # 3. Outlier Capping (nếu được kích hoạt)
        if self.outlier_method == "capping" and self.outlier_bounds_:
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=lower, upper=upper)

        # 4. Log1p Transformation cho các biến tương tác click
        if self.apply_log1p:
            for col in self.SKEWED_NUMERICAL_COLS:
                if col in df.columns and "clicks" in col:
                    df[col] = np.log1p(np.maximum(0.0, df[col]))

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng toàn bộ bước tiền xử lý đã học lên tập dữ liệu bất kỳ (Train, Val hoặc Test).
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor chưa được fit! Vui lòng gọi .fit() trước.")

        # Bước 1: Chuẩn bị đặc trưng số và ordinal
        prepared_df = self._prepare_features(X, is_train=False)

        # Bước 2: Chuẩn hóa các cột số bằng Scaler đã fit
        scaled_array = self.scaler_.transform(prepared_df[self.cols_to_scale_])
        scaled_df = pd.DataFrame(scaled_array, columns=self.cols_to_scale_, index=X.index)

        # Bước 3: One-Hot Encoding cho các biến danh mục
        nominal_data = X[self.NOMINAL_CATEGORICAL_COLS].fillna("Missing").astype(str)
        encoded_array = self.one_hot_encoder_.transform(nominal_data)
        encoded_df = pd.DataFrame(encoded_array, columns=self.one_hot_feature_names_, index=X.index)

        # Bước 4: Ghép nối bảng đặc trưng hoàn chỉnh
        indicator_cols = ["imd_was_missing", "has_no_submissions", "has_no_clicks"]
        indicators_df = prepared_df[indicator_cols].copy()

        result_df = pd.concat([scaled_df, encoded_df, indicators_df], axis=1)

        # Đảm bảo đúng thứ tự cột
        result_df = result_df[self.feature_names_out_]
        return result_df

    def resample_train_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = "smote",
        sampling_strategy: Any = "auto",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Áp dụng kỹ thuật xử lý mất cân bằng lớp CHỈ TRÊN TẬP HUẤN LUYỆN (Train set).
        Tuyệt đối không resample trên Val hoặc Test set để bảo toàn tính xác thực thực nghiệm.
        """
        logger.info(f"Áp dụng kỹ thuật cân bằng lớp '{method}' trên tập Train...")
        if method == "smote":
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        elif method == "under":
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        elif method == "none":
            return X_train.copy(), y_train.copy()
        else:
            raise ValueError(f"Phương pháp resample không được hỗ trợ: {method}")

        logger.info(f"Kích thước sau {method}: X={X_res.shape}, y={y_res.shape}")
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

    def save_artifacts(self, output_dir: Optional[Path] = None):
        """Lưu toàn bộ artifacts (scaler, encoder, tham số) vào thư mục."""
        out = output_dir or PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler_, out / "scaler.joblib")
        joblib.dump(self.one_hot_encoder_, out / "one_hot_encoder.joblib")
    def save_artifacts(self, output_dir: Optional[Path] = None):
        """Lưu toàn bộ artifacts (scaler, encoder, tham số, metadata) vào thư mục."""
        out = output_dir or PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler_, out / "scaler.joblib")
        joblib.dump(self.one_hot_encoder_, out / "one_hot_encoder.joblib")

        state = {
            "scaling_method": self.scaling_method,
            "outlier_method": self.outlier_method,
            "iqr_multiplier": self.iqr_multiplier,
            "apply_log1p": self.apply_log1p,
            "filter_pre_course_unreg": self.filter_pre_course_unreg,
            "impute_values_": self.impute_values_,
            "outlier_bounds_": self.outlier_bounds_,
            "one_hot_feature_names_": self.one_hot_feature_names_,
            "cols_to_scale_": self.cols_to_scale_,
            "feature_names_out_": self.feature_names_out_,
            "is_fitted_": self.is_fitted_,
        }
        joblib.dump(state, out / "preprocessor_state.joblib")

        meta = {
            "scaling_method": self.scaling_method,
            "outlier_method": self.outlier_method,
            "iqr_multiplier": self.iqr_multiplier,
            "apply_log1p": self.apply_log1p,
            "filter_pre_course_unreg": self.filter_pre_course_unreg,
            "feature_names_out": self.feature_names_out_,
            "impute_values": self.impute_values_,
            "outlier_bounds": {k: [float(v[0]), float(v[1])] for k, v in self.outlier_bounds_.items()},
        }
        with open(out / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu artifacts và metadata vào: {out}")

    @classmethod
    def load_from_artifacts(cls, artifacts_dir: Optional[Path] = None) -> "OULADPreprocessor":
        """Khởi tạo và nạp lại trạng thái preprocessor từ artifacts đã lưu trên disk."""
        directory = artifacts_dir or PROCESSED_DIR
        state = joblib.load(directory / "preprocessor_state.joblib")
        instance = cls(
            scaling_method=state["scaling_method"],
            outlier_method=state["outlier_method"],
            iqr_multiplier=state["iqr_multiplier"],
            apply_log1p=state["apply_log1p"],
            filter_pre_course_unreg=state["filter_pre_course_unreg"],
        )
        instance.scaler_ = joblib.load(directory / "scaler.joblib")
        instance.one_hot_encoder_ = joblib.load(directory / "one_hot_encoder.joblib")
        instance.impute_values_ = state["impute_values_"]
        instance.outlier_bounds_ = state["outlier_bounds_"]
        instance.one_hot_feature_names_ = state["one_hot_feature_names_"]
        instance.cols_to_scale_ = state["cols_to_scale_"]
        instance.feature_names_out_ = state["feature_names_out_"]
        instance.is_fitted_ = state["is_fitted_"]
        return instance


def generate_week4_visualizations(
    raw_df: pd.DataFrame,
    preprocessor: OULADPreprocessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train_clean: pd.DataFrame,
    y_train: pd.Series,
    X_resampled: pd.DataFrame,
    y_resampled: pd.Series,
    figures_dir: Path,
):
    """
    Sinh 6 biểu đồ phân tích chuyên sâu cho Báo cáo Tuần 4 (từ hình 15 đến hình 20).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # =========================================================================
    # HÌNH 15: Phân tích Missing Values & Hiệu quả Imputation
    # =========================================================================
    logger.info("Đang tạo Hình 15: 15_missing_values_heatmap_and_imputation.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    missing_counts = raw_df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=True)
    missing_pct = (missing_cols / len(raw_df)) * 100

    colors = ["#e74c3c" if "unregistration" in c else "#3498db" for c in missing_pct.index]
    axes[0].barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Tỷ lệ thiếu (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Tỷ lệ Missing Values các biến gốc trước tiền xử lý", fontsize=12, fontweight="bold")
    for i, v in enumerate(missing_pct.values):
        axes[0].text(v + 0.8, i, f"{v:.1f}% ({missing_cols.iloc[i]:,})", va="center", fontsize=9)
    axes[0].set_xlim(0, 80)

    # Biểu đồ so sánh phân phối điểm bài tập trước và sau Impute
    raw_scores = raw_df["avg_score"].dropna()
    imputed_scores = raw_df["avg_score"].fillna(0.0)

    axes[1].hist(raw_scores, bins=25, alpha=0.6, label="Chỉ tính SV có nộp bài (N=25,820)", color="#2ecc71", density=True)
    axes[1].hist(imputed_scores, bins=25, alpha=0.45, label="Sau khi Impute = 0 cho SV không nộp (N=32,593)", color="#e67e22", density=True)
    axes[1].set_xlabel("Điểm đánh giá trung bình (avg_score)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mật độ phân phối", fontsize=11, fontweight="bold")
    axes[1].set_title("Phân phối avg_score trước & sau khi Impute score=0 cho SV vắng", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(figures_dir / "15_missing_values_heatmap_and_imputation.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # HÌNH 16: Tác động của Sinh viên Withdrawn Sớm
    # =========================================================================
    logger.info("Đang tạo Hình 16: 16_early_unregistration_impact.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Nhóm thời điểm rút môn
    raw_df["unreg_timing"] = "Không rút môn (Hoàn thành)"
    raw_df.loc[raw_df["date_unregistration"] < 0, "unreg_timing"] = "Rút trước ngày khai giảng (< 0)"
    raw_df.loc[(raw_df["date_unregistration"] >= 0) & (raw_df["date_unregistration"] <= 28), "unreg_timing"] = "Rút trong tháng đầu (0 - 28 ngày)"
    raw_df.loc[raw_df["date_unregistration"] > 28, "unreg_timing"] = "Rút sau tháng đầu (> 28 ngày)"

    timing_counts = raw_df["unreg_timing"].value_counts()
    explode = (0.05, 0.08, 0.05, 0.05)
    axes[0].pie(
        timing_counts.values,
        labels=timing_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],
        explode=explode,
        wedgeprops={"edgecolor": "black", "linewidth": 1},
    )
    axes[0].set_title("Cơ cấu Sinh viên theo Thời điểm Hủy đăng ký (date_unregistration)", fontsize=12, fontweight="bold")

    # Clicks giữa nhóm rút trước khai giảng vs nhóm còn lại
    pre_start_clicks = raw_df[raw_df["date_unregistration"] < 0]["total_clicks"]
    active_clicks = raw_df[raw_df["date_unregistration"].isna() | (raw_df["date_unregistration"] >= 0)]["total_clicks"]

    zero_pre = (pre_start_clicks == 0).sum() / len(pre_start_clicks) * 100
    zero_active = (active_clicks == 0).sum() / len(active_clicks) * 100

    compare_bars = pd.DataFrame({
        "Nhóm sinh viên": ["Rút trước khai giảng (N=2,678)", "Nhóm tham gia khóa học (N=29,915)"],
        "Tỷ lệ hoàn toàn 0 click (%)": [zero_pre, zero_active],
    })
    bars = axes[1].bar(compare_bars["Nhóm sinh viên"], compare_bars["Tỷ lệ hoàn toàn 0 click (%)"], color=["#e74c3c", "#3498db"], edgecolor="black", width=0.5)
    axes[1].set_ylabel("Tỷ lệ không có lượt click nào (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Tỷ lệ 0 Click: Rút trước khai giảng vs Sinh viên vào khóa", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height + 2, f"{height:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(figures_dir / "16_early_unregistration_impact.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # HÌNH 17: Phát hiện và Xử lý Ngoại lai (Outlier Capping & Log1p)
    # =========================================================================
    logger.info("Đang tạo Hình 17: 17_outliers_boxplots_and_capping.png")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    raw_clicks = train_df["total_clicks"].values
    capped_clicks = np.clip(raw_clicks, 0, preprocessor.outlier_bounds_["total_clicks"][1])
    log_capped_clicks = np.log1p(capped_clicks)

    # 1. Boxplot raw vs capped
    axes[0, 0].boxplot([raw_clicks, capped_clicks], tick_labels=["Gốc (Raw)", "Sau Capping (Train bound)"], showmeans=True)
    axes[0, 0].set_ylabel("Tổng lượt click (total_clicks)", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("Boxplot total_clicks trước và sau Outlier Capping", fontsize=12, fontweight="bold")

    # 2. Histogram log1p
    axes[0, 1].hist(log_capped_clicks, bins=35, color="#2980b9", edgecolor="black", alpha=0.8)
    axes[0, 1].set_xlabel("log1p(total_clicks)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Tần suất", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Phân phối total_clicks sau khi Capping & Log1p Transformation", fontsize=12, fontweight="bold")

    # 3. Studied credits raw vs capped
    raw_creds = train_df["studied_credits"].values
    capped_creds = np.clip(raw_creds, 0, preprocessor.outlier_bounds_["studied_credits"][1])
    axes[1, 0].boxplot([raw_creds, capped_creds], tick_labels=["Gốc (Raw)", "Sau Capping (Max = 240)"], showmeans=True)
    axes[1, 0].set_ylabel("Số tín chỉ đăng ký (studied_credits)", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("Boxplot studied_credits trước và sau Capping", fontsize=12, fontweight="bold")

    # 4. Daily click mean
    raw_daily = train_df["daily_click_mean"].fillna(0).values
    capped_daily = np.clip(raw_daily, 0, preprocessor.outlier_bounds_["daily_click_mean"][1])
    axes[1, 1].hist(raw_daily, bins=30, alpha=0.5, label="Gốc (Skew=1.56, Max=221)", color="#e74c3c")
    axes[1, 1].hist(capped_daily, bins=30, alpha=0.6, label="Sau Capping", color="#27ae60")
    axes[1, 1].set_xlabel("daily_click_mean", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Tần suất", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("Phân phối daily_click_mean trước & sau Capping", fontsize=12, fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(figures_dir / "17_outliers_boxplots_and_capping.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # HÌNH 18: Phân phối Đặc trưng Số sau khi Chuẩn hóa (Scaling)
    # =========================================================================
    logger.info("Đang tạo Hình 18: 18_feature_scaling_distributions.png")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    features_to_plot = ["total_clicks", "studied_credits", "avg_score", "active_days"]
    for i, col in enumerate(features_to_plot):
        ax = axes[i // 2, i % 2]
        if col in X_train_clean.columns:
            sns.kdeplot(X_train_clean[col], ax=ax, color="#8e44ad", fill=True, alpha=0.4, linewidth=2)
            ax.set_title(f"Phân phối {col} sau {preprocessor.scaling_method.capitalize()}Scaler", fontsize=12, fontweight="bold")
            ax.set_xlabel("Giá trị sau chuẩn hóa (Scaled)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Mật độ", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(figures_dir / "18_feature_scaling_distributions.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # HÌNH 19: Tính Phân tầng và Tránh Trùng lặp của Train/Val/Test Split
    # =========================================================================
    logger.info("Đang tạo Hình 19: 19_train_val_test_stratification.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    split_proportions = pd.DataFrame({
        "Tập dữ liệu": ["Train (70%)", "Validation (15%)", "Test (15%)"],
        "Tỷ lệ At-Risk": [train_df["at_risk"].mean() * 100, val_df["at_risk"].mean() * 100, test_df["at_risk"].mean() * 100],
        "Tỷ lệ Not At-Risk": [(1 - train_df["at_risk"].mean()) * 100, (1 - val_df["at_risk"].mean()) * 100, (1 - test_df["at_risk"].mean()) * 100],
    })
    split_proportions.set_index("Tập dữ liệu").plot(
        kind="bar",
        stacked=True,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black",
        ax=axes[0],
        width=0.45,
    )
    axes[0].set_ylabel("Tỷ lệ phân bổ (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Bảo toàn Phân bổ Nhãn Mục tiêu (Stratification Consistency)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis="x", rotation=0)

    # Đếm số sinh viên đa môn học và nguy cơ rò rỉ
    counts_students = raw_df.groupby("id_student").size()
    multi_courses = counts_students.value_counts().sort_index()

    bars = axes[1].bar(
        [f"{k} môn" for k in multi_courses.index],
        multi_courses.values,
        color="#34495e",
        edgecolor="black",
        alpha=0.85,
    )
    axes[1].set_xlabel("Số khóa học một sinh viên đăng ký", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Số lượng sinh viên", fontsize=11, fontweight="bold")
    axes[1].set_title("Phân bố Sinh viên Đăng ký Nhiều Khóa (Cơ sở Group-aware Split)", fontsize=12, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, h + 200, f"{h:,}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(figures_dir / "19_train_val_test_stratification.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # HÌNH 20: Xử lý Mất cân bằng Lớp (SMOTE & Resampling)
    # =========================================================================
    logger.info("Đang tạo Hình 20: 20_class_imbalance_and_resampling.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 2-class Imbalance: Trước vs sau SMOTE
    before_counts = y_train.value_counts().sort_index()
    after_counts = y_resampled.value_counts().sort_index()

    comp_df = pd.DataFrame({
        "Gốc (Raw Train)": before_counts.values,
        "Sau SMOTE (Balanced)": after_counts.values,
    }, index=["Not At-Risk (0)", "At-Risk (1)"])

    comp_df.plot(kind="bar", ax=axes[0], color=["#95a5a6", "#3498db"], edgecolor="black", width=0.55)
    axes[0].set_ylabel("Số lượng mẫu trong tập Train", fontsize=11, fontweight="bold")
    axes[0].set_title("Phân bổ Nhãn Nhị phân: Trước vs Sau khi áp dụng SMOTE", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=0)

    # 2. 4-class Imbalance trong OULAD gốc
    raw_4class = raw_df["final_result"].value_counts()[["Distinction", "Pass", "Fail", "Withdrawn"]]
    colors_4 = ["#f39c12", "#27ae60", "#e67e22", "#c0392b"]
    axes[1].bar(raw_4class.index, raw_4class.values, color=colors_4, edgecolor="black", width=0.5)
    axes[1].set_ylabel("Số lượng sinh viên", fontsize=11, fontweight="bold")
    axes[1].set_title("Mức độ mất cân bằng trong Phân loại 4 lớp gốc OULAD", fontsize=12, fontweight="bold")
    for i, (k, v) in enumerate(raw_4class.items()):
        pct = v / len(raw_df) * 100
        axes[1].text(i, v + 250, f"{pct:.1f}%\n({v:,})", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(figures_dir / "20_class_imbalance_and_resampling.png", dpi=300)
    plt.close(fig)
    logger.info(f"Đã xuất đầy đủ 6 biểu đồ vào {figures_dir}")


def run_pipeline(
    input_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    split_strategy: str = "stratified",
    scaling_method: str = "robust",
    outlier_method: str = "capping",
    apply_smote: bool = True,
    filter_pre_course_unreg: bool = False,
    save_figures: bool = True,
) -> Dict[str, Any]:
    """
    Thực thi toàn bộ luồng Tiền xử lý dữ liệu Tuần 4 và xuất các artifacts.
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = FIGURES_DIR

    logger.info("=== BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU TUẦN 4 ===")
    preprocessor = OULADPreprocessor(
        scaling_method=scaling_method,
        outlier_method=outlier_method,
        apply_log1p=True,
        filter_pre_course_unreg=filter_pre_course_unreg,
        random_state=42,
    )

    # 1. Đọc dữ liệu
    df_raw = preprocessor.load_dataset(input_csv)

    # 2. Xử lý rút môn sớm
    df_clean, withdrawal_stats = preprocessor.filter_early_withdrawals(df_raw)

    # 3. Chia tập Train / Val / Test (70 / 15 / 15)
    train_df, val_df, test_df = preprocessor.split_train_val_test(
        df_clean,
        test_size=0.15,
        val_size=0.15,
        split_strategy=split_strategy,
        stratify_col="at_risk",
        group_col="id_student",
    )

    # Tách target
    y_train = train_df["at_risk"].copy()
    y_val = val_df["at_risk"].copy()
    y_test = test_df["at_risk"].copy()

    y_train_4class = train_df["final_result"].copy()
    y_val_4class = val_df["final_result"].copy()
    y_test_4class = test_df["final_result"].copy()

    # 4. Fit Preprocessor TRÊN TẬP TRAIN ONLY
    preprocessor.fit(train_df, y_train)

    # 5. Transform Train, Val, Test độc lập
    logger.info("Đang áp dụng biến đổi cho tập Train...")
    X_train_transformed = preprocessor.transform(train_df)
    logger.info("Đang áp dụng biến đổi cho tập Validation...")
    X_val_transformed = preprocessor.transform(val_df)
    logger.info("Đang áp dụng biến đổi cho tập Test...")
    X_test_transformed = preprocessor.transform(test_df)

    # 6. Xử lý mất cân bằng lớp (SMOTE) CHỈ TRÊN TẬP TRAIN
    if apply_smote:
        X_train_resampled, y_train_resampled = preprocessor.resample_train_data(
            X_train_transformed, y_train, method="smote"
        )
    else:
        X_train_resampled, y_train_resampled = X_train_transformed.copy(), y_train.copy()

    # 7. Lưu các tập dữ liệu đã xử lý vào thư mục data/processed
    logger.info("Đang lưu trữ dữ liệu đã tiền xử lý vào disk...")
    X_train_transformed.to_csv(out_dir / "train_features.csv", index=False)
    y_train.to_frame().to_csv(out_dir / "train_target.csv", index=False)

    X_val_transformed.to_csv(out_dir / "val_features.csv", index=False)
    y_val.to_frame().to_csv(out_dir / "val_target.csv", index=False)

    X_test_transformed.to_csv(out_dir / "test_features.csv", index=False)
    y_test.to_frame().to_csv(out_dir / "test_target.csv", index=False)

    # Lưu thêm phiên bản train sau SMOTE
    X_train_resampled.to_csv(out_dir / "train_features_smote.csv", index=False)
    y_train_resampled.to_frame().to_csv(out_dir / "train_target_smote.csv", index=False)

    # Lưu target 4 lớp
    y_train_4class.to_frame().to_csv(out_dir / "train_target_4class.csv", index=False)
    y_val_4class.to_frame().to_csv(out_dir / "val_target_4class.csv", index=False)
    y_test_4class.to_frame().to_csv(out_dir / "test_target_4class.csv", index=False)

    # 8. Lưu artifacts
    preprocessor.save_artifacts(out_dir)

    # 9. Sinh biểu đồ
    if save_figures:
        logger.info("Sinh biểu đồ phân tích trực quan hóa...")
        generate_week4_visualizations(
            raw_df=df_raw,
            preprocessor=preprocessor,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            X_train_clean=X_train_transformed,
            y_train=y_train,
            X_resampled=X_train_resampled,
            y_resampled=y_train_resampled,
            figures_dir=figures_dir,
        )

    summary = {
        "status": "SUCCESS",
        "total_records": len(df_raw),
        "train_shape": list(X_train_transformed.shape),
        "val_shape": list(X_val_transformed.shape),
        "test_shape": list(X_test_transformed.shape),
        "smote_train_shape": list(X_train_resampled.shape),
        "num_features": len(preprocessor.feature_names_out_),
        "withdrawal_stats": withdrawal_stats,
    }
    logger.info(f"=== PIPELINE HOÀN TẤT THÀNH CÔNG ===\n{json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline Tiền xử lý OULAD Tuần 4")
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn file CSV đầu vào")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất kết quả")
    parser.add_argument("--split-strategy", type=str, default="stratified", choices=["stratified", "group_stratified"])
    parser.add_argument("--scaling", type=str, default="robust", choices=["robust", "standard", "minmax"])
    parser.add_argument("--outlier-method", type=str, default="capping", choices=["capping", "none"])
    parser.add_argument("--smote", action="store_true", default=True, help="Áp dụng SMOTE trên Train")
    parser.add_argument("--filter-pre-start", action="store_true", default=False, help="Lọc bỏ SV rút môn trước khai giảng và 0 click")
    parser.add_argument("--no-figures", action="store_true", default=False, help="Không vẽ lại biểu đồ")

    args = parser.parse_args()
    run_pipeline(
        input_csv=Path(args.input) if args.input else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        split_strategy=args.split_strategy,
        scaling_method=args.scaling,
        outlier_method=args.outlier_method,
        apply_smote=args.smote,
        filter_pre_course_unreg=args.filter_pre_start,
        save_figures=not args.no_figures,
    )
