"""
Suite Kiểm thử Độc lập: src/tests/test_preprocessing.py
Kiểm tra tính đúng đắn, toàn vẹn và Zero Data Leakage của Pipeline Tiền xử lý OULAD Tuần 4.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Thiết lập encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import OULADPreprocessor, PROCESSED_DIR


class TestOULADPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Khởi tạo và nạp dữ liệu kiểm thử."""
        cls.preprocessor = OULADPreprocessor(
            scaling_method="robust",
            outlier_method="capping",
            apply_log1p=True,
            random_state=42,
        )
        cls.raw_df = cls.preprocessor.load_dataset()
        cls.train_df, cls.val_df, cls.test_df = cls.preprocessor.split_train_val_test(
            cls.raw_df, test_size=0.15, val_size=0.15, split_strategy="stratified"
        )
        cls.preprocessor.fit(cls.train_df, cls.train_df["at_risk"])
        cls.X_train = cls.preprocessor.transform(cls.train_df)
        cls.X_val = cls.preprocessor.transform(cls.val_df)
        cls.X_test = cls.preprocessor.transform(cls.test_df)

    def test_01_no_missing_values(self):
        """Test 1: Đảm bảo không còn bất kỳ giá trị NaN / Null nào sau khi tiền xử lý."""
        train_nans = self.X_train.isna().sum().sum()
        val_nans = self.X_val.isna().sum().sum()
        test_nans = self.X_test.isna().sum().sum()

        self.assertEqual(train_nans, 0, f"Tập Train còn {train_nans} giá trị NaN!")
        self.assertEqual(val_nans, 0, f"Tập Validation còn {val_nans} giá trị NaN!")
        self.assertEqual(test_nans, 0, f"Tập Test còn {test_nans} giá trị NaN!")

    def test_02_feature_shapes_and_column_alignment(self):
        """Test 2: Đảm bảo số lượng cột và tên cột hoàn toàn khớp nhau giữa Train, Val, Test."""
        self.assertEqual(self.X_train.shape[1], self.X_val.shape[1])
        self.assertEqual(self.X_train.shape[1], self.X_test.shape[1])
        self.assertListEqual(list(self.X_train.columns), list(self.X_val.columns))
        self.assertListEqual(list(self.X_train.columns), list(self.X_test.columns))
        self.assertGreaterEqual(self.X_train.shape[1], 40, "Số lượng đặc trưng phải >= 40")

    def test_03_zero_data_leakage_imputation(self):
        """Test 3: Kiểm tra Zero Data Leakage - Giá trị imputation chỉ được học từ Train."""
        train_reg_median = float(self.train_df["date_registration"].median())
        self.assertAlmostEqual(
            self.preprocessor.impute_values_["date_registration_median"],
            train_reg_median,
            places=4,
            msg="Giá trị Impute date_registration bị rò rỉ từ tập khác!",
        )

        train_imd_mode = self.train_df["imd_band"].mode().iloc[0]
        self.assertEqual(
            self.preprocessor.impute_values_["imd_band_mode"],
            train_imd_mode,
            msg="Mode của imd_band không khớp với tập Train!",
        )

    def test_04_stratification_consistency(self):
        """Test 4: Kiểm tra tính phân tầng - Tỷ lệ nhãn At-Risk đồng nhất giữa các tập (+/- 0.5%)."""
        train_ratio = self.train_df["at_risk"].mean()
        val_ratio = self.val_df["at_risk"].mean()
        test_ratio = self.test_df["at_risk"].mean()

        self.assertAlmostEqual(train_ratio, val_ratio, delta=0.005, msg="Lệch tỷ lệ At-Risk giữa Train và Val!")
        self.assertAlmostEqual(train_ratio, test_ratio, delta=0.005, msg="Lệch tỷ lệ At-Risk giữa Train và Test!")

    def test_05_group_stratified_split_zero_student_overlap(self):
        """Test 5: Kiểm tra Group Stratified Split - Không có bất kỳ sinh viên nào xuất hiện ở cả Train và Test."""
        prep_group = OULADPreprocessor(random_state=42)
        train_g, val_g, test_g = prep_group.split_train_val_test(
            self.raw_df, split_strategy="group_stratified", group_col="id_student"
        )
        train_students = set(train_g["id_student"])
        val_students = set(val_g["id_student"])
        test_students = set(test_g["id_student"])

        self.assertEqual(len(train_students.intersection(test_students)), 0, "Có sinh viên bị rò rỉ giữa Train và Test!")
        self.assertEqual(len(train_students.intersection(val_students)), 0, "Có sinh viên bị rò rỉ giữa Train và Val!")
        self.assertEqual(len(val_students.intersection(test_students)), 0, "Có sinh viên bị rò rỉ giữa Val và Test!")

    def test_06_all_features_are_numeric(self):
        """Test 6: Kiểm tra mọi đặc trưng đầu ra đều có kiểu số (float hoặc int)."""
        for col in self.X_train.columns:
            self.assertTrue(
                np.issubdtype(self.X_train[col].dtype, np.number),
                f"Cột {col} vẫn còn kiểu không phải số: {self.X_train[col].dtype}",
            )

    def test_07_smote_resampling_balance(self):
        """Test 7: Kiểm tra SMOTE cân bằng chính xác tỷ lệ 50:50 trên tập Train."""
        y_train = self.train_df["at_risk"].copy()
        X_res, y_res = self.preprocessor.resample_train_data(self.X_train, y_train, method="smote")

        counts = y_res.value_counts()
        self.assertEqual(counts[0], counts[1], "SMOTE chưa tạo được số lượng mẫu 50:50 trên Train!")
        self.assertEqual(len(X_res), len(y_res))

    def test_08_outlier_bounds_applied(self):
        """Test 8: Kiểm tra ngưỡng chặn ngoại lai Outlier Capping được áp dụng chuẩn xác."""
        self.assertIn("studied_credits", self.preprocessor.outlier_bounds_)
        lower, upper = self.preprocessor.outlier_bounds_["studied_credits"]
        # Raw max credits is 655, but after capping it must be <= upper bound
        prep_raw = self.preprocessor._prepare_features(self.train_df)
        self.assertLessEqual(prep_raw["studied_credits"].max(), upper)

    def test_09_artifacts_saved_and_reloadable(self):
        """Test 9: Kiểm tra tính bền vững - Artifacts đã lưu vào disk có thể nạp lại và cho kết quả giống hệt."""
        scaler_file = PROCESSED_DIR / "scaler.joblib"
        enc_file = PROCESSED_DIR / "one_hot_encoder.joblib"
        state_file = PROCESSED_DIR / "preprocessor_state.joblib"
        meta_file = PROCESSED_DIR / "preprocessing_metadata.json"

        self.assertTrue(scaler_file.exists(), "File scaler.joblib không tồn tại!")
        self.assertTrue(enc_file.exists(), "File one_hot_encoder.joblib không tồn tại!")
        self.assertTrue(state_file.exists(), "File preprocessor_state.joblib không tồn tại!")
        self.assertTrue(meta_file.exists(), "File preprocessing_metadata.json không tồn tại!")

        loaded_prep = OULADPreprocessor.load_from_artifacts(PROCESSED_DIR)
        sample = self.val_df.head(10)
        out1 = self.preprocessor.transform(sample)
        out2 = loaded_prep.transform(sample)

        np.testing.assert_allclose(out1.values, out2.values, err_msg="Kết quả transform sau khi load lại bị lệch!")


if __name__ == "__main__":
    unittest.main()
