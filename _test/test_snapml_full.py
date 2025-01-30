import pandas as pd
import numpy as np
import json
import yaml
from snapml import GraphFeaturePreprocessor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


def load_params(yaml_file):
    # YAML 설정 파일 로드
    print(f"Loading configuration from {yaml_file}...")
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)
    print("Configuration loaded successfully.")
    return params


def generate_gfp_features(dataset_path, gfp_params, gfp_output_path):
    # GFP 특징 생성 및 CSV 파일 저장
    print("Initializing SnapML GFP...")
    gfp = GraphFeaturePreprocessor()
    gfp.set_params(gfp_params)

    # 데이터셋 로드
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Raw Dataset shape: {df.shape}")

    # Object 타입 데이터 인코딩
    print("Encoding 'counterpart_id' using LabelEncoder...")
    le = LabelEncoder()
    df['counterpart_id_enc'] = le.fit_transform(df['counterpart_id'])

    # 필수 열 선택 및 변환
    selected_columns = ['transaction_id', 'account_id', 'counterpart_id_enc', 'hour', 'min', 'sec']
    selected_df = df[selected_columns]
    selected_df['timestamp'] = selected_df['hour'] * 3600 + selected_df['min'] * 60 + selected_df['sec']
    gfp_input_df = selected_df[['transaction_id', 'account_id', 'counterpart_id_enc', 'timestamp']]

    # GFP 특징 생성
    print("Generating graph features...")
    gfp_features = gfp.fit_transform(gfp_input_df.to_numpy(dtype=np.float64))
    print("Enriched dataset shape: ", gfp_features.shape)

    features_name = get_gfp_features_name(gfp_features, gfp.get_params())

    # GFP 특징 DataFrame 저장
    gfp_features_df = pd.DataFrame(gfp_features, columns=features_name)
    gfp_features_df.to_csv(gfp_output_path, index=False)
    print(f"GFP features saved to {gfp_features_df}")

    return gfp_features_df


def get_gfp_features_name(gfp_features, gfp_params):
    # GFP로 생성된 특징 컬럼 이름 리턴
    print("Generating feature names for GFP output...")
    colnames = []

    # Add raw feature names
    colnames.extend(["transactionID", "sourceAccountID", "targetAccountID", "timestamp"])

    # Add graph pattern feature names
    for pattern in ['fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle']:
        if pattern in gfp_params and gfp_params[pattern]:
            bins = len(gfp_params[pattern + '_bins'])
            if pattern in ['fan', 'degree']:
                for i in range(bins - 1):
                    colnames.append(
                        f"{pattern}_in_bins_{gfp_params[pattern + '_bins'][i]}-{gfp_params[pattern + '_bins'][i + 1]}")
                colnames.append(f"{pattern}_in_bins_{gfp_params[pattern + '_bins'][-1]}-inf")
                for i in range(bins - 1):
                    colnames.append(
                        f"{pattern}_out_bins_{gfp_params[pattern + '_bins'][i]}-{gfp_params[pattern + '_bins'][i + 1]}")
                colnames.append(f"{pattern}_out_bins_{gfp_params[pattern + '_bins'][-1]}-inf")
            else:
                for i in range(bins - 1):
                    colnames.append(
                        f"{pattern}_bins_{gfp_params[pattern + '_bins'][i]}-{gfp_params[pattern + '_bins'][i + 1]}")
                colnames.append(f"{pattern}_bins_{gfp_params[pattern + '_bins'][-1]}-inf")

    vert_feat_names = ["fan", "deg", "ratio", "avg", "sum", "min", "max", "median", "var", "skew", "kurtosis"]

    # Add vertex statistics feature names
    for orig in ['source', 'dest']:
        for direction in ['out', 'in']:
            # Add fan, deg, and ratio features
            for k in [0, 1, 2]:
                if k in gfp_params["vertex_stats_feats"]:
                    colnames.append(f"{orig}_{vert_feat_names[k]}_{direction}")
            for col in gfp_params["vertex_stats_cols"]:
                # Add avg, sum, min, max, median, var, skew, and kurtosis features
                for k in [3, 4, 5, 6, 7, 8, 9, 10]:
                    if k in gfp_params["vertex_stats_feats"]:
                        colnames.append(f"{orig}_{vert_feat_names[k]}_col{col}_{direction}")

    return colnames


def merge_gfp_with_dataset(original_dataset_path, gfp_features_path, merged_gfp_output_path):
    # 기존 데이터셋과 GFP 특징 병합
    print("Merging original dataset with GFP features...")

    # 데이터 로드
    print(f"Loading original dataset from {original_dataset_path}...")
    original_df = pd.read_csv(original_dataset_path)
    gfp_df = pd.read_csv(gfp_features_path)

    # 불필요한 컬럼 제거 후 병합
    original_df = original_df.drop(columns=['transaction_id', 'account_id', 'counterpart_id', 'hour', 'min', 'sec'])
    merged_df = pd.concat([original_df.reset_index(drop=True), gfp_df.reset_index(drop=True)], axis=1)

    # 저장
    merged_df.to_csv(merged_gfp_output_path, index=False)
    print(f"Merged dataset saved to {merged_gfp_output_path}")

    return merged_gfp_output_path


def train_and_evaluate_model(merged_dataset_path, model_type, model_params, importances_output_path):
    # 모델 학습 및 평가
    print(f"Loading merged dataset for {model_type} training...")
    df = pd.read_csv(merged_dataset_path)

    # Object 타입 데이터 인코딩
    print("Encoding object-type columns...")
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # X, y 분리
    X = df.drop(columns=["laundering_yn"])
    y = df["laundering_yn"]  # target 변수

    print(X.shape, y.shape)

    # 데이터 분할 (80% 학습, 20% 테스트)
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 분할 (70% 학습, 15% 검증, 15% 테스트)
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                 test_size=0.15,
    #                                                 stratify=y, #target의 class비율에 맞춰서 분리
    #                                                 random_state=42)

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    #                                               test_size=0.15, stratify=y_train, 
    #                                               random_state=42)

    # 모델 선택
    print(f"Initializing {model_type} model...")
    if model_type == "XGBoost":
        print("Training XGBoost model...")
        model = XGBClassifier(**model_params)
    elif model_type == "LightGBM":
        print("Training LightGBM model...")
        model = LGBMClassifier(**model_params)
    elif model_type == "CatBoost":
        print("Training CatBoost model...")
        model = CatBoostClassifier(**model_params, verbose=0)  # CatBoost는 verbose가 기본적으로 켜져 있음
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 모델 학습
    print("Training model...")
    model.fit(X_train, y_train)

    # 모델 평가
    # y_pred_val = model.predict(X_val)
    # print("Model Val Accuracy:", accuracy_score(y_val, y_pred_val))
    # print("Classification Report:\n", classification_report(y_val, y_pred_val))
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Model Performance Metrics")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-Score   : {f1:.4f}\n")

    print(f"{model_type} Model Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model_type} Classification Report:\n", classification_report(y_test, y_pred))

    # Feature Importance 저장
    print("Saving feature importances...")
    feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    print(feature_importances.sort_values(by='Importance', ascending=False))
    feature_importances_sorted = feature_importances.sort_values(by='Importance', ascending=False)
    feature_importances_sorted.to_csv(importances_output_path, index=False)
    print(f"{model_type} feature importances saved to {importances_output_path}")

    print("Process completed successfully!")


def main():
    print("Starting main process...")

    params = load_params("snapml.yaml")

    # GFP 특징 생성 및 저장
    gfp_features_path = generate_gfp_features(
        params["data"]["dataset_path"],
        params["gfp"],
        params["data"]["gfp_output_path"]
    )

    # 기존 데이터와 GFP 데이터 병합
    merged_dataset_path = merge_gfp_with_dataset(
        params["data"]["dataset_path"],
        params["data"]["gfp_output_path"],
        params["data"]["merged_gfp_output_path"]
    )

    # 모델 학습 및 평가
    train_and_evaluate_model(
        merged_dataset_path,
        params["model"]["type"],
        params["model"]["params"],
        params["data"]["importances_output_path"]
    )


if __name__ == "__main__":
    main()

