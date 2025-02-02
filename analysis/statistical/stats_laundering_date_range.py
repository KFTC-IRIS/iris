# 필요한 라이브러리 로드
import pandas as pd
import numpy as np
import glob

# 모든 schema 데이터를 로드
file_paths = glob.glob("../../_datasets/laundering_schema_id/transaction_laundering_schema_*.csv")

# 모든 데이터를 하나의 DataFrame으로 합치기
df_list = [pd.read_csv(file) for file in file_paths]
df_combined = pd.concat(df_list, ignore_index=True)

# 연도 추가 (기본적으로 2025년 가정)
df_combined["year"] = 2025

# timestamp 생성
df_combined["timestamp"] = pd.to_datetime(df_combined[["year", "month", "day"]])

# 각 schema별 최대/최소 날짜 간격 계산
schema_date_range = df_combined.groupby("laundering_schema_type")["timestamp"].agg([min, max])
schema_date_range["date_range_days"] = (schema_date_range["max"] - schema_date_range["min"]).dt.days

# 각 그룹(laundering_schema_id) 내 최대 날짜 간격 계산
group_date_range = df_combined.groupby(["laundering_schema_type", "laundering_schema_id"])["timestamp"].agg([min, max])
group_date_range["date_range_days"] = (group_date_range["max"] - group_date_range["min"]).dt.days

# 결과 출력
import ace_tools_open as tools
tools.display_dataframe_to_user(name="Schema-wise Date Range", dataframe=schema_date_range)
tools.display_dataframe_to_user(name="Group-wise Date Range", dataframe=group_date_range)

schema_date_range.to_csv('../_results/stats_laundering_date_range/schema_date_range.csv')
group_date_range.to_csv('../_results/stats_laundering_date_range/id_date_range.csv')

schema_date_range['date_range_days'].describe().to_csv('../_results/stats_laundering_date_range/stats_schema_date_range.csv')
group_date_range['date_range_days'].describe().to_csv('../_results/stats_laundering_date_range/stats_id_date_range.csv')
