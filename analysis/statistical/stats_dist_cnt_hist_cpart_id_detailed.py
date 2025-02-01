import pandas as pd

file_path = '../../_datasets/combined_dataset.parquet'
df = pd.read_parquet(file_path)
# file_path = "../../_datasets/combined_dataset_sample.csv"
# df = pd.read_csv(file_path)

# counterpart_id에서 _숫자 패턴 제거 (일반적인 counterpart_category)
df['counterpart_category'] = df['counterpart_id'].astype(str).str.replace(r'_\d+$', '', regex=True)

# 숫자로만 이루어진 counterpart_id는 account_id로 통합
df['detailed_category'] = df['counterpart_id'].astype(str).apply(
    lambda x: 'account_id' if x.isdigit() else x
)

# 특정 값(cash, cryptocurrency_exchange, foreign)만 `_숫자` 패턴 유지
detailed_categories = {'cash', 'cryptocurrency_exchange', 'foreign'}
df['detailed_category'] = df.apply(
    lambda x: x['counterpart_id'] if x['counterpart_category'] in detailed_categories
    else x['detailed_category'], axis=1
)

# laundering_yn 값별 detailed_category 빈도 계산
counterpart_counts = df.groupby(['detailed_category', 'laundering_yn']).size().unstack(fill_value=0)

# 컬럼명 변경 및 정렬
counterpart_counts.columns = ['laundering_yn_0', 'laundering_yn_1']
counterpart_counts_sorted = counterpart_counts.sort_values(by='laundering_yn_1', ascending=False)

# 인덱스 재설정
counterpart_counts_sorted.reset_index(inplace=True)

# 결과 저장
output_file = "../_results/stats_dist_cnt/counterpart_id_counts_detailed.csv"
counterpart_counts_sorted.to_csv(output_file, index=False)
print(counterpart_counts_sorted)
print(f"Saved detailed counterpart category counts to {output_file}")
