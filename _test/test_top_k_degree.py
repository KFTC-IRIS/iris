import json
import pandas as pd
import dask.dataframe as dd

def print_top_k(file_name, parquet_file, k):
    # JSON 파일에서 데이터 로드
    with open(file_name, 'r') as f:
        data_dict = json.load(f)

    # 값이 큰 순서대로 정렬
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_keys = [key for key, _ in sorted_items[:k]]

    # Parquet 파일 로드
    ddf = dd.read_parquet(parquet_file)
    df = ddf.compute()

    # 필터링하여 각 키의 자금세탁 및 정상 거래 건수 카운트
    laundering_counts = {}
    for key in top_k_keys:
        filtered_df = df[(df['source'] == key) | (df['target'] == key)]
        laundering_count = filtered_df[filtered_df['laundering_yn'] == 1].shape[0]
        normal_count = filtered_df[filtered_df['laundering_yn'] == 0].shape[0]
        laundering_counts[key] = {'laundering_yn=0': normal_count, 'laundering_yn=1': laundering_count}

    # 결과 출력
    for key, counts in laundering_counts.items():
        print(f"{key}: Normal Transactions = {counts['laundering_yn=0']}, Laundering Transactions = {counts['laundering_yn=1']}")


# JSON 및 Parquet 파일 이름
json_file = '../_datasets/degree_centrality_cd_v0.1.json'
parquet_file = '../_datasets/combined_dataset_v0.1.parquet'

# top-10개 출력
print_top_k(json_file, parquet_file, 10)

"""
other_1: 22.13198643877874, Normal Transactions = 40134277, Laundering Transactions = 0
bank_commission_1: 11.920451349559888, Normal Transactions = 21616618, Laundering Transactions = 0
financing_1: 3.983072185710205, Normal Transactions = 7222927, Laundering Transactions = 0
taxes_1: 3.9782106158245862, Normal Transactions = 7214111, Laundering Transactions = 0
pension_1: 1.4859143512263662, Normal Transactions = 2694566, Laundering Transactions = 0
highway_tolls_1: 1.1094768628757157, Normal Transactions = 2011932, Laundering Transactions = 0
condominium_1: 0.5541991148148843, Normal Transactions = 1004988, Laundering Transactions = 0
phone_internet_6: 0.45097236912197264, Normal Transactions = 817796, Laundering Transactions = 0
phone_internet_8: 0.4500850885019681, Normal Transactions = 816187, Laundering Transactions = 0
phone_internet_9: 0.44929265702220017, Normal Transactions = 814750, Laundering Transactions = 0
"""
