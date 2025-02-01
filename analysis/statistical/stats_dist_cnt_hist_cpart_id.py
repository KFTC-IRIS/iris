import pandas as pd

# file_path = "../../_datasets/combined_dataset_sample.csv"
# df = pd.read_csv(file_path)
file_path = '../../_datasets/combined_dataset.parquet'
df = pd.read_parquet(file_path)

# counterpart_id에서 _숫자 패턴 제거
df['counterpart_category'] = df['counterpart_id'].astype(str).str.replace(r'_\d+$', '', regex=True)

"""
account_id, 'pension' 'taxes' 'bank_commission' 'other' 'financial_assets_managed'
'real_estate' 'supermarket_groceries' 'pharmacy' 'phone_internet'
'insurance' 'electricity_gas_bill' 'clothing'
'public_administration_services' 'furniture' 'fuel' 'online_shopping'
'gardening' 'professional_services' 'legal_expenses' 'cash' 'condominium'
'financing' 'work_income' 'car_accessories_maintenance'
'restaurants_takeaway' 'highway_tolls' 'personal_care' 'medical_visit'
'jewelry' 'pet_shop' 'cryptocurrency_exchange' 'transportation'
'hospital' 'television' 'foreign' 'sports' 'travel' 'renovations'
'domestic_utilities' 'hotels' 'music' 'shows_concerts' 'laundry'
'leisure' 'school_fee' 'parking_washing' 'associations_clubs' 'subsidies'
'training_course' 'consumer_electronics' 'vehicle' 'stationery'
'inheritance' 'art' 'water_bill' 'alcohol_tobacco' 'domestic_services'
'gold_diamonds' 'luxury' 'veterinarian' 'boat' 'consulting' 'betting'
'games' 'children_shop' 'publishing’
"""

# account_id는 'account_id'로 통합
df.loc[df['counterpart_id'].astype(str).str.isdigit(), 'counterpart_category'] = 'account_id'

# laundering_yn 값별 counterpart_category 빈도 계산
counterpart_counts = df.groupby(['counterpart_category', 'laundering_yn']).size().unstack(fill_value=0)

counterpart_counts.columns = ['laundering_yn_0', 'laundering_yn_1']
counterpart_counts_sorted = counterpart_counts.sort_values(by='laundering_yn_1', ascending=False)
print(counterpart_counts_sorted)

# 인덱스 재설정
counterpart_counts_sorted.reset_index(inplace=True)

output_file = "../_results/stats_dist_cnt/counterpart_id_counts.csv"
counterpart_counts_sorted.to_csv(output_file, index=False)
print(f"Saved custom bin counts to {output_file}")
