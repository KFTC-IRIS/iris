###  통합 데이터셋(combined_dataset)
- combined_dataset_v0.1.parquet
  - transaction_dataset.parquet, account_dataset.parquet, train_target_dataset.parquet 통합
- combined_dataset_v0.2.parquet
  - add column : source, target (account_id/counterpart_id에 대해 transaction_direction 값(inbound or outbound)을 고려) 
  - drop column : account_id, counterpart_id, transaction_direction
- combined_dataset_v0.3.parquet
  - add column : source_deg, target_deg (degree centrality)
