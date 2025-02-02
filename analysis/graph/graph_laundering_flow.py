import pandas as pd

for schema in [1, 2, 3, 4, 5, 6, 7]:
    for id in range(0, 200):
        print(schema, id)
        file_path = f'../../_datasets/laundering_schema_id/transaction_laundering_schema_{schema}_id{id}.csv'
        try:
            df = pd.read_csv(file_path)
        except:
            print(f'##### There is no {file_path}')
            continue

        df['year'] = 2024   # 2024 가정
        df.rename(columns={'min': 'minute', 'sec': 'second'}, inplace=True)

        df['timestamp'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hour', 'minute', 'second']]
        )
        df_sorted = df.sort_values(by='timestamp')

        # 자금 흐름
        flow_data = df_sorted[
            ['transaction_id', 'account_id', 'counterpart_id', 'transaction_direction', 'amount', 'timestamp']]
        flow_data['flow'] = flow_data.apply(
            lambda row: f"{row['counterpart_id']} → {row['account_id']}" if row['transaction_direction'] == 'inbound'
            else f"{row['account_id']} → {row['counterpart_id']}",
            axis=1
        )

        flow_data = flow_data[['timestamp', 'transaction_id', 'flow', 'amount']]
        print(flow_data)
        flow_data.to_csv(f'../_results/laundering_flow/laundering_flow_schema_{schema}_id{id}.csv', index=False)
