import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = "../../_datasets/combined_dataset.parquet"
df = pd.read_parquet(file_path)
print(df.dtypes)

output_dir = "../_results/stats_dist_hist"
os.makedirs(output_dir, exist_ok=True)

# object → int
df['age'] = pd.to_numeric(df['age'], errors='coerce')

columns_to_plot = [
    'month', 'day',
    'hour'
    'weekday', 'channel', 'payment_system',
    'category_0', 'category_1', 'category_2', 'age',
    'assigned_bank_type', 'assigned_bank',
    'amount', 'initial_balance'
]

# laundering_yn=0: 307561470
# laundering_yn=1: 35107
for column in columns_to_plot:
    if 'amount' in column or 'initial_balance' in column:
        # laundering_yn=0
        sns.kdeplot(
            df[df['laundering_yn'] == 0]['amount'],
            color='blue',
            label='laundering_yn=0',
            fill=True,
            alpha=0.5
        )

        # laundering_yn=1
        sns.kdeplot(
            df[df['laundering_yn'] == 1]['amount'],
            color='red',
            label='laundering_yn=1',
            fill=True,
            alpha=0.5
        )
        plt.ylabel("Density")
    else:
        # laundering_yn=0
        sns.countplot(
            x=df[df['laundering_yn'] == 0][column],
            # order=df[column].value_counts().index,
            color='blue',
            label='laundering_yn=0',
            alpha=0.5,
            stat='percent',
            linewidth=0
        )

        # laundering_yn=1
        sns.countplot(
            x=df[df['laundering_yn'] == 1][column],
            # order=df[column].value_counts().index,
            color='red',
            label='laundering_yn=1',
            alpha=0.5,
            stat='percent',
            linewidth=0
        )
        # plt.ylabel("Frequency")
        plt.ylabel("Percent")

    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.title(f"Distribution of {column}")

    plt.xlabel(column)
    plt.legend()

    output_path = os.path.join(output_dir, f"{column}_dist.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved combined plot for {column} to {output_path}")
