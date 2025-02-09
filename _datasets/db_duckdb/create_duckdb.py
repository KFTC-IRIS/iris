import duckdb
import pandas as pd
import os
from datetime import datetime


class TransactionAnalyzer:
    def __init__(self, db_name='transaction_analysis.db', threads=16, memory_limit='20GB'):
        """DuckDB 초기화 및 설정"""
        self.db_name = db_name
        self.con = duckdb.connect(self.db_name)
        self.con.execute(f"SET threads TO {threads}")
        self.con.execute(f"SET memory_limit='{memory_limit}'")

    def create_basic_views(self, parquet_file):
        """기본 분석 뷰 생성"""
        print(f"\n{parquet_file} 파일 분석 시작...")

        # 기본 트랜잭션 뷰 생성
        self.con.execute(f"""
            CREATE OR REPLACE VIEW base_transactions AS 
            SELECT * FROM parquet_scan('{parquet_file}')
        """)

    def select_all(self):
        # 1. 전체 통계
        total_trans = self.con.execute("""
            SELECT *
            FROM base_transactions
        """).fetchdf()
        return total_trans

    def analyze_and_save_summary(self):
        """분석 결과 요약 및 저장"""
        print("\n=== 분석 결과 요약 ===")

        # 1. 전체 통계
        total_stats = self.con.execute("""
            SELECT 
                COUNT(*) as total_transactions,
                COUNT(DISTINCT account_id) as unique_accounts,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM base_transactions
        """).fetchdf()

        print("\n1. 전체 통계:")
        print(f"- 총 거래 건수: {total_stats['total_transactions'].iloc[0]:,}건")
        print(f"- 고유 계정 수: {total_stats['unique_accounts'].iloc[0]:,}개")
        print(f"- 총 거래 금액: ${total_stats['total_amount'].iloc[0]:,.2f}")
        print(f"- 평균 거래 금액: ${total_stats['avg_amount'].iloc[0]:,.2f}")

        # 2. 요일별 거래 패턴
        weekly_pattern = self.con.execute("""
            SELECT 
                weekday,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_amount,
                SUM(amount) as total_amount
            FROM base_transactions
            GROUP BY weekday
            ORDER BY 
                CASE weekday
                    WHEN 'Monday' THEN 1
                    WHEN 'Tuesday' THEN 2
                    WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday' THEN 4
                    WHEN 'Friday' THEN 5
                    WHEN 'Saturday' THEN 6
                    WHEN 'Sunday' THEN 7
                END
        """).fetchdf()

        print("\n2. 요일별 거래 패턴:")
        for _, row in weekly_pattern.iterrows():
            print(f"- {row['weekday']}: {row['transaction_count']:,}건 "
                  f"(평균 ${row['avg_amount']:,.2f})")

        # 3. 결제 시스템별 통계
        payment_stats = self.con.execute("""
            SELECT 
                payment_system,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_amount,
                COUNT(DISTINCT account_id) as unique_accounts
            FROM base_transactions
            GROUP BY payment_system
            ORDER BY transaction_count DESC
            LIMIT 5
        """).fetchdf()

        print("\n3. 상위 결제 시스템:")
        for _, row in payment_stats.iterrows():
            print(f"- {row['payment_system']}: {row['transaction_count']:,}건 "
                  f"(계정 수: {row['unique_accounts']:,}개)")

        # 4. 상위 거래 계정
        top_accounts = self.con.execute("""
            SELECT 
                account_id,
                COUNT(*) as transaction_count,
                SUM(amount) as total_amount,
                COUNT(DISTINCT counterpart_id) as unique_counterparts
            FROM base_transactions
            GROUP BY account_id
            ORDER BY transaction_count DESC
            LIMIT 5
        """).fetchdf()

        print("\n4. 가장 활발한 계정:")
        for _, row in top_accounts.iterrows():
            print(f"- 계정 {row['account_id']}: {row['transaction_count']:,}건 "
                  f"(거래처 수: {row['unique_counterparts']:,}개)")

        total_stats.to_csv('analysis_results/total_stats.csv', index=False)
        weekly_pattern.to_csv('analysis_results/weekly_pattern.csv', index=False)
        payment_stats.to_csv('analysis_results/payment_stats.csv', index=False)
        top_accounts.to_csv('analysis_results/top_accounts.csv', index=False)

    def analyze_correlations(self):
        """컬럼 분석 및 상관관계 분석"""
        print("\n=== 컬럼 및 상관관계 분석 ===")

        # 1. 컬럼 정보 확인
        print("\n1. 컬럼 목록 및 데이터 타입:")
        columns_info = self.con.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'base_transactions'
        """).fetchdf()
        print(columns_info)

        # 2. 수치형 컬럼만 선택
        numeric_columns = self.con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'base_transactions'
            AND data_type IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL')
        """).fetchall()
        numeric_columns = [col[0] for col in numeric_columns]

        print("\n2. 수치형 컬럼:")
        print(numeric_columns)

        # 3. 상관관계 분석
        if len(numeric_columns) > 1:
            print("\n3. 수치형 컬럼 간 상관관계:")

            # 상관관계 매트릭스 생성
            correlations = []
            for i, col1 in enumerate(numeric_columns):
                row = []
                for j, col2 in enumerate(numeric_columns):
                    if i <= j:  # 대각선 및 상단 삼각형만 계산
                        corr = self.con.execute(f"""
                            SELECT CORR({col1}, {col2}) as correlation
                            FROM base_transactions
                        """).fetchone()[0]
                        row.append(corr)
                    else:
                        row.append(None)  # 하단 삼각형은 None으로 채움
                correlations.append(row)

            # 결과를 DataFrame으로 변환
            import pandas as pd
            corr_df = pd.DataFrame(correlations,
                                   columns=numeric_columns,
                                   index=numeric_columns)

            # 결과 저장
            corr_df.to_csv('analysis_results/correlations.csv')
            print("\n상관관계 분석 결과가 'analysis_results/correlations.csv'에 저장되었습니다.")

            # 주요 상관관계 출력 (절대값 0.3 이상)
            print("\n주요 상관관계:")
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns[i + 1:], i + 1):
                    corr_value = corr_df.iloc[i, j]
                    if corr_value and abs(corr_value) > 0.3:
                        print(f"- {col1} vs {col2}: {corr_value:.4f}")

        # 4. 범주형 컬럼 분석
        categorical_columns = self.con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'base_transactions'
            AND data_type NOT IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL')
        """).fetchall()
        categorical_columns = [col[0] for col in categorical_columns]

        print("\n4. 범주형 컬럼 분포:")
        for col in categorical_columns:
            print(f"\n{col} 분포:")
            distribution = self.con.execute(f"""
                SELECT {col}, COUNT(*) as count
                FROM base_transactions
                GROUP BY {col}
                ORDER BY count DESC
                LIMIT 5
            """).fetchdf()
            print(distribution)

        # 5. 범주형-수치형 컬럼 간 관계 분석
        print("\n5. 범주형-수치형 컬럼 간 관계:")
        for cat_col in categorical_columns:
            for num_col in numeric_columns:
                if num_col == 'amount':  # amount와의 관계만 분석
                    stats = self.con.execute(f"""
                        SELECT 
                            {cat_col},
                            COUNT(*) as count,
                            AVG({num_col}) as avg_value,
                            STDDEV({num_col}) as std_value
                        FROM base_transactions
                        GROUP BY {cat_col}
                        HAVING COUNT(*) > 100
                        ORDER BY avg_value DESC
                        LIMIT 5
                    """).fetchdf()

                    print(f"\n{cat_col}별 {num_col} 통계:")
                    print(stats)

    def analyze_all_correlations(self):
        """모든 변수 간의 상관관계 및 분포 분석"""
        print("\n=== 전체 상관관계 및 분포 분석 ===")

        # 1. 범주형 변수 간의 관계 분석 (크래머의 V 계수 사용)
        def cramers_v(col1, col2):
            query = f"""
            WITH contingency AS (
                SELECT {col1}, {col2}, COUNT(*) as freq
                FROM base_transactions
                GROUP BY {col1}, {col2}
            ),
            chi_square AS (
                SELECT 
                    SUM(
                        POW(freq - expected, 2) / expected
                    ) as chi_sq
                FROM (
                    SELECT 
                        t1.freq,
                        CAST(t1.row_total * t1.col_total AS FLOAT) / t1.total as expected
                    FROM (
                        SELECT 
                            contingency.*,
                            SUM(freq) OVER (PARTITION BY {col1}) as row_total,
                            SUM(freq) OVER (PARTITION BY {col2}) as col_total,
                            SUM(freq) OVER () as total
                        FROM contingency
                    ) t1
                )
            ),
            dimensions AS (
                SELECT 
                    (COUNT(DISTINCT {col1}) - 1) * 
                    (COUNT(DISTINCT {col2}) - 1) as df,
                    COUNT(DISTINCT {col1}) as r,
                    COUNT(DISTINCT {col2}) as c
                FROM base_transactions
            )
            SELECT 
                SQRT(
                    chi_sq.chi_sq / (
                        (SELECT MIN(r, c) - 1 FROM dimensions) * 
                        (SELECT COUNT(*) FROM base_transactions)
                    )
                ) as cramers_v
            FROM chi_square
            """

            try:
                result = self.con.execute(query).fetchone()[0]
                return result if result else 0
            except:
                return 0

        # 범주형 컬럼 선택
        categorical_columns = self.con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'base_transactions'
            AND data_type IN ('VARCHAR', 'TEXT')
        """).fetchall()
        categorical_columns = [col[0] for col in categorical_columns]

        print("\n1. 범주형 변수 간 관계 분석:")
        cat_correlations = []
        for i, col1 in enumerate(categorical_columns):
            row = []
            for j, col2 in enumerate(categorical_columns):
                if i <= j:
                    if i == j:
                        row.append(1.0)
                    else:
                        v = cramers_v(col1, col2)
                        row.append(v)
                else:
                    row.append(None)
            cat_correlations.append(row)

        # 결과를 DataFrame으로 변환
        import pandas as pd
        cat_corr_df = pd.DataFrame(cat_correlations,
                                   columns=categorical_columns,
                                   index=categorical_columns)

        # 결과 저장
        cat_corr_df.to_csv('analysis_results/categorical_correlations.csv')
        print("\n범주형 변수 간 상관관계가 'categorical_correlations.csv'에 저장되었습니다.")

        # 주요 관계 출력 (크래머의 V > 0.3)
        print("\n주요 범주형 변수 관계:")
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns[i + 1:], i + 1):
                v = cat_correlations[i][j]
                if v and v > 0.3:
                    print(f"- {col1} vs {col2}: {v:.4f}")

        # 2. 조건부 분포 분석
        print("\n2. 주요 범주형 변수의 조건부 분포:")

        # 각 범주형 변수 쌍에 대해 조건부 분포 계산
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns[i + 1:], i + 1):
                # 상위 5개 조합에 대한 분포만 출력
                query = f"""
                    WITH distribution AS (
                        SELECT 
                            {col1},
                            {col2},
                            COUNT(*) as count,
                            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
                        FROM base_transactions
                        GROUP BY {col1}, {col2}
                        ORDER BY count DESC
                        LIMIT 5
                    )
                    SELECT *
                    FROM distribution
                """

                try:
                    result = self.con.execute(query).fetchdf()
                    if not result.empty:
                        print(f"\n{col1} vs {col2} 상위 조합:")
                        print(result)
                except Exception as e:
                    print(f"Error analyzing {col1} vs {col2}: {str(e)}")

        # 3. 시계열 패턴 분석
        print("\n3. 시계열 패턴 분석:")
        temporal_patterns = self.con.execute("""
            SELECT 
                month,
                weekday,
                hour,
                COUNT(*) as transaction_count,
                COUNT(DISTINCT payment_system) as unique_payment_systems,
                COUNT(DISTINCT category_0) as unique_categories
            FROM base_transactions
            GROUP BY month, weekday, hour
            ORDER BY 
                month,
                CASE weekday
                    WHEN 'Monday' THEN 1
                    WHEN 'Tuesday' THEN 2
                    WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday' THEN 4
                    WHEN 'Friday' THEN 5
                    WHEN 'Saturday' THEN 6
                    WHEN 'Sunday' THEN 7
                END,
                hour
        """).fetchdf()

        temporal_patterns.to_csv('analysis_results/temporal_patterns.csv', index=False)
        print("\n시계열 패턴이 'temporal_patterns.csv'에 저장되었습니다.")

        # 4. 결제 시스템과 카테고리 간의 연관성
        print("\n4. 결제 시스템과 카테고리 간의 연관성:")
        payment_category = self.con.execute("""
            SELECT 
                payment_system,
                category_0,
                COUNT(*) as count,
                AVG(amount) as avg_amount,
                COUNT(DISTINCT account_id) as unique_accounts
            FROM base_transactions
            GROUP BY payment_system, category_0
            HAVING COUNT(*) > 100
            ORDER BY count DESC
            LIMIT 10
        """).fetchdf()

        print("\n상위 결제 시스템-카테고리 조합:")
        print(payment_category)
        payment_category.to_csv('analysis_results/payment_category_analysis.csv', index=False)

    def analyze_laundering_patterns(self):
        """자금 세탁 패턴 종합 분석"""
        print("\n=== 자금 세탁 패턴 종합 분석 ===")

        # 1. 자금 세탁 기본 통계
        print("\n1. 자금 세탁 기본 통계:")
        basic_stats = self.con.execute("""
            SELECT 
                laundering_yn,
                COUNT(*) as transaction_count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM base_transactions) as percentage,
                AVG(amount) as avg_amount,
                COUNT(DISTINCT account_id) as unique_accounts,
                COUNT(DISTINCT laundering_schema_type) as schema_types
            FROM base_transactions
            GROUP BY laundering_yn
        """).fetchdf()
        print(basic_stats)

        # 2. 스키마 유형별 분석
        print("\n2. 자금 세탁 스키마 유형별 분석:")
        schema_analysis = self.con.execute("""
            SELECT 
                laundering_schema_type,
                COUNT(*) as transaction_count,
                COUNT(DISTINCT account_id) as unique_accounts,
                AVG(amount) as avg_amount,
                SUM(amount) as total_amount,
                COUNT(DISTINCT payment_system) as payment_systems_used
            FROM base_transactions
            WHERE laundering_schema_type IS NOT NULL
            GROUP BY laundering_schema_type
            ORDER BY transaction_count DESC
        """).fetchdf()
        print(schema_analysis)

        # 3. 시간대별 자금 세탁 패턴
        print("\n3. 시간대별 자금 세탁 패턴:")
        temporal_patterns = self.con.execute("""
            WITH time_patterns AS (
                SELECT 
                    hour,
                    laundering_yn,
                    COUNT(*) as transaction_count,
                    AVG(amount) as avg_amount
                FROM base_transactions
                GROUP BY hour, laundering_yn
            )
            SELECT 
                hour,
                MAX(CASE WHEN laundering_yn = 'Y' THEN transaction_count ELSE 0 END) as laundering_count,
                MAX(CASE WHEN laundering_yn = 'N' THEN transaction_count ELSE 0 END) as normal_count,
                MAX(CASE WHEN laundering_yn = 'Y' THEN avg_amount ELSE 0 END) as laundering_avg_amount,
                MAX(CASE WHEN laundering_yn = 'N' THEN avg_amount ELSE 0 END) as normal_avg_amount
            FROM time_patterns
            GROUP BY hour
            ORDER BY hour
        """).fetchdf()
        print(temporal_patterns)

        # 4. 결제 시스템별 자금 세탁 비율
        print("\n4. 결제 시스템별 자금 세탁 비율:")
        payment_analysis = self.con.execute("""
            SELECT 
                payment_system,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN laundering_yn = 'Y' THEN 1 ELSE 0 END) as laundering_count,
                SUM(CASE WHEN laundering_yn = 'Y' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as laundering_percentage,
                AVG(CASE WHEN laundering_yn = 'Y' THEN amount ELSE NULL END) as laundering_avg_amount,
                AVG(CASE WHEN laundering_yn = 'N' THEN amount ELSE NULL END) as normal_avg_amount
            FROM base_transactions
            GROUP BY payment_system
            HAVING total_transactions > 100
            ORDER BY laundering_percentage DESC
        """).fetchdf()
        print(payment_analysis)

        # 5. 금액대별 자금 세탁 분포
        print("\n5. 금액대별 자금 세탁 분포:")
        amount_distribution = self.con.execute("""
            WITH amount_ranges AS (
                SELECT 
                    CASE 
                        WHEN amount <= 100 THEN '0-100'
                        WHEN amount <= 1000 THEN '101-1000'
                        WHEN amount <= 10000 THEN '1001-10000'
                        WHEN amount <= 100000 THEN '10001-100000'
                        ELSE '100000+'
                    END as amount_range,
                    laundering_yn,
                    COUNT(*) as count
                FROM base_transactions
                GROUP BY 1, laundering_yn
            )
            SELECT 
                amount_range,
                MAX(CASE WHEN laundering_yn = 'Y' THEN count ELSE 0 END) as laundering_count,
                MAX(CASE WHEN laundering_yn = 'N' THEN count ELSE 0 END) as normal_count,
                MAX(CASE WHEN laundering_yn = 'Y' THEN count ELSE 0 END) * 100.0 / 
                    SUM(count) as laundering_percentage
            FROM amount_ranges
            GROUP BY amount_range
            ORDER BY 
                CASE amount_range
                    WHEN '0-100' THEN 1
                    WHEN '101-1000' THEN 2
                    WHEN '1001-10000' THEN 3
                    WHEN '10001-100000' THEN 4
                    WHEN '100000+' THEN 5
                END
        """).fetchdf()
        print(amount_distribution)

        # 6. 스키마 유형별 거래 패턴
        print("\n6. 스키마 유형별 거래 패턴:")
        schema_patterns = self.con.execute("""
            SELECT 
                laundering_schema_type,
                payment_system,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_amount,
                COUNT(DISTINCT account_id) as unique_accounts,
                COUNT(DISTINCT counterpart_id) as unique_counterparts
            FROM base_transactions
            WHERE laundering_schema_type IS NOT NULL
            GROUP BY laundering_schema_type, payment_system
            ORDER BY laundering_schema_type, transaction_count DESC
        """).fetchdf()
        print(schema_patterns)

        # 7. 계정별 자금 세탁 활동
        print("\n7. 계정별 자금 세탁 활동:")
        account_analysis = self.con.execute("""
            SELECT 
                account_id,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN laundering_yn = 'Y' THEN 1 ELSE 0 END) as laundering_transactions,
                COUNT(DISTINCT laundering_schema_type) as unique_schemas,
                SUM(CASE WHEN laundering_yn = 'Y' THEN amount ELSE 0 END) as laundering_amount,
                COUNT(DISTINCT counterpart_id) as unique_counterparts
            FROM base_transactions
            WHERE laundering_yn = 'Y'
            GROUP BY account_id
            ORDER BY laundering_transactions DESC
            LIMIT 10
        """).fetchdf()
        print(account_analysis)

        # 결과 저장
        basic_stats.to_csv('analysis_results/laundering_basic_stats.csv', index=False)
        schema_analysis.to_csv('analysis_results/laundering_schema_analysis.csv', index=False)
        temporal_patterns.to_csv('analysis_results/laundering_temporal_patterns.csv', index=False)
        payment_analysis.to_csv('analysis_results/laundering_payment_analysis.csv', index=False)
        amount_distribution.to_csv('analysis_results/laundering_amount_distribution.csv', index=False)
        schema_patterns.to_csv('analysis_results/laundering_schema_patterns.csv', index=False)
        account_analysis.to_csv('analysis_results/laundering_account_analysis.csv', index=False)


def main():
    # 분석기 초기화
    analyzer = TransactionAnalyzer()

    # 결과를 파일로 저장
    os.makedirs('analysis_results', exist_ok=True)

    # 파일 경로 설정 (실제 경로에 맞게 수정)
    parquet_file = "../mapped_transactions.parquet"

    # 뷰 생성
    analyzer.create_basic_views(parquet_file)

    trans_all = analyzer.select_all()
    print(type(trans_all))
    print(len(trans_all))

    # # 상관관계 분석 수행
    # analyzer.analyze_correlations()
    #
    # # 분석 결과 요약
    # analyzer.analyze_and_save_summary()
    #
    # analyzer.analyze_all_correlations()
    #
    # analyzer.analyze_laundering_patterns()


if __name__ == "__main__":
    main()

