from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, lag, mean, stddev, when, isnan, lit
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def run_spark_etl():
    try:
        spark = SparkSession.builder \
            .appName("AirflowETL") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        input_path = "/opt/airflow/data/convenient_dataset.csv"
        output_path = "/opt/airflow/output/processed_financial_data"
        if not os.path.exists(input_path):
            raise Exception(f"Input file not found at {input_path}")
        
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        
        df = df.withColumn("Date", col("Date").cast("date"))
        df = df.dropDuplicates()
        df = remove_low_variance_cols(df, min_unique=20)
        
        numeric_cols = [f.name for f in df.schema if f.dataType.typeName() in ["double", "int", "float"]]
        for col_name in numeric_cols:
            median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
            df = df.withColumn(col_name, when(isnan(col(col_name)), median_val).otherwise(col(col_name)))

        df = df.withColumn("Daily_Volatility", col("Daily_High") - col("Daily_Low")) \
            .withColumn("Daily_Return_Pct", (col("Close_Price") - col("Open_Price")) / col("Open_Price") * 100)
        
        window_spec = Window.orderBy("Date").rowsBetween(-6, 0)
        df = df.withColumn("7D_MA_Close", mean("Close_Price").over(window_spec))
        
        window_spec_rsi = Window.orderBy("Date")
        df = df.withColumn("Price_Change", col("Close_Price") - lag("Close_Price", 1).over(window_spec_rsi)) \
            .withColumn("Gain", when(col("Price_Change") > 0, col("Price_Change")).otherwise(0)) \
            .withColumn("Loss", when(col("Price_Change") < 0, -col("Price_Change")).otherwise(0))
        
        avg_gain = df.select(mean("Gain")).first()[0]
        avg_loss = df.select(mean("Loss")).first()[0]
        rs = avg_gain / avg_loss if avg_loss != 0 else lit(0)
        df = df.withColumn("RSI", lit(100 - (100 / (1 + rs))))
        
        window_spec_anomaly = Window.orderBy("Date").rowsBetween(-30, 0)
        df = df.withColumn("Rolling_Mean", mean("Close_Price").over(window_spec_anomaly)) \
            .withColumn("Rolling_Std", stddev("Close_Price").over(window_spec_anomaly)) \
            .withColumn("Z_Score", (col("Close_Price") - col("Rolling_Mean")) / col("Rolling_Std")) \
            .withColumn("Is_Anomaly", when(abs(col("Z_Score")) > 3, 1).otherwise(0))
        
        df = df.withColumn("Market_Regime",
            when(col("Daily_Return_Pct") > 0.5, "Bullish")
            .when(col("Daily_Return_Pct") < -0.5, "Bearish")
            .otherwise("Neutral"))
        
        df.write.mode("overwrite").option("header", "true").csv(output_path)
        df.write.mode("overwrite").parquet(f"{output_path}.parquet")
        
        visualize_data(df)
        
        spark.stop()
    except Exception as e:
        if 'spark' in locals():
            spark.stop()
        raise

def remove_low_variance_cols(df, min_unique=5, max_cardinality_pct=1):
    unique_counts = {c: df.select(c).distinct().count() for c in df.columns}
    
    low_var_cols = [c for c, cnt in unique_counts.items() if cnt < min_unique]
    
    high_card_cols = []
    for c in [f.name for f in df.schema if f.dataType.typeName() in ["string"]]:
        unique_ratio = unique_counts[c] / df.count() * 100
        if unique_ratio > max_cardinality_pct:
            high_card_cols.append(c)
    
    cols_to_drop = list(set(low_var_cols + high_card_cols))
    return df.drop(*cols_to_drop)

def visualize_data(df):
    numeric_cols = [f.name for f in df.schema if f.dataType.typeName() in ["double", "int", "float"]]
    correlation_df = df.select(numeric_cols).limit(1000).toPandas()
    corr_matrix = correlation_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Financial Indicators")
    plt.tight_layout()
    plt.savefig("/opt/airflow/output/correlation_matrix.png")
    plt.close()
    
    ts_df = df.select("Date", "Close_Price", "7D_MA_Close", "Daily_Volatility").limit(1000).toPandas()
    
    plt.figure(figsize=(15, 6))
    plt.plot(ts_df['Date'], ts_df['Close_Price'], label='Close Price', alpha=0.7)
    plt.plot(ts_df['Date'], ts_df['7D_MA_Close'], label='7D Moving Average', color='orange')
    plt.title("Price Trend with Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("/opt/airflow/output/price_trend.png")
    plt.close()
    
    try:
        result = seasonal_decompose(ts_df.set_index('Date')['Close_Price'], model='additive', period=30)
        result.plot()
        plt.suptitle("Seasonal Decomposition of Close Price")
        plt.tight_layout()
        plt.savefig("/opt/airflow/output/seasonal_decomposition.png")
        plt.close()
    except Exception as e:
        print(f"Could not perform seasonal decomposition: {e}")

def generate_report():
    try:
        import pdfkit
        from jinja2 import Environment, FileSystemLoader
        
        if not all(os.path.exists(f"/opt/airflow/output/{img}") for img in [
            "correlation_matrix.png", 
            "price_trend.png",
            "seasonal_decomposition.png"
        ]):
            raise Exception("Visualization files missing for report generation")
        
        env = Environment(loader=FileSystemLoader('/opt/airflow/reports'))
        template = env.get_template('report_template.html')
        
        html_out = template.render(
            date=datetime.now().strftime("%Y-%m-%d"),
            correlation_plot="correlation_matrix.png",
            price_plot="price_trend.png",
            seasonal_plot="seasonal_decomposition.png"
        )
        
        with open("/opt/airflow/output/financial_report.html", "w") as f:
            f.write(html_out)
        
        pdfkit.from_string(html_out, '/opt/airflow/output/financial_report.pdf')
    except Exception as e:
        print(f"Report generation failed: {e}")
        raise

def data_quality_check():
    try:
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.parquet("/opt/airflow/output/processed_financial_data.parquet")
        
        missing_values = {col: df.filter(df[col].isNull()).count() for col in df.columns}
        
        anomaly_check = df.filter(abs(col("Z_Score")) > 3).count()
        
        market_regime_dist = df.groupBy("Market_Regime").count().collect()
        
        with open("/opt/airflow/output/data_quality_log.txt", "w") as f:
            f.write(f"=== Data Quality Report ===\n")
            f.write(f"Generated at: {datetime.now()}\n\n")
            f.write(f"Missing Values:\n")
            for col, count in missing_values.items():
                f.write(f"{col}: {count} missing\n")
            
            f.write(f"\nAnomaly Count (Z-Score > 3): {anomaly_check}\n")
            
            f.write(f"\nMarket Regime Distribution:\n")
            for row in market_regime_dist:
                f.write(f"{row['Market_Regime']}: {row['count']}\n")
        
        spark.stop()
    except Exception as e:
        if 'spark' in locals():
            spark.stop()
        raise

with DAG(
    "financial_data_etl_enhanced",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 3, 31),
    catchup=False,
    tags=["finance", "analytics"],
) as dag:

    run_etl_task = PythonOperator(
        task_id="run_spark_etl",
        python_callable=run_spark_etl,
    )
    
    generate_report_task = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )
    
    data_quality_task = PythonOperator(
        task_id="data_quality_check",
        python_callable=data_quality_check,
    )

    run_etl_task >> [generate_report_task, data_quality_task]
    
    

# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, datediff, lag, mean, stddev, when, isnan
# from pyspark.sql.window import Window
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from financial_economics_2000_present.opt.airflow.dags.remove_low_variance import remove_low_variance_cols

# default_args = {
#     "owner": "airflow",
#     "depends_on_past": False,
#     "retries": 1,
#     "retry_delay": timedelta(minutes=5),
# }

# def run_spark_etl():
#     try:
#         spark = SparkSession.builder \
#             .appName("AirflowETL") \
#             .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
#             .getOrCreate()
        
#         input_path = "/opt/airflow/data/convenient_dataset.csv"
#         output_path = "/opt/airflow/output/processed_financial_data.csv"
#         if not os.path.exists(input_path):
#             raise Exception(f"Input file not found at {input_path}")
        
#         df = spark.read.csv(input_path, header=True, inferSchema=True)
        
#         df = df.withColumn("Date", col("Date").cast("date"))

#         df = df.dropDuplicates()
#         df = remove_low_variance_cols(df, min_unique=20)
        
#         numeric_cols = [f.name for f in df.schema if f.dataType.typeName() in ["double", "int", "float"]]
#         for col_name in numeric_cols:
#             median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
#             df = df.withColumn(col_name, when(isnan(col(col_name)), median_val).otherwise(col(col_name)))

#         df = df.withColumn("Daily_Volatility", col("Daily_High") - col("Daily_Low")) \
#             .withColumn("Daily_Return_Pct", (col("Close_Price") - col("Open_Price")) / col("Open_Price") * 100)

#         window_spec = Window.orderBy("Date").rowsBetween(-6, 0)
#         df = df.withColumn("7D_MA_Close", mean("Close_Price").over(window_spec))
        
#         df.write.csv(output_path, header=True, mode="overwrite")
#         spark.stop()
#     except Exception as e:
#         if 'spark' in locals():
#             spark.stop()
#         raise

# with DAG(
#     "financial_data_etl",
#     default_args=default_args,
#     schedule="@daily",
#     start_date=datetime(2025, 3, 31),
#     catchup=False,
# ) as dag:

#     run_etl_task = PythonOperator(
#         task_id="run_spark_etl",
#         python_callable=run_spark_etl,
#     )
#     run_etl_task 
