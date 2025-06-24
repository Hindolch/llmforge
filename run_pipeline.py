# flows/reddit_pipeline.py
from prefect import flow
from tasks.data_ingestion_task import reddit_ingestion_task
from tasks.data_processor_task import reddit_processor_task


@flow
def reddit_ingestion_flow():
    df = reddit_ingestion_task()
    print("✅ Reddit Ingestion Flow complete. Sample:\n", df.head())
    df_cleaned = reddit_processor_task(df)
    print("✅ Reddit Processing Flow complete. Sample:\n", df_cleaned.head())
if __name__ == "__main__":
    reddit_ingestion_flow()
