from unittest.mock import patch, MagicMock
import pandas as pd

def test_pipeline_flow():
    # dummy DataFrame-like objects for mocking
    dummy_df = pd.DataFrame({"text": ["Hello"], "comments": [[]]})
    cleaned_df = pd.DataFrame({"prompt": ["Hi"], "completion": ["Hello back"]})

    with patch("tasks.data_ingestion_task.reddit_ingestion_task") as mock_ingest, \
         patch("tasks.data_processor_task.reddit_processor_task") as mock_process, \
         patch("tasks.finetune_task.trigger_modal_finetune.submit") as mock_trigger:

        mock_ingest.return_value = dummy_df
        mock_process.return_value = cleaned_df
        mock_trigger.return_value = None

        # Import after mocks are in place
        from run_pipeline import reddit_ingestion_flow

        try:
            reddit_ingestion_flow()
        except Exception:
            pass  
        
        mock_ingest.assert_called_once()
        mock_process.assert_called_once()
        mock_trigger.assert_called_once()
