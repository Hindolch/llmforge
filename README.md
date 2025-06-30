### 🚀 LLMForge

**LLMForge** is a modular LLMOps pipeline for Reddit-based data curation and LoRA fine-tuning on `TinyLLaMA-1.1B` every 12 hours. Designed to reflect real-world MLOps orchestration, it automates ingestion, cleaning, training, and Hugging Face syncing — all backed by `Prefect`, `Modal`, and clean CI/CD practices.

---

## ✅ Key Features

- 🔎 Reddit-based prompt-completion dataset creation
- 🧹 Toxicity filtering using Detoxify
- ⚙️ Fine-tuning TinyLLaMA with PEFT + LoRA adapters
- ☁️ Hugging Face Dataset + Model Hub uploads
- 🔁 Fully orchestrated using `Prefect` flows and tasks
- 🚀 GPU-powered training on Modal (`T4` instance)
- 🧪 CI pipeline with `pytest` and GitHub Actions
- 📧 Email alerts on pipeline trigger
- ⏱ The whole workflow from start to finsh is being done every 12 hours as said above. (`Fully Automated`) 

---

## 🧠 Why This Matters

Built with an engineer's mindset under tight resource constraints, LLMForge shows:

- 💡 Real-world thinking in orchestrated MLOps
- 🎯 Efficient LoRA fine-tuning without overfitting
- 🧼 NLP-focused cleaning pipelines that reduce noise
- ⚙️ Cloud-first thinking (Modal + Prefect + Hugging Face)

---

## 🔧 Architecture Overview

| Module             | Role                                                  |
|-------------------|-------------------------------------------------------|
| `src/`            | All pipeline logic (ingestion, processing, finetune) |
| `tasks/`          | Prefect-wrapped tasks for orchestration              |
| `tests/`          | Basic CI-compatible test coverage                    |
| `modal`           | Runs remote GPU training using fine-tuned HF dataset |
| `prefect.yaml`    | Contains deployment metadata for Prefect Cloud       |

---

### 🛠️ Features Implemented

| Feature                          | Status |
|----------------------------------|--------|
| Reddit Ingestion with PRAW       | ✅     |
| Toxicity Filtering (Detoxify)    | ✅     |
| Prompt-Completion Generation     | ✅     |
| Hugging Face Dataset Push        | ✅     |
| LoRA Fine-tuning on Modal        | ✅     |
| Hugging Face Model Push          | ✅     |
| Prefect Workflow Automation      | ✅     |
| CI + `pytest` test integration   | ✅     |
| Email Alerts per pipeline run    | ✅     |
| Streamlit Inference UI (local)   | ✅     |

---

### ⚙️ Prefect Orchestration

- ✅ This project was deployed on **Prefect Cloud** using `prefect.yaml`
- 🔁 A **worker pool** is configured, enabling scalable background runs
- ❌ However, due to budget constraints, periodic scheduling is disabled

💡 Despite system limitations, the orchestration is ready for production use.

---

## 📁 Project Structure

```bash
.
├── run_pipeline.py            # Main flow trigger
├── inference.py               # (Optional) Streamlit-based inference
├── download_model.py          # HF pull helper
├── prefect.yaml               # Prefect deployment metadata
├── requirements.txt
├── src/
│   ├── data_ingestion.py
│   ├── data_processor.py
│   ├── email.py
│   └── finetune.py
├── tasks/
│   ├── data_ingestion_task.py
│   ├── data_processor_task.py
│   └── finetune_task.py
├── tests/
│   └── test_pipeline.py
└── .env.example               # Add your creds here
````

---

### 📷 Screenshots


### Prefect Flow, Worker Pool & Pipeline Deployment
![Screenshot from 2025-06-30 12-45-19](https://github.com/user-attachments/assets/0af1526f-1020-40a5-baf7-108b4610da67)
![Screenshot from 2025-06-30 18-52-34](https://github.com/user-attachments/assets/673dc554-a1df-4c11-ae13-553ab6c4f325)
![Screenshot from 2025-06-30 18-52-12](https://github.com/user-attachments/assets/936881dd-9d21-4afa-8e6b-56d6922e4405)

### Modal UI of the Job
![Screenshot from 2025-06-30 19-36-05](https://github.com/user-attachments/assets/74a9533e-0a70-4cda-a368-5f9862588a57)

### HF Fine-Tuned Model Update & Dataset 
![Screenshot from 2025-06-30 18-05-28](https://github.com/user-attachments/assets/f7c61de8-9ed0-449c-9dee-1ebb08f3ad8d)

### CI: GitHub Actions passing 
![Screenshot from 2025-06-30 18-59-53](https://github.com/user-attachments/assets/e563a12f-064f-4998-a0f2-98d92e99ca50)

### Streamlit Inference UI
![Screenshot from 2025-06-30 18-30-01](https://github.com/user-attachments/assets/0a60489a-4eaf-4d32-974f-30869d95f0ec)

### Email Notification
![Screenshot from 2025-06-30 18-37-01](https://github.com/user-attachments/assets/330a8303-ae8a-4ce8-88d6-a448a3e2e35c)


## 🧪 Testing

```bash
pytest tests/
```

✅ All test cases pass (including basic ETL & pipeline stub validation)

---

### ⚠️ Dev Notes on System Constraints

LLMForge was built entirely on a **4GB GPU laptop**, hence:

* Streamlit app was tested locally, not deployed
* Prefect's recurring schedules were not activated
* Adapter merging was skipped to save VRAM

Despite that, the entire pipeline is:

* Modular, scalable, and cleanly orchestrated
* Cloud-ready: can run on GPU infra + Prefect Cloud + HF Hub

---

## 🧠 Future Extensions

* [ ] Merge LoRA adapters for full model export
* [ ] Cloud host the Streamlit UI (via Modal or Render)
* [ ] Add cron-like retriggers using Prefect schedules
* [ ] Support multi-subreddit ingestion and balancing
* [ ] Evaluate and log Rouge/Loss metrics post fine-tune

---

## 🧑‍💻 Author

**Hindol R. Choudhury**
*MLOps • LLM Infra • Applied AI*
📫 [LinkedIn](https://www.linkedin.com/in/hindol-choudhury/)


---

