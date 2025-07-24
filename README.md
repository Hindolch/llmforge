# 🚀 LLMForge

**LLMForge** is a modular LLMOps pipeline originally designed for Reddit-based dataset curation and LoRA fine-tuning on `TinyLLaMA-1.1B`. It automates ingestion, cleaning, training, and Hugging Face syncing — all backed by `Prefect`, `Modal`, and CI/CD best practices.

---

## ✅ What’s New (June 2025)

We’ve added **a one-line fine-tuning CLI flow** using Modal GPU compute and any Hugging Face dataset.

```bash
./llmforge finetune \
  --dataset-repo yourusername/your-dataset \
  --model-repo yourusername/your-model-name \
  --hf-auth ./hf_token.txt \
  --push-to-hub
````

No setup beyond a HF token file and basic venv install. Full walkthrough and screenshots below 👇

---

## 🧠 Why This Update Matters

This update streamlines LLMForge into a **modular LoRA trainer** for the OSS community:

* 🔥 No MLOps knowledge needed to fine-tune & push your own model
* 🧼 Old full-pipeline logic is retained but commented for now
* 🔁 Fast launch, easy to build on top of

---

## ⚙️ Key Features

* 🔎 Reddit-based prompt-completion dataset creation
* 🧹 Toxicity filtering using Detoxify
* ⚙️ LoRA fine-tuning on Modal with GPU (`T4`)
* ☁️ Hugging Face Dataset + Model Hub syncing
* 🔁 `Prefect` orchestration pipeline (legacy)
* 🚀 New! One-command fine-tuning using unsloth via CLI + Modal
* 🧪 CI setup with `pytest` and GitHub Actions
* 📧 Email alerts for pipeline (optional)

---

## 🆕 New: Simple Finetune CLI

After cloning the repo:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place your Hugging Face token in a file named `hf_token.txt`.

Then run:

```bash
./llmforge finetune \
  --dataset-repo yourusername/your-dataset(whatever dataset repo name you want to give) \
  --model-repo yourusername/your-model-name(same naming conventions like dataset) \
  --hf-auth ./hf_token.txt \
  --push-to-hub
```

That's it. Modal will handle the GPU job and push the model to your HF.

---

## 🧼 Notes on Code Cleanup

* `tinylama` and old full-pipeline fine-tuning logic are **commented** (not deleted).
* The 12-hour looped pipeline is **inactive**, but logic is retained for future revival.
* `modal secret` creation is handled dynamically using your HF token.

---

## 🔧 Architecture Overview

| Module     | Role                                    |
| ---------- | --------------------------------------- |
| `src/`     | Pipeline + finetuning logic             |
| `tasks/`   | Prefect-wrapped tasks for orchestration |
| `tests/`   | Test coverage                           |
| `llmforge` | CLI to trigger training                 |

---

### 🛠️ Features Implemented

| Feature                        | Status |
| ------------------------------ | ------ |
| Reddit Ingestion with PRAW     | ✅      |
| Toxicity Filtering (Detoxify)  | ✅      |
| Prompt-Completion Generation   | ✅      |
| Hugging Face Dataset Push      | ✅      |
| LoRA Fine-tuning on Modal      | ✅      |
| Hugging Face Model Push        | ✅      |
| CLI-based Finetune Trigger     | ✅      |
| CI + `pytest` test integration | ✅      |
| Email Alerts                   | ✅      |
| Streamlit Inference UI (local) | ✅      |

---

## 📁 Project Structure

```bash
.
├── hf_uploader.py
├── inference.py
├── LICENSE
├── llmforge                   # CLI wrapper for launching fine-tune jobs
├── prefect.yaml
├── __pycache__
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── data_ingestion.py
│   ├── data_processor.py
│   ├── email.py
│   ├── finetune.py
│   ├── __init__.py
│   └── __pycache__
├── tasks/
│   ├── data_ingestion_task.py
│   ├── data_processor_task.py
│   ├── finetune_task.py
│   └── __pycache__
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── venv
└──

---
```
### 📷 Screenshots

#### 🔄 One-command Modal GPU Job

<img width="1920" height="994" alt="460599508-74a9533e-0a70-4cda-a368-5f9862588a57" src="https://github.com/user-attachments/assets/5b58a537-54e0-429e-8e04-3d443e14c87e" />


#### ✅ Model Pushed to Hugging Face

![hf\_model\_push](https://github.com/user-attachments/assets/f7c61de8-9ed0-449c-9dee-1ebb08f3ad8d)

#### 🧪 CI + Prefect Pipelines

![ci\_pass](https://github.com/user-attachments/assets/e563a12f-064f-4998-a0f2-98d92e99ca50)
![prefect\_worker](https://github.com/user-attachments/assets/936881dd-9d21-4afa-8e6b-56d6922e4405)

#### 🎛️ Streamlit Inference UI

![streamlit](https://github.com/user-attachments/assets/0a60489a-4eaf-4d32-974f-30869d95f0ec)

#### Model and Dataset Pushed to Hugging Face after CLI run
<img width="586" height="170" alt="Screenshot from 2025-07-24 23-45-55" src="https://github.com/user-attachments/assets/a98f1c00-0cf2-4d77-9956-e2242183006e" />
<img width="574" height="125" alt="Screenshot from 2025-07-24 23-46-00" src="https://github.com/user-attachments/assets/98795585-1879-4014-b3e8-ecef0d4f9046" />

### Terminal outputs after I ran CLI command
<img width="1697" height="629" alt="Screenshot from 2025-07-24 23-07-25" src="https://github.com/user-attachments/assets/5382bb0c-2ed6-4398-8649-5b1749706781" />

<img width="1685" height="631" alt="Screenshot from 2025-07-24 23-07-31" src="https://github.com/user-attachments/assets/b14b9ba5-eb87-40ae-adcc-44d4331f4042" />
<img width="1687" height="250" alt="Screenshot from 2025-07-24 23-07-43" src="https://github.com/user-attachments/assets/8b9d7509-02a2-476c-ba8b-c577198a4e71" />

---

## 🧪 Testing

```bash
pytest tests/
```

✅ All tests pass for ETL logic and CLI triggers

---

### ⚠️ Dev Notes

LLMForge was built on a **4GB GPU laptop**:

* Modal handles all remote fine-tuning
* The 12-hr pipeline is **currently disabled** for cost reasons
* Adapter merging was skipped for memory savings

Despite that:

* It’s cloud-ready and reproducible
* Works on real Reddit + HF datasets
* Fully OSS and tweakable

---

## 🧠 Future Extensions

* [ ] Merge adapters for complete model export
* [ ] Auto-deploy Streamlit via Modal or Render
* [ ] Reactivate scheduled flows (Prefect)
* [ ] Add Rouge/BLEU scoring post-finetune
* [ ] Support data balancing + multi-source ingestion

---

## 🧑‍💻 Author

**Hindol R. Choudhury**
*MLOps • LLM Infra • Applied AI*
📫 [LinkedIn](https://www.linkedin.com/in/hindol-choudhury/)

---

