import modal
from modal import App, Image, gpu

app = App("tinyllama-finetuner")

# Define image and add local dir to it
image = Image.debian_slim().pip_install("torch").add_local_dir(
    local_path=".",  # local folder with your .jsonl file
    remote_path="/notebooks"
)
@app.function(gpu="T4", timeout=60 * 30, image=image)
def run_finetune(jsonl_path: str = "/notebooks/data_cleaned.jsonl"):
    print(f"🔥 Starting finetuning with data: {jsonl_path}")
    
    # Load and inspect sample
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                print("Sample →", line)
                break

    # ✅ Save to /tmp instead of /root/data
    import os
    output_dir = "/tmp/tinyllama"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "final_model.pt")
    with open(model_path, "w") as f:
        f.write("dummy model weights")

    print(f"✅ Saved model to {model_path}")
    return model_path
