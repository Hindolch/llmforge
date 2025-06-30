from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import streamlit as st

@st.cache_resource
def load_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        model = PeftModel.from_pretrained(base_model, "kenzi123/finetuned-tinyllama-lora").to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        raise


st.title("Finetuned TinyLlama 🦙")
model, tokenizer = load_model()

history = st.session_state.get("history", "")

user_input = st.text_input("🧑 You:")

if user_input:
    prompt = history + f"<|user|> {user_input}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    st.write(f"🤖 TinyLLama: {response}")
    st.session_state.history = prompt + response + "\n"