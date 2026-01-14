import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "0"

MODEL_ID = "haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B"

print(f"Device: {torch.cuda.get_device_name(0)}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map=None,
    low_cpu_mem_usage=True
)
model.to("cuda")

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    inputs = inputs.to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "London, 1850. The fog was thick upon the streets,"

print("\n" + "=" * 50)
print(f"PROMPT: {prompt}\n")
try:
    print(generate(prompt))
except Exception as e:
    print(f"Error: {e}")
print("=" * 50)
