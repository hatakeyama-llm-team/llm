# %%
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %%
model_dirs=glob.glob("../../models/hf/*")
#model_dirs=[model_dir for model_dir in model_dirs if model_dir.find("epoch")>=0]

def upload_model(model_dir):
    model_name=model_dir.split("/")[-1]
    model=AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",
      torch_dtype=torch.bfloat16,
    )
    tokenizer=AutoTokenizer.from_pretrained(model_dir)

    model.push_to_hub(model_name)
    tokenizer.push_to_hub(model_name)

# %%
for model_dir in model_dirs:
    upload_model(model_dir)


