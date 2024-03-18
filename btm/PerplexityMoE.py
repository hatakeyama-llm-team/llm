
from transformers import pipeline
import torch
import numpy as np


def perplexity(model, tokenizer, text) -> torch.Tensor:
    tokenized_input = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model(tokenized_input, labels=tokenized_input)
    ppl = torch.exp(output.loss)
    return ppl.item()
    
class PerplexityMoE:
    def __init__(self,max_new_tokens=100,
    repetition_penalty=1.5
    ):
        self.models=[]
        self.coef=[]
        self.max_new_tokens=max_new_tokens
        self.repetition_penalty=repetition_penalty

    def set_coefs(self,coef):
        self.coef=np.array(coef)

    def append_ELM(self,model,tokenizer):
        pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,
                      max_new_tokens=self.max_new_tokens,
                      repetition_penalty=self.repetition_penalty,
                      )
        self.models.append((model,tokenizer,pipe))
        self.coef.append(1)

    def calc_perplexity(self,text):
        ppl_list=[]
        for model,tokenizer,_ in self.models:
            ppl_list.append(perplexity(model,tokenizer,text))

        return ppl_list

    def ask(self,text,verbose=True,return_ppl_list=True):
        ppl_array=np.array(self.calc_perplexity(text))
        ppl_array=ppl_array*np.array(self.coef)
        best_model_id=np.where(ppl_array==min(ppl_array))[0][0]
        if verbose:
            print("perplexity list")
            for i,ppl in enumerate(ppl_array):
                print(i,ppl)
            print(f"model id {best_model_id} is used")
        pipe=self.models[best_model_id][2]
        gen_text=pipe(text)[0]['generated_text']

        if return_ppl_list:
            return gen_text,ppl_array,best_model_id
        else:
            return gen_text