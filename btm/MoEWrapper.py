from transformers import GPT2Config, GPT2Model
import torch
import numpy as np


class MoEWrapper(GPT2Model):

    config_class = GPT2Config
    #load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"

    verbose=True
    fix_mode=False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.model_list=[]


    def append_model(self,model):
        self.model_list.append(model)
    
    def set_tokenizer(self,tokenizer):
        self.tokenizer=tokenizer

    def set_model_id(self,model_id):
        self.model=self.model_list[model_id]

    def calc_perplexity(self,tokenized_input):
        ppl_list=[]
        for model in self.model_list:
            ppl_list.append(perplexity(model,tokenized_input))
        return np.array(ppl_list)

    def fix_model(self,model_id):
        self.set_model_id(model_id)
        self.fix_mode=True

    def set_flexible_mode(self):
        self.fix_mode=False

    # wrapper functions
    #def forward(self,*args, **kwargs):
    #    ret=self.model.forward(*args,**kwargs)
    #    return ret

    def generate(self,input_ids, attention_mask,
                  **generate_kwargs):

        if not self.fix_mode:
            ppl_array=self.calc_perplexity(input_ids)
            best_model_id=np.where(ppl_array==min(ppl_array))[0][0]
            self.set_model_id(best_model_id)
    
            if self.verbose:
                print(f"model {best_model_id} will be used")
                print("ppl array: ",ppl_array)


        ret=self.model.generate(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                  **generate_kwargs)
        return ret


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def perplexity(model, tokenized_input) -> torch.Tensor:
    with torch.inference_mode():
        output = model(tokenized_input.to(device), labels=tokenized_input)
    ppl = torch.exp(output.loss)
    return ppl.item()