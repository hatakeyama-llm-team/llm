from transformers import PreTrainedModel
from .PPLMoEConfig import MoEConfig
from transformers import AutoModelForCausalLM
import torch
import numpy as np


class PPLMoEModel(PreTrainedModel):
    config_class = MoEConfig
    verbose = True
    fix_mode = False

    def __init__(self, config):
        super().__init__(config)
        self.model_list = []
        for model_name in self.config_class.model_list:
            self.append_model(model_name)

        self.set_model_id(0)

    def append_model(self, model_name):
        print("loading ", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model_list.append(model)

    def set_model_id(self, model_id):
        self.model = self.model_list[model_id]

    def calc_perplexity(self, tokenized_input):
        ppl_list = []
        for model in self.model_list:
            ppl_list.append(perplexity(model, tokenized_input))
        return np.array(ppl_list)

    def fix_model(self, model_id):
        self.set_model_id(model_id)
        self.fix_mode = True

    def set_flexible_mode(self):
        self.fix_mode = False

    def generate(self, input_ids, attention_mask,
                 **generate_kwargs):

        if not self.fix_mode:
            ppl_array = self.calc_perplexity(input_ids)
            best_model_id = np.where(ppl_array == min(ppl_array))[0][0]
            self.set_model_id(best_model_id)

            if self.verbose:
                print(f"model {best_model_id} will be used")
                print("ppl array: ", ppl_array)

        ret = self.model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  **generate_kwargs)
        return ret


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perplexity(model, tokenized_input) -> torch.Tensor:
    with torch.inference_mode():
        output = model(tokenized_input.to(device), labels=tokenized_input)
    ppl = torch.exp(output.loss)
    return ppl.item()
