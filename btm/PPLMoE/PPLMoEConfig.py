from transformers import PretrainedConfig
from typing import List


class PPLMoEConfig(PretrainedConfig):
    model_type = "moewrapper"
    model_list = [
        "kanhatakeyama/01b_model_30b_token",
        "kanhatakeyama/01b_model_30b_token",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
