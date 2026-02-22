from transformers import PretrainedConfig


class ManthanM1Config(PretrainedConfig):
    model_type = "manthan_m1"

    def __init__(
        self,
        vlm_config=None,
        llm_config=None,
        vlm_system_prompt="Analyze this image comprehensively. Describe the main subjects, background, text (OCR), spatial relationships, colors, and any notable details. Output in a structured format.",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vlm_config = vlm_config
        self.llm_config = llm_config
        self.vlm_system_prompt = vlm_system_prompt
