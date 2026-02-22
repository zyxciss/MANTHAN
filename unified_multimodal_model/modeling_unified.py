import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel, AutoProcessor, AutoTokenizer, AutoConfig
from configuration_unified import UnifiedMultimodalConfig

class UnifiedMultimodalModel(PreTrainedModel):
    config_class = UnifiedMultimodalConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize VLM (Qwen3-VL-4B-Instruct)
        # Note: In a real scenario, you might load these from sub-configs or paths
        # For packaging, we assume the weights are stored in the same repo or downloaded
        vlm_cfg = AutoConfig.for_model(**config.vlm_config) if config.vlm_config else None
        
        # Qwen-VL models are typically loaded with AutoModelForCausalLM in transformers
        # If it fails with AutoModelForCausalLM, we can try AutoModel
        try:
            self.vlm = AutoModelForCausalLM.from_config(vlm_cfg) if vlm_cfg else None
        except ValueError:
            self.vlm = AutoModel.from_config(vlm_cfg) if vlm_cfg else None
        
        # Initialize LLM (gpt-oss-20b)
        llm_cfg = AutoConfig.for_model(**config.llm_config) if config.llm_config else None
        self.llm = AutoModelForCausalLM.from_config(llm_cfg) if llm_cfg else None
        
        self.vlm_system_prompt = config.vlm_system_prompt

    @property
    def _tied_weights_keys(self):
        # Return an empty list or the tied weights of sub-modules if any
        keys = []
        if self.vlm and hasattr(self.vlm, "_tied_weights_keys"):
            keys.extend([f"vlm.{k}" for k in self.vlm._tied_weights_keys])
        if self.llm and hasattr(self.llm, "_tied_weights_keys"):
            keys.extend([f"llm.{k}" for k in self.llm._tied_weights_keys])
        return keys

    @property
    def all_tied_weights_keys(self):
        return self._tied_weights_keys

    def forward(self, *args, **kwargs):
        # Forward pass is usually for training, which we aren't doing.
        # We can implement a dummy forward or raise NotImplementedError
        raise NotImplementedError("This model is designed for inference via generate() only.")

    @torch.no_grad()
    def generate(
        self,
        images,
        text_prompt,
        vlm_processor,
        llm_tokenizer,
        max_new_tokens=512,
        **kwargs
    ):
        """
        Unified generate method.
        1. VLM processes image -> structured text
        2. LLM processes structured text + user prompt -> final response
        """
        device = self.device
        
        # --- Stage 1: VLM Image Understanding ---
        # Construct prompt for VLM to get a comprehensive description
        vlm_messages = [
            {"role": "system", "content": self.vlm_system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": images},
                {"type": "text", "text": "Describe this image in detail."}
            ]}
        ]
        
        vlm_text = vlm_processor.apply_chat_template(vlm_messages, tokenize=False, add_generation_prompt=True)
        vlm_inputs = vlm_processor(text=[vlm_text], images=[images], padding=True, return_tensors="pt").to(device)
        
        # Generate description
        vlm_outputs = self.vlm.generate(**vlm_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(vlm_inputs.input_ids, vlm_outputs)
        ]
        image_description = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # --- Stage 2: LLM Reasoning ---
        # Construct prompt for LLM combining the image description and the user's actual prompt
        llm_prompt = f"""<|im_start|>system
You are a highly capable AI assistant. You have been provided with a detailed description of an image. Use this description to answer the user's request.
<|im_end|>
<|im_start|>user
[Image Description Provided by Vision Module]
{image_description}

[User Request]
{text_prompt}
<|im_end|>
<|im_start|>assistant
"""
        
        llm_inputs = llm_tokenizer(llm_prompt, return_tensors="pt").to(device)
        
        # Generate final response
        llm_outputs = self.llm.generate(
            **llm_inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        final_response = llm_tokenizer.decode(
            llm_outputs[0][llm_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return final_response
