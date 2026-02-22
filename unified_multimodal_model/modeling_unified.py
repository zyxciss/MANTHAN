import json
import os
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoModel, AutoProcessor, AutoTokenizer, AutoConfig
from configuration_unified import UnifiedMultimodalConfig


class UnifiedMultimodalModel(nn.Module):
    """
    A unified multimodal model that wraps a VLM and an LLM as a single nn.Module.
    
    Externally it behaves as one model with:
      - A single combined state_dict
      - A single .safetensors checkpoint
      - A single generate() call
    
    Internally it runs a two-stage pipeline:
      Stage 1: VLM (Qwen3-VL-4B) converts image -> structured text
      Stage 2: LLM (gpt-oss-20b) reasons over structured text + user prompt
    """

    def __init__(self, vlm=None, llm=None, config=None):
        super().__init__()
        self.config = config or UnifiedMultimodalConfig()
        # Register sub-models as submodules so their params appear in state_dict
        self.vlm = vlm
        self.llm = llm

    @property
    def device(self):
        """Return device of the first parameter found."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def vlm_system_prompt(self):
        return self.config.vlm_system_prompt

    # ─── Save / Load (combined single checkpoint) ───────────────────────

    def save_pretrained(self, save_directory):
        """
        Save the unified model as a single artifact:
          - config.json           (unified config with both sub-model configs)
          - model.safetensors     (combined state_dict of vlm + llm)
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1. Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_type": self.config.model_type,
                "vlm_config": self.config.vlm_config,
                "llm_config": self.config.llm_config,
                "vlm_system_prompt": self.config.vlm_system_prompt,
            }, f, indent=2)

        # 2. Collect the full state_dict (vlm.* + llm.* keys)
        #    Move everything to CPU for serialisation
        state_dict = {k: v.cpu().contiguous() for k, v in self.state_dict().items()}

        # 3. Save as a single safetensors file
        safetensors_path = os.path.join(save_directory, "model.safetensors")
        print(f"Saving {len(state_dict)} tensors to {safetensors_path} ...")
        save_file(state_dict, safetensors_path)

        print(f"Saved unified model to {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory, dtype=torch.bfloat16, device_map="auto"):
        """
        Load the unified model from a single repository:
          1. Read config.json to find sub-model architectures
          2. Instantiate empty sub-models from their configs
          3. Load the combined state_dict from model.safetensors
        """
        # 1. Load config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            raw = json.load(f)

        config = UnifiedMultimodalConfig(
            vlm_config=raw["vlm_config"],
            llm_config=raw["llm_config"],
            vlm_system_prompt=raw.get("vlm_system_prompt", ""),
        )

        # 2. Create empty sub-models from their configs (no weights yet)
        vlm_cfg = AutoConfig.for_model(**config.vlm_config)
        vlm_cfg.torch_dtype = dtype
        try:
            vlm = AutoModelForCausalLM.from_config(vlm_cfg)
        except ValueError:
            vlm = AutoModel.from_config(vlm_cfg)

        llm_cfg = AutoConfig.for_model(**config.llm_config)
        llm_cfg.torch_dtype = dtype
        llm = AutoModelForCausalLM.from_config(llm_cfg)

        # 3. Build the unified wrapper (still on CPU, random weights)
        model = cls(vlm=vlm, llm=llm, config=config)

        # 4. Load the combined safetensors checkpoint
        safetensors_path = os.path.join(save_directory, "model.safetensors")
        print(f"Loading weights from {safetensors_path} ...")
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)

        # 5. Cast to desired dtype and move to device
        model = model.to(dtype=dtype)
        if device_map == "auto" and torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        print("Unified model loaded successfully.")
        return model

    # ─── Forward / Generate ──────────────────────────────────────────────

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This model is inference-only. Use model.generate() instead."
        )

    @torch.no_grad()
    def generate(
        self,
        images,
        text_prompt,
        vlm_processor,
        llm_tokenizer,
        max_new_tokens=512,
        **kwargs,
    ):
        """
        Unified generate method.
          1. VLM processes image -> structured text description
          2. LLM processes description + user prompt -> final response
        """
        # Determine device from the VLM
        device = next(self.vlm.parameters()).device

        # ── Stage 1: VLM Image Understanding ──────────────────────────
        vlm_messages = [
            {"role": "system", "content": self.vlm_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]

        vlm_text = vlm_processor.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True
        )
        vlm_inputs = vlm_processor(
            text=[vlm_text], images=[images], padding=True, return_tensors="pt"
        ).to(device)

        vlm_outputs = self.vlm.generate(**vlm_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(vlm_inputs.input_ids, vlm_outputs)
        ]
        image_description = vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"[VLM Description] {image_description[:200]}...")

        # ── Stage 2: LLM Reasoning ───────────────────────────────────
        llm_device = next(self.llm.parameters()).device

        llm_prompt = (
            "<|im_start|>system\n"
            "You are a highly capable AI assistant. You have been provided "
            "with a detailed description of an image. Use this description "
            "to answer the user's request.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "[Image Description]\n"
            f"{image_description}\n\n"
            "[User Request]\n"
            f"{text_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        llm_inputs = llm_tokenizer(llm_prompt, return_tensors="pt").to(llm_device)

        llm_outputs = self.llm.generate(
            **llm_inputs, max_new_tokens=max_new_tokens, **kwargs
        )

        final_response = llm_tokenizer.decode(
            llm_outputs[0][llm_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return final_response
