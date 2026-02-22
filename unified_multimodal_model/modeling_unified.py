import json
import os
import shutil
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoModel, AutoProcessor, AutoTokenizer, AutoConfig
from configuration_unified import ManthanM1Config


class ManthanM1(nn.Module):
    """
    Manthan-M1: A unified multimodal model.

    Externally it behaves as one model with:
      - A single HuggingFace model repository
      - A single generate() call
      - A top-level model.safetensors.index.json (so HF shows correct param count)
      - No visible intermediate steps

    Internally it runs a two-stage pipeline:
      Stage 1: VLM (Qwen3-VL-4B) converts image -> structured text
      Stage 2: LLM (gpt-oss-20b) reasons over structured text + user prompt

    The sub-models are stored in their ORIGINAL formats (including MXFP4
    quantization) to avoid weight bloat from dequantization.
    """

    def __init__(self, vlm=None, llm=None, config=None):
        super().__init__()
        self.config = config or ManthanM1Config()
        self.vlm = vlm
        self.llm = llm

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def vlm_system_prompt(self):
        return self.config.vlm_system_prompt

    # ─── Save / Load ─────────────────────────────────────────────────────
    #
    # Repo layout:
    #   Manthan-M1/
    #     config.json                       <- unified Manthan-M1 config
    #     model.safetensors.index.json      <- merged weight map (HF param count)
    #     vlm/                              <- VLM weights + config (BF16, ~8GB)
    #     llm/                              <- LLM weights + config (MXFP4, ~12GB)
    #     vlm_processor/                    <- VLM processor files
    #     llm_tokenizer/                    <- LLM tokenizer files

    def save_pretrained(self, save_directory):
        """Save Manthan-M1 preserving original weight formats."""
        os.makedirs(save_directory, exist_ok=True)

        # 1. Save unified config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": self.config.model_type,
                    "architectures": ["ManthanM1"],
                    "vlm_config": self.config.vlm_config,
                    "llm_config": self.config.llm_config,
                    "vlm_system_prompt": self.config.vlm_system_prompt,
                },
                f,
                indent=2,
            )

        # 2. Save VLM in its native format
        vlm_dir = os.path.join(save_directory, "vlm")
        print(f"Saving VLM to {vlm_dir} ...")
        self.vlm.save_pretrained(vlm_dir, safe_serialization=True)

        # 3. Save LLM in its native format (preserves MXFP4 quantization)
        llm_dir = os.path.join(save_directory, "llm")
        print(f"Saving LLM to {llm_dir} ...")
        self.llm.save_pretrained(llm_dir, safe_serialization=True)

        # 4. Generate top-level model.safetensors.index.json
        #    This merges the weight maps from vlm/ and llm/ so that
        #    HuggingFace Hub correctly reports the total parameter count.
        print("Generating top-level model.safetensors.index.json ...")
        self._generate_merged_index(save_directory)

        print(f"Saved Manthan-M1 to {save_directory}")

    @staticmethod
    def _generate_merged_index(save_directory):
        """
        Build a top-level model.safetensors.index.json that references
        all safetensors files in vlm/ and llm/ subdirectories.
        
        This makes HuggingFace Hub compute the correct total parameter
        count (~24B) for the unified model.
        """
        weight_map = {}
        total_size = 0

        for subdir_name in ("vlm", "llm"):
            subdir = os.path.join(save_directory, subdir_name)
            
            # Check for existing index file first
            index_path = os.path.join(subdir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    sub_index = json.load(f)
                for param_name, filename in sub_index.get("weight_map", {}).items():
                    # Prefix param names and fix file paths to be relative
                    unified_key = f"{subdir_name}.{param_name}"
                    weight_map[unified_key] = f"{subdir_name}/{filename}"
                total_size += sub_index.get("metadata", {}).get("total_size", 0)
            else:
                # Single safetensors file
                st_path = os.path.join(subdir, "model.safetensors")
                if os.path.exists(st_path):
                    with safe_open(st_path, framework="pt") as f:
                        for key in f.keys():
                            unified_key = f"{subdir_name}.{key}"
                            weight_map[unified_key] = f"{subdir_name}/model.safetensors"
                    total_size += os.path.getsize(st_path)

        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }

        index_path = os.path.join(save_directory, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"  Merged index: {len(weight_map)} params, {total_size / (1024**3):.2f} GB")

    @classmethod
    def from_pretrained(cls, save_directory, dtype=torch.bfloat16, device_map="auto"):
        """
        Load Manthan-M1 from a single repository.
        Each sub-model is loaded with its own from_pretrained, which
        correctly handles quantized formats (MXFP4) without dequantizing.
        """
        # 1. Load unified config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            raw = json.load(f)

        config = ManthanM1Config(
            vlm_config=raw["vlm_config"],
            llm_config=raw["llm_config"],
            vlm_system_prompt=raw.get("vlm_system_prompt", ""),
        )

        # 2. Load VLM from its subdirectory (BF16)
        vlm_dir = os.path.join(save_directory, "vlm")
        print(f"Loading VLM from {vlm_dir} ...")
        try:
            vlm = AutoModelForCausalLM.from_pretrained(
                vlm_dir, torch_dtype=dtype, device_map=device_map
            )
        except ValueError:
            vlm = AutoModel.from_pretrained(
                vlm_dir, torch_dtype=dtype, device_map=device_map
            )

        # 3. Load LLM from its subdirectory (preserves MXFP4 if Triton available,
        #    otherwise falls back to BF16 dequant automatically)
        llm_dir = os.path.join(save_directory, "llm")
        print(f"Loading LLM from {llm_dir} ...")
        llm = AutoModelForCausalLM.from_pretrained(
            llm_dir, torch_dtype=dtype, device_map=device_map
        )

        # 4. Build the unified wrapper
        model = cls(vlm=vlm, llm=llm, config=config)
        model.eval()
        print("Manthan-M1 loaded successfully.")
        return model

    # ─── Forward / Generate ──────────────────────────────────────────────

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Manthan-M1 is inference-only. Use model.generate() instead."
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
