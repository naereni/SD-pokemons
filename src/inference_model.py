from diffusers import DiffusionPipeline
from config import config
import torch


class InferenceModel:
    def __init__(self, device=None) -> None:
        self.device = device

        weight_dtype = torch.float32
        if config.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif config.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.pipeline = DiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            revision=config.revision,
            variant=config.variant,
            torch_dtype=weight_dtype,
        )
        if device is None:
            self.device = "mps"  # "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = self.pipeline.to(device)
        # load attention processors
        self.pipeline.load_lora_weights(
            config.output_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        # run inference
        self.generator = torch.Generator(device)
        if config.seed is not None:
            self.generator = self.generator.manual_seed(config.seed)

    def generate(self, num_images=1):
        images = []
        with torch.cuda.amp.autocast():
            for _ in range(num_images):
                images.append(
                    self.pipeline(
                        config.validation_prompt,
                        num_inference_steps=config.num_inference_steps,
                        generator=self.generator,
                    ).images[0]
                )
        return images
