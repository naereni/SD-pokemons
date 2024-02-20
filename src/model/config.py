from dataclasses import dataclass

"""To see help about args see link
https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py#:~:text=README.md%22%0A%0A%0Adef-parse_args-%3A%0A%20%20%20%20parser%20%3D%20argparse.ArgumentParser"""


@dataclass
class Config:
    # define args
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    train_data_dir: str = "src/data/dataset"
    validation_prompt: str = (
        "highres, masterpiece, sugimori ken style, pokemon creature, without a background, "
    )
    output_dir: str = "src/model/LoRA-pokemons-weights"
    seed: int = 515786
    resolution: int = 512
    random_flip: bool = True
    train_batch_size: int = 1
    max_train_steps: int = 15000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    mixed_precision: str = None
    report_to: str = "wandb"
    enable_xformers_memory_efficient_attention: bool = False
    max_grad_norm: int = 1
    lr_scheduler: int = "cosine"
    checkpointing_steps: int = 500
    lr_warmup_steps: int = 0
    num_inference_steps: int = 30

    # defaut args
    image_column: str = "image"
    caption_column: str = "text"
    num_validation_images: int = 4
    validation_epochs: int = 1
    num_train_epochs: int = 100
    dataloader_num_workers: int = 4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    logging_dir: str = "logs"
    noise_offset: float = 0
    local_rank: int = -1
    rank: int = 4

    # default false args
    allow_tf32: bool = False
    center_crop: bool = False
    gradient_checkpointing: bool = False
    scale_lr: bool = False

    # None args
    infirence_seed: int = None
    snr_gamma: float = None
    cache_dir: str = None
    max_train_samples: int = None
    dataset_config_name: str = None
    variant: str = None
    revision: str = None
    dataset_name: str = None
    prediction_type: str = None
    resume_from_checkpoint: str = None
    checkpoints_total_limit: int = None


config = Config()
