from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RulePromptArguments:

    device: str = field(default="cuda")
    data_dir: str = field(default=None, metadata={"help": "Path to data directory"})
    cache_dir: str = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
    dataset: str = field(default=None, metadata={"help": "choose datasets"})
    model: str = field(default="roberta")
    model_name_or_path: str = field(default="roberta-large")
    nocut: bool = field(default=False)
    max_len: int = field(default=150)

    # For rule mining and pseudo label generation
    select: int = field(default=12, metadata={"help": "EmbVerbalizer select num"})
    support: float = field(default=0.1)
    num_verbalizers: int = field(default=5)
    max_text_length: int = field(default=128, metadata={"help": "SimCSE max text length"})
    simcse_model_name_short: str = field(default=None)
    simcse_model_name_or_path: str = field(default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})

    # For training
    num_iterations: int = field(default=3, metadata={"help": "Number of iterations"})
    epochs: int = field(default=7)
    learning_rate: float = field(default=1e-8)
    proportion_ft: float = field(default=0.85)
    batch_s: int = field(default=64)
    pooling: Optional[str]=field(default="pooler", metadata={"help": "Pooling method"})