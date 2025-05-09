from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class ROMELTIHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    mid_kl_factor: float
    last_kl_factor: float
    nll_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # Midlayer Constraint
    midlayers: List[int]

    # Sentence Transformers
    sentence_model_name: str = None
    top_k: int = 4
