""" 
Transformer Basic config
"""

from default_config import basic_cfg


cfg = basic_cfg
cfg.use_model = 'transformer'

cfg.custom_model_params = {
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_token_embedding": False,
    "attention_hidden_sizes": 32,
    "num_heads": 1,
    "attention_dropout": 0.0,
    "ffn_hidden_sizes": 32,
    "ffn_filter_sizes": 32,
    "ffn_dropout": 0.0,
    "scheduler_sampling": 1,  # 0 means teacher forcing, 1 means use last prediction
    "skip_connect_circle": False,
    "skip_connect_mean": False
}