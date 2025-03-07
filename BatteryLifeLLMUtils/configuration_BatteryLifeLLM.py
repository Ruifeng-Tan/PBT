import torch
import torch.nn as nn
import transformers
from math import sqrt
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, AutoTokenizer
from transformers import PreTrainedModel
from layers.Embed import PatchEmbeddingTimeLLM
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
from layers.fusion import GatedFusion
from utils.tools import sample_top_p
from typing import List, Literal, Optional, Tuple, TypedDict
from transformers import PretrainedConfig, AutoConfig
from transformers import CONFIG_MAPPING
import argparse
transformers.logging.set_verbosity_error()

class BatteryElectrochemicalConfig(PretrainedConfig):
    model_type = "battery_life"
    def __init__(self, 
                 args,
                 **kwargs):
        super().__init__(**kwargs)
        self.configs = args
    
    def get_configs(self):
        ns = argparse.Namespace(**self.configs)
        return ns


class BatteryLifeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BatteryLifeLLM`]. It is used to instantiate an
    BatteryLifeLLM model according to the specified arguments, defining the model architecture. 

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        ec_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_grid_pinpoints (`List`, *optional*, defaults to `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`):
            A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    Example:

    ```python
    >>> from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> ec_config = BatteryElectrochemicalConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava-Next llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> configuration = LlavaNextConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> model = LlavaNextForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "battery_life"
    is_composition = False

    def __init__(
        self,
        ec_config=None,
        text_config=None,
        ignore_index=-100,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index


        if isinstance(ec_config, dict):
            # ec_config["model_type"] = (
            #     ec_config["model_type"] if "model_type" in ec_config else "battery_life_model"
            # )
            # ec_config = CONFIG_MAPPING[ec_config["model_type"]](**ec_config)
            ec_config = BatteryElectrochemicalConfig(ec_config)
        elif ec_config is None:
            pass

        self.ec_config = ec_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        




