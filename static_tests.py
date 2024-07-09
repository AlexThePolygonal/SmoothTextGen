import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_printoptions(precision=6)


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import peft

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


from smoothllm import *
from gptneo_decompose import GradmodGPTNeoAttn, UngradmodGPTNeoAttn

set_determininsm(42)

base_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M').to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

peft_config = peft.LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.2, inference_mode=False, task_type="CAUSAL_LM"
)
finetune_base_model = peft.get_peft_model(base_model, peft_config)

male_toks = [tokenizer.encode(word, return_tensors="pt").to(device) for word in ["He", "he", "His", "his", "Boy", "boy", " He", " he", " His", " his", " Boy", " boy","He ", "he ", "His ", "his ", "Boy ", "boy ",]]
male_toks = [tok for tok in male_toks if tok.shape[1] == 1]
female_toks = [tokenizer.encode(word, return_tensors="pt").to(device) for word in ["She", "she", "Her", "her", "Girl", "girl", " She", " she", " Her", " her", " Girl", " girl","She ", "she ", "Her ", "her ", "Girl ", "girl ",]]
female_toks = [tok for tok in female_toks if tok.shape[1] == 1]



def remove_token_loss(toks, tokprobs, list_of_toks = male_toks):
  mask = torch.eq(toks, list_of_toks[0])
  for tok in list_of_toks:
    mask = torch.logical_or(mask, torch.eq(toks, tok))
  return ((tokprobs) * mask).sum(dim = -1).sum(dim=-1)

def llm_ratio(toks):
    llm_rl = F.log_softmax(base_model(toks)[0], dim=-1)[0, torch.arange(toks.shape[1]), toks[0]].sum()  # Log-likelihood of the sequence under finetuned base model
    llm_sft = F.log_softmax(finetune_base_model(toks)[0], dim=-1)[0, torch.arange(toks.shape[1]), toks[0]].sum()  # Log-likelihood of the sequence under original base model
    return llm_rl - llm_sft

def rhlf_loss(toks, tokprobs):
   return remove_token_loss(toks, tokprobs, male_toks) - remove_token_loss(toks, tokprobs, female_toks)  - llm_ratio(toks[:, :, 0]) 

def compare_generation_test():
    print("Running generation correstness test")
    model = SmoothModelForCausalLM(
        base_model, 
        base_model.get_input_embeddings().weight,
        GradmodGPTNeoAttn,
        UngradmodGPTNeoAttn
    )

    smooth_config = SmoothGenerationConfig()
    smooth_config.eos_token_id = tokenizer.eos_token_id
    smooth_config.do_sampling = False
    smooth_config.use_kv_cache = False
    smooth_config.do_hard_rounding = True
    smooth_config.ban_repeat_ngrams = False

    set_determininsm(42)
    base_tokens = tokenizer.encode("One ", return_tensors="pt").to(device)
    cachefree_output = model.generate(base_tokens, 20, smooth_config)

    set_determininsm(42)
    smooth_config.use_kv_cache = True
    cache_output = model.generate(base_tokens, 20, smooth_config)

    assert(cache_output.toks.allclose(cachefree_output.toks))
    assert(cache_output.tokprobs.allclose(cachefree_output.tokprobs))

    smooth_config.do_sampling = True
    smooth_config.do_hard_rounding = False
    smooth_config.sampling_temp = 0.2

    set_determininsm(42)
    base_tokens = tokenizer.encode("One ", return_tensors="pt").to(device)
    cachefree_output = model.generate(base_tokens, 20, smooth_config)

    set_determininsm(42)
    smooth_config.use_kv_cache = True
    cache_output = model.generate(base_tokens, 20, smooth_config)

    assert(cache_output.toks.allclose(cachefree_output.toks))
    assert(cache_output.tokprobs.allclose(cachefree_output.tokprobs))

    print("Running generation correstness test: OK")
    
def compare_determinism_test():
    finetune_model = SmoothModelForCausalLM(
        finetune_base_model, 
        base_model.get_input_embeddings().weight,
        GradmodGPTNeoAttn,
        UngradmodGPTNeoAttn,
    )
    optimizer = torch.optim.Adam(finetune_model.model.parameters(), 2e-3)

    smooth_config = SmoothGenerationConfig()
    smooth_config.eos_token_id = tokenizer.eos_token_id
    smooth_config.do_sampling = False
    smooth_config.use_kv_cache = False
    smooth_config.do_hard_rounding = True
    smooth_config.ban_repeat_ngrams = False

    








compare_generation_test()

