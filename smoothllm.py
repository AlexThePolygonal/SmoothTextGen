import numpy as np
import random
import torch
import torch.nn.functional as ftorch
import copy
import warnings
from typing import Tuple, List
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def set_determininsm(seed : int) -> None:
    """
    Set deterministic execution of all libraries used in this proj
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def save_random_state(device):
   res = {}
   res["torch"] = torch.get_rng_state()
   if device.type == "cuda":
      res[torch.cuda.get_device_name(device)] = torch.cuda.get_rng_state(device)
   res["np"] = np.random.get_state()
   res["random"] = random.getstate()

def load_random_state(res, device):
#    if not res:
#       return
   torch.set_rng_state(res["torch"])
   if device.type == "cuda":
      res[torch.cuda.get_device_name(device)] = torch.cuda.set_rng_state(device)
   np.random.set_state(res['np'])
   random.setstate(res["random"])
   
   

# def logit_entropy(logits, dim=-1):
#   """
#   Compute the Shannon entropy of the probas given by the logits
#   """
#   return (ftorch.log_softmax(logits, dim=dim) * ftorch.softmax(logits, dim=dim)).sum(dim=dim) * -1. * (1 / torch.log(2.))

# def prob_entropy(probs, dim=-1):
#   """
#   Compute the Shannon entropy
#   """
#   return (probs * torch.log(probs)).sum(dim=dim) * (1. / torch.log(2.))


def logit_entropy(logits: torch.Tensor, mult: torch.Tensor, dim: int = -1) -> torch.Tensor:
    '''
    Calculate the entropy of logits.

    Args:
        logits (Tensor): The input logits.
        mult (Tensor): The multiplier.
        dim (int, optional): The dimension along which to compute the entropy. Defaults to -1.

    Returns:
        Tensor: The entropy of the logits.
    '''
    return (torch.log_softmax(mult * logits, dim=dim) * torch.softmax(mult * logits, dim=dim)).sum(dim=dim) * -1.

def entropy_bounded_softmax(logits: torch.Tensor, entropy_upper_bound: float = 1., temperature_step_factor: float = 1.1) -> torch.Tensor:
    '''
    Returns the softmax of logits with temperature chosen s.t. that entropy < entropy_upper_bound.

    Args:
        logits (Tensor): The input logits.
        entropy_upper_bound (float, optional): The upper bound of entropy. Defaults to 1.
        temperature_step_factor (float, optional): The step factor for adjusting temperature. Defaults to 1.1.

    Returns:
        Tensor: The bounded softmax output.
    '''
    shape = logits.shape[0:-1] + (1,)
    multiplier = torch.ones(shape, device=logits.device)
    with torch.no_grad():
        excess_entropy_mask = (logit_entropy(logits, multiplier) > entropy_upper_bound).reshape(shape)
        while torch.any(excess_entropy_mask):
            multiplier *= torch.where(excess_entropy_mask, temperature_step_factor, 1.)
            excess_entropy_mask = (logit_entropy(logits, multiplier) > entropy_upper_bound).reshape(shape)
    return torch.softmax(multiplier * logits, dim=-1)

def maxprob_bounded_softmax(logits: torch.Tensor, minimal_maxprob: float = 0.4, temperature_step_factor: float = 1.1) -> torch.Tensor:
    '''
    Returns the softmax of logits with temperature chosen s.t. that max p_i > minimal_maxprob.

    Args:
        logits (Tensor): The input logits.
        minimal_maxprob (float, optional): The minimal maximum probability. Defaults to 0.4.
        temperature_step_factor (float, optional): The step factor for adjusting temperature. Defaults to 1.1.

    Returns:
        Tensor: The bounded softmax output.
    '''
    shape = logits.shape[0:-1] + (1,)
    multiplier = torch.ones(shape, device=logits.device)
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        probs_mask = probs.max(dim=-1, keepdim=True)[0] < minimal_maxprob
        while torch.any(probs_mask):
            multiplier *= torch.where(probs_mask, temperature_step_factor, 1.)
            probs = torch.softmax(multiplier * logits, dim=-1)
            probs_mask = probs.max(dim=-1, keepdim=True)[0] < minimal_maxprob
    return torch.softmax(multiplier * logits, dim=-1)


def ban_repeat_ngrams(
        logits: torch.Tensor, 
        toks: torch.LongTensor, 
        tokprobs: torch.Tensor, 
        d: dict, 
        no_repeat_ngram_size: int = 6
    ):
    batch_size, seq_length, topk = toks.shape

    # Iterate over each sequence in the batch
    for batch_idx in range(batch_size):
        forbidden_toks_d = d.get(batch_idx, {})
        input_toks = toks[batch_idx, :, 0]
        input_logits = logits[batch_idx]
        cur_ngram_prefix = None
        banned_toks = []
        if seq_length > no_repeat_ngram_size:
            cur_ngram_prefix = tuple(input_toks[-no_repeat_ngram_size + 1:].cpu().tolist())
            banned_toks = forbidden_toks_d.get(cur_ngram_prefix, [])
        for banned_tok in banned_toks:
            input_logits[banned_tok] = -1000

    return logits

def update_banned_toks(
        logits: torch.Tensor, 
        toks: torch.LongTensor,
        tokprobs: torch.Tensor, 
        d: dict,
        chosen_toks : torch.Tensor,
        no_repeat_ngram_size: int = 6
        ):
    res_d = copy.deepcopy(d)
    batch_size, seq_length, topk = toks.shape
    
    for batch_idx in range(batch_size):
        forbidden_toks_d = res_d.get(batch_idx, {})
        input_toks = toks[batch_idx, :, 0]
        input_tok = chosen_toks[batch_idx, 0]
        if (seq_length >= no_repeat_ngram_size):
            cur_ngram_prefix = tuple(input_toks[-no_repeat_ngram_size + 1:].cpu().tolist())
            cur_banned = forbidden_toks_d.get(cur_ngram_prefix, [])
            cur_banned.append(input_tok)
            forbidden_toks_d[cur_ngram_prefix] = cur_banned
            res_d[batch_idx] =  forbidden_toks_d
    return res_d
 

# class SmoothGenerationOutput():
#   model : nn.Module
#   toks : torch.LongTensor
#   tokprobs : torch.Tensor
#   choose_tok : ChooseTokenTopk
#   parameters : List



class SmoothGenerationConfig():
   use_kv_cache = False
   is_initial = True
   random_state : dict
   eos_token_id = 0

   ban_repeat_ngrams = True
   no_repeat_ngram_size = 6
   banned_ngram_dict = {}

   topk = 5
   logits_to_probs = lambda _, x : maxprob_bounded_softmax(x, minimal_maxprob=0.7, temperature_step_factor=1.1)
   do_hard_rounding = False

   do_sampling = True
   sampling_fudge_factor = 0.

   def __init__(self):
      pass

class SmoothGenerationOutput():
  model : torch.nn.Module
  toks : torch.LongTensor
  tokprobs : torch.Tensor
  parameters : List[SmoothGenerationConfig]

class SmoothModelForCausalLM(torch.nn.Module):
  """
  Base class for smooth text generation
  Transforms a autoregressive text generation model to a smooth text generation model
  """
  model : AutoModelForCausalLM
  embedding_matrix : torch.Tensor

  def __init__(self, model, embedding_matrix):
    """
    Transform a model into a smooth model

    model --- the base model
    embedding_matrix --- the embedding matrix of the base model
    """
    super().__init__()
    self.model = model
    self.embedding_matrix = embedding_matrix

  def call_model(self, toks, tokprobs):
    emb = (self.embedding_matrix[toks] * tokprobs.unsqueeze(-1)).sum(axis=-2)
    return self.model(inputs_embeds = emb)
  
  def forward(self, toks, tokprobs, old_config : SmoothGenerationConfig):
    config = copy.deepcopy(old_config)
    # Generate
    logits = self.call_model(toks, tokprobs).logits[:, -1, :]
    # add Gumbel(0,1) divided by the fudge factor
    if config.do_sampling:
      logits += torch.distributions.Gumbel(0,1).sample(logits.shape).to(logits.device) * config.sampling_fudge_factor

    if config.ban_repeat_ngrams:
        logits = ban_repeat_ngrams(logits, toks, tokprobs, config.banned_ngram_dict, config.no_repeat_ngram_size)
    
    top_logits, top_tok = torch.topk(logits, k=config.topk)

    if config.ban_repeat_ngrams:
       config.banned_ngram_dict = update_banned_toks(logits, toks, tokprobs, config.banned_ngram_dict, top_tok, config.no_repeat_ngram_size)
    
    logits_to_probs = config.logits_to_probs
    top_probs = logits_to_probs(top_logits)

    if config.do_hard_rounding:
       hard_rounded = torch.zeros_like(top_probs, device=top_probs.device)
       hard_rounded[:, 0] = 1.
       top_probs = hard_rounded - top_probs.detach() + top_probs

    return top_tok, top_probs, config

  def generalize_tokens(self, toks, topk):
    """
    Transforms a tensor of tokens into a a tensor of generalized tokens
    """
    device = toks.device
    input_ids = toks
    input_probs = torch.ones_like(input_ids, dtype=torch.zeros(1).dtype, device=device)

    other_ids = torch.zeros(input_ids.shape + (topk - 1,), dtype = input_ids.dtype, device=device)
    other_probs = torch.zeros(input_ids.shape + (topk - 1,), dtype = input_probs.dtype, device=device)
    return torch.cat((input_ids.unsqueeze(-1), other_ids), -1), torch.cat((input_probs.unsqueeze(-1), other_probs), -1)
  
  def generate(self, input_tokens, max_iters: int, config: SmoothGenerationConfig):
    """
    Generate smooth text using the model

    Parameters:
    input_tokens --- input tokens (ordinary or generalized)
    """
    toks, tokprobs = None, None
    if type(input) is tuple:
      toks, tokprobs = input_tokens
    else:
      toks, tokprobs = self.generalize_tokens(input_tokens, config.topk)

    # config.random_state = save_random_state(input_tokens.device)


    params_seq = [copy.deepcopy(config) for i in range(toks.shape[1] + 1)]
    init_len = toks.shape[1]
    for i in range(max_iters):
      with torch.no_grad():
        # Global parameters
        params_seq[-1].is_initial = False

        # Save the random states
        params_seq[-1].random_state = save_random_state(toks.device)
        
        # Generate one step
        newtok, newprobs, newparams = self.forward(toks, tokprobs, params_seq[-1])
        max_tok = newtok[:, 0]

        # save & update
        params_seq.append(newparams)
        toks = torch.cat((toks, newtok.unsqueeze(1)), dim=1)
        tokprobs = torch.cat((tokprobs, newprobs.unsqueeze(1)), dim=1)
        if (max_tok == config.eos_token_id):
          break

    res = SmoothGenerationOutput()
    res.model = self
    res.toks = toks
    res.tokprobs = tokprobs
    res.parameters = params_seq
    return res
  

class SmoothLoss():
  loss = None

  class LossValue():
    model_output : SmoothGenerationOutput
    loss = None

    def __init__(self, output, loss):
      self.model_output = output
      self.loss = loss

    def value(self):
      return self.loss(self.model_output.toks, self.model_output.tokprobs).item()

    def backwards(self):
      """
      The key function of this project, efficient backprop for stacked models
      """
      model = self.model_output.model
      parameters = self.model_output.parameters
      toks = self.model_output.toks
      tokprobs = self.model_output.tokprobs

      # compute (∂𝓛/∂τ_i)
      # we store the intermediate grads in tokprobs
      tokprobs.requires_grad_()
      loss_val = self.loss(toks, tokprobs)
      loss_val.backward()
      init_grad = tokprobs.grad.clone().detach()

      # compute the range
      batch_size, toks_len, topk = toks.shape
      init_len = 0
      while parameters[init_len].is_initial:
        init_len = init_len + 1

      # update ∂𝓛/∂τ_j ← ∂𝓛/∂τ_j + ∂τ_i(τ_1 … τ_i-1)/∂τ_j for all j < i
      for i in reversed(range(init_len, toks_len - 1)):
        cur_toks = toks[:, :i, :]
        cur_tokprobs = tokprobs[:, :i, :]

        # restore the original random state for reproducibility
        load_random_state(parameters[i].random_state, toks.device)

        # re-generate the output
        newtok, newprobs, _ = self.model_output.model.forward(cur_toks, cur_tokprobs, self.model_output.choose_tok, parameters[i])

        # check that the re-generated output is the same
        if not torch.equal(newtok, toks[:, i, :]):
          print(f"At position {i}")
          print("Re-generated tokens:", newtok, newprobs)
          print("Original tokens:", toks[:, i, :], tokprobs[:, i, :])
        assert torch.equal(newtok, toks[:, i, :])

        # propagate the gradients
        last_grad = tokprobs.grad[:, i, :]
        newprobs.backward(last_grad)

      tokprobs.requires_grad_(False)


  def __init__(self, loss):
    super().__init__()
    self.loss = loss

  def __call__(self, output):
    return self.LossValue(output, self.loss)