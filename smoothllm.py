import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# torch.set_default_dtype(torch.float64)

def set_determininsm(seed: int) -> None:
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


def logit_entropy(
    logits: torch.Tensor, mult: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return (
        torch.log_softmax(mult * logits, dim=dim)
        * torch.softmax(mult * logits, dim=dim)
    ).sum(dim=dim) * -1.0


def bound_entropy(
    logits: torch.Tensor, entropy_upper_bound: float = 1.0
) -> torch.Tensor:
    shape = logits.shape[0:-1] + (1,)
    multiplier = torch.ones(shape, device=logits.device)
    with torch.no_grad():
        excess_entropy_mask = (
            logit_entropy(logits, multiplier) > entropy_upper_bound
        ).reshape(shape)
        while torch.any(excess_entropy_mask):
            multiplier *= torch.where(excess_entropy_mask, 1.1, 1.0)
            excess_entropy_mask = (
                logit_entropy(logits, multiplier) > entropy_upper_bound
            ).reshape(shape)
    return multiplier * logits

class SmoothGenerationOutput:
    def __init__(
        self,
        model: torch.nn.Module,
        toks: torch.LongTensor,
        tokprobs: torch.Tensor,
        kv_cache: Union[None, torch.Tensor],
    ):
        self.model = model
        self.toks = toks
        self.tokprobs = tokprobs
        self.kv_cache = kv_cache

class SmoothModelForCausalLM(torch.nn.Module):
    model: AutoModelForCausalLM
    embedding_matrix: torch.Tensor

    def __init__(self, model, embedding_matrix):
        super().__init__()
        self.model = model
        self.embedding_matrix = embedding_matrix
        
    def forward(
        self,
        toks: torch.LongTensor,
        tokprobs: torch.Tensor,
        use_cache: bool,
        past_key_values=None,
    ):        
        if use_cache:
            # past_key_values = None
            if past_key_values is None:
                emb = (self.embedding_matrix[toks] * tokprobs.unsqueeze(-1)).sum(axis=-2)
            else:
                emb = (
                    self.embedding_matrix[toks[:, -1, :]]
                    * tokprobs[:, -1, :].unsqueeze(-1)
                ).sum(axis=-2)
            output = self.model(
                inputs_embeds=emb, use_cache=True, past_key_values=past_key_values, output_hidden_states=True
            )
        else:
            emb = (self.embedding_matrix[toks] * tokprobs.unsqueeze(-1)).sum(axis=-2)
            output = self.model(inputs_embeds=emb, output_hidden_states=True)
            output.past_key_values = None

        if past_key_values is None:
            logits = output.logits[:, -1, :]
        else:
            logits = output.logits[:, :]
        kv_cache = output.past_key_values
        

        top_logits, top_tok = torch.topk(logits, k=5)

        top_logits = bound_entropy(top_logits, 1)
        top_probs = F.softmax(top_logits, dim=-1)

        return top_tok, top_probs, kv_cache, logits, emb, output.hidden_states
    
    def generalize_tokens(self, toks, topk=5):
        """
        Transforms a tensor of tokens into a a tensor of generalized tokens
        """
        device = toks.device
        input_ids = toks.clone()
        input_probs = torch.ones_like(
            input_ids, dtype=torch.zeros(1).dtype, device=device
        )

        other_ids = torch.zeros(
            input_ids.shape + (topk - 1,), dtype=input_ids.dtype, device=device
        )
        other_probs = torch.zeros(
            input_ids.shape + (topk - 1,), dtype=input_probs.dtype, device=device
        )
        return torch.cat((input_ids.unsqueeze(-1), other_ids), -1), torch.cat(
            (input_probs.unsqueeze(-1), other_probs), -1
        )

    def generate_double(
      self, input_tokens, max_iters: int
    ):
      """
      Generate smooth text using the model, producing two sequences in parallel:
      one with cache and one without.

      Parameters:
      input_tokens --- input tokens (ordinary or generalized)
      """
      toks, tokprobs = (
          input_tokens
          if isinstance(input_tokens, tuple)
          else self.generalize_tokens(input_tokens, 5)
      )
      cache = None

      # Create copies for the non-cached sequence
      toks_no_cache = toks.clone().detach()
      tokprobs_no_cache = tokprobs.clone().detach()

      for i in range(max_iters):
          with torch.no_grad():
              # Generate one step with cache
              newtok, newprobs, new_cache, logits, emb, hid = self.forward(
                  toks, tokprobs, True, cache
              )
              max_tok = newtok[:, 0]

              # Generate one step without cache
              newtok_no_cache, newprobs_no_cache, _, logits_no_cache, emb_no_cache, hid_no_cache = self.forward(
                  toks_no_cache, tokprobs_no_cache, False, None
              )
              max_tok_no_cache = newtok_no_cache[:, 0]


              if len(emb.shape) == 3:
                print(">>=<<")
                print(emb - emb_no_cache[:, :, :])
                print(logits - logits_no_cache)
              else:                
                print("<<=>>")
                for t, t_no_cache in zip(hid, hid_no_cache):
                    print("-->")
                    print(t - t_no_cache)
                
                print(emb - emb_no_cache[:, -1, :])
                print(logits - logits_no_cache)


              # Update cached sequence
              toks = torch.cat((toks, newtok.unsqueeze(1)), dim=1)
              tokprobs = torch.cat((tokprobs, newprobs.unsqueeze(1)), dim=1)
              cache = new_cache

              # Update non-cached sequence
              toks_no_cache = torch.cat(
                  (toks_no_cache, newtok_no_cache.unsqueeze(1)), dim=1
              )
              tokprobs_no_cache = torch.cat(
                  (tokprobs_no_cache, newprobs_no_cache.unsqueeze(1)), dim=1
              )

      res_cached = SmoothGenerationOutput(
          model=self,
          toks=toks,
          tokprobs=tokprobs,
          kv_cache=cache,
      )

      res_no_cache = SmoothGenerationOutput(
          model=self,
          toks=toks_no_cache,
          tokprobs=tokprobs_no_cache,
          kv_cache=None,
      )

      return res_cached, res_no_cache

