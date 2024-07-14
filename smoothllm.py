import numpy as np
import random
import torch
import torch.nn.functional as F
import copy
import warnings
from typing import Tuple, List, Union, Callable, Optional, TypeAlias
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from copy import deepcopy
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped


def total_grad(model: torch.nn.Module) -> torch.Tensor:
    """
    Get the total gradient of all the free parameters of the model
    """
    return torch.cat(
        [torch.flatten(p.grad) for p in model.parameters() if p.requires_grad]
    ).view(-1, 1)


# Add the gradients taken by the hook and add it into the flow
class GradMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hook):
        ctx.hook = hook
        return input

    @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(ctx, grad_output):
        temp = ctx.hook(grad_output)
        return grad_output + temp, None


class GradModded(torch.nn.Module):
    base_layer: torch.nn.Module
    gradmod: GradMod
    hook: Callable

    def __init__(self, base: torch.nn.Module, hook: Callable):
        super().__init__()
        self.base_layer = base
        self.gradmod = GradMod()
        self.hook = hook

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gradmod.apply(self.base_layer(input), self.hook)


def set_determininsm(seed: int) -> None:
    """
    Set deterministic execution for all libraries used in this project
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def save_random_state(device: torch.device) -> dict:
    """
    Save the random state to ensure perfect reproducibility
    """
    res = {}
    res["torch"] = torch.get_rng_state()
    if device.type == "cuda":
        res[torch.cuda.get_device_name(device)] = torch.cuda.get_rng_state(device)
    res["np"] = np.random.get_state()
    res["random"] = random.getstate()
    return res


def load_random_state(res: dict, device: torch.device) -> None:
    """
    Load the random state to exactly recompute the previous results
    """
    torch.set_rng_state(res["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(res[torch.cuda.get_device_name(device)], device)
    np.random.set_state(res["np"])
    random.setstate(res["random"])


def logit_entropy(
    logits: torch.Tensor, mult: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Calculate the entropy of logits.

    Args:
        logits (Tensor): The input logits, [..., dict_size]
        mult (Tensor): The multiplier, having a shape which is broadcastable to the logit shape
        dim (int, optional): The dimension along which to compute the entropy. Defaults to -1.

    Returns:
        Tensor: The entropy of the logits.
    """
    return (
        torch.log_softmax(mult * logits, dim=dim)
        * torch.softmax(mult * logits, dim=dim)
    ).sum(dim=dim) * -1.0


def bound_entropy(
    logits: torch.Tensor, entropy_upper_bound: float = 1.0
) -> torch.Tensor:
    """
    Returns the softmax of logits with temperature chosen s.t. that entropy < entropy_upper_bound.

    Args:
        logits (torch.Tensor): The input logits, of shape [..., dict_size]
        entropy_upper_bound (float, optional): The upper bound of entropy. Defaults to 1.

    Returns:
        torch.Tensor: The bounded softmax output.
    """
    shape = logits.shape[0:-1] + (1,)  # of shape [..., 1]
    multiplier = torch.ones(shape, device=logits.device)
    with torch.no_grad():
        excess_entropy_mask = (
            logit_entropy(logits, multiplier) > entropy_upper_bound
        ).reshape(
            shape
        )  # of shape [..., 1]
        while torch.any(excess_entropy_mask):
            multiplier *= torch.where(excess_entropy_mask, 1.1, 1.0)
            excess_entropy_mask = (
                logit_entropy(logits, multiplier) > entropy_upper_bound
            ).reshape(shape)
    return multiplier * logits


def ban_repeat_ngrams(
    prev_tokens: torch.LongTensor, no_repeat_ngram_size: int = 6
) -> List[List[int]]:
    """
    Identify tokens to be banned based on previous tokens to prevent repeated n-grams.

    Args:
        prev_tokens (torch.LongTensor): The previously generated tokens, shape (batch_size, seq_length).
        no_repeat_ngram_size (int, optional): The size of n-grams to consider. Defaults to 6.

    Returns:
        List[List[int]]: A list of lists containing banned token ids for each sequence in the batch.
    """
    batch_size, seq_length = prev_tokens.shape
    banned_tokens = [[] for _ in range(batch_size)]

    # Only check for n-grams if we have enough tokens
    if seq_length >= no_repeat_ngram_size:
        for batch_idx in range(batch_size):
            # Get the last n-1 tokens
            last_tokens = prev_tokens[batch_idx, -(no_repeat_ngram_size - 1) :].tolist()

            # Check for matching n-grams in the sequence
            for start_idx in range(seq_length - no_repeat_ngram_size + 1):
                ngram = prev_tokens[
                    batch_idx, start_idx : start_idx + no_repeat_ngram_size - 1
                ].tolist()
                if ngram == last_tokens:
                    # If we find a match, ban the token that follows this n-gram
                    banned_token = prev_tokens[
                        batch_idx, start_idx + no_repeat_ngram_size - 1
                    ].item()
                    banned_tokens[batch_idx].append(banned_token)

    return banned_tokens


class SmoothGenerationConfig:
    def __init__(
        self,
        use_kv_cache: bool = True,
        eos_token_id: int = 0,
        ban_repeat_ngrams: bool = False,
        no_repeat_ngram_size: int = 6,
        topk: int = 5,
        do_hard_rounding: bool = False,
        do_sample: bool = True,
        temperature: float = 0.0,
        entropy_bound: float = 1.0,
        do_clip_norms: bool = True,
        clip_norm: float = 1.0,
        do_quit_on_eos: bool = False,
        debug: bool = False,
    ):
        self.use_kv_cache = use_kv_cache
        self.eos_token_id = eos_token_id
        self.ban_repeat_ngrams = ban_repeat_ngrams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.topk = topk
        self.do_hard_rounding = do_hard_rounding
        self.do_sample = do_sample
        self.temperature = temperature
        self.entropy_bound = entropy_bound
        self.do_clip_norms = do_clip_norms
        self.clip_norm = clip_norm
        self.do_quit_on_eos = do_quit_on_eos
        self.debug = debug

    def __repr__(self):
        return (
            f"SmoothGenerationConfig("
            f"use_kv_cache={self.use_kv_cache}, "
            f"eos_token_id={self.eos_token_id}, "
            f"ban_repeat_ngrams={self.ban_repeat_ngrams}, "
            f"no_repeat_ngram_size={self.no_repeat_ngram_size}, "
            f"topk={self.topk}, "
            f"do_hard_rounding={self.do_hard_rounding}, "
            f"do_sample={self.do_sample}, "
            f"temperature={self.temperature}, "
            f"entropy_bound={self.entropy_bound}, "
            f"do_clip_norms={self.do_clip_norms}, "
            f"clip_norm={self.clip_norm})"
        )


class SmoothGenerationOutput:
    def __init__(
        self,
        model: torch.nn.Module,
        toks: torch.LongTensor,  # of shape [batch_size, ,topk]
        tokprobs: torch.Tensor,
        kv_cache: Union[None, torch.Tensor],
        after_eos_mask: torch.BoolTensor,  # masks eos tokens and every token which is preceded by an eos token
        saved_states: List[Tuple],
        config: SmoothGenerationConfig,
        generation_start_idx: int,
    ):
        self.model = model
        self.toks = toks
        self.tokprobs = tokprobs
        self.kv_cache = kv_cache
        self.after_eos_mask = after_eos_mask
        self.saved_states = saved_states
        self.config = config
        self.generation_start_idx = generation_start_idx

    def __repr__(self):
        return (
            f"SmoothGenerationOutput("
            f"model={self.model.__class__.__name__}, "
            f"toks.shape={self.toks.shape}, "
            f"tokprobs.shape={self.tokprobs.shape}, "
            f"kv_cache={'None' if self.kv_cache is None else 'Tensor'}, ",
            f"after_eos_mask={self.after_eos_mask}"
            f"saved_states_count={len(self.saved_states)}, "
            f"config={self.config.__class__.__name__}, "
            f"generation_start_idx={self.generation_start_idx})",
        )


Cache: TypeAlias = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class SmoothModelForCausalLM(torch.nn.Module):
    """
    Base class for smooth text generation
    Transforms a autoregressive text generation model to a smooth text generation model
    """

    model: AutoModelForCausalLM
    embedding_matrix: torch.Tensor
    model_modder: Callable
    model_unmodder: Callable

    def __init__(self, model, embedding_matrix, model_modder, model_unmodder):
        """
        Transform a model into a smooth model

        model --- the base model
        embedding_matrix --- the embedding matrix of the base model
        model_modder --- a function which adds the kv_cache hooks to the model
        model_unmodder --- a function which removes the kv_cache hooks from the model
        """
        super().__init__()
        self.model = model
        self.embedding_matrix = embedding_matrix
        self.model_modder = model_modder
        self.model_unmodder = model_unmodder

    def forward(
        self,
        toks: torch.LongTensor,  # Shape: [batch_size, seq_len, topk]
        tokprobs: torch.Tensor,  # Shape: [batch_size, seq_len, topk]
        config: SmoothGenerationConfig,
        past_key_values: Optional[
            Cache
        ] = None,  # shape of cache given by the structure of the attention layers: [batch_size, num_heads, seq_len, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, Optional[Cache]]:
        
        # call the model and obtain the logits
        # for *reasons*, its different in every config case…
        if config.use_kv_cache:
            if past_key_values is None:
                emb = (self.embedding_matrix[toks] * tokprobs.unsqueeze(-1)).sum(dim=-2)
                output = self.model(
                    inputs_embeds=emb, use_cache=True, past_key_values=past_key_values
                )
            else:
                emb = (
                    (
                        self.embedding_matrix[toks[:, -1, :]]
                        * tokprobs[:, -1, :].unsqueeze(-1)
                    )
                    .sum(dim=-2)
                    .unsqueeze(1)
                )
                output = self.model(
                    inputs_embeds=emb, use_cache=True, past_key_values=past_key_values
                )
        else:
            emb = (self.embedding_matrix[toks] * tokprobs.unsqueeze(-1)).sum(dim=-2)
            output = self.model(inputs_embeds=emb)
            output.past_key_values = None

        # Extract logits from the last token
        logits = output.logits[:, -1, :]
        saved_state = ()
        kv_cache = output.past_key_values

        # gumbel-softmax sampling trick impl
        if config.do_sample:
            logits += (
                torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)
                * config.temperature
            )

        # repeat-ngram banning
        if config.ban_repeat_ngrams:
            banned_tokens = ban_repeat_ngrams(
                toks[:, :, 0], config.no_repeat_ngram_size
            )
            for batch_idx, banned in enumerate(banned_tokens):
                logits[batch_idx, banned] = float("-inf")
            # saved_state = saved_state + (banned_tokens, )

        # next step computation
        top_logits, top_tok = torch.topk(logits, k=config.topk)
        top_logits = bound_entropy(top_logits, config.entropy_bound)
        top_probs = F.softmax(top_logits, dim=-1)

        # hard rounding if necessary
        if config.do_hard_rounding:
            hard_rounded = torch.zeros_like(top_probs, device=top_probs.device)
            hard_rounded[:, 0] = 1.0
            top_probs = hard_rounded - top_probs.detach() + top_probs

        return top_tok, top_probs, saved_state, kv_cache

    def generalize_tokens(
        self, toks: torch.LongTensor, topk: int
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Transforms a tensor of tokens into a a tensor of generalized tokens
        toks --- Tensor of shape [batch_size, seq_len]
        returns the generalized tokens, as a pair toks, tokprobs, both of shape [batch_size, seq_len, topk]
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

    def generate(self, input_tokens, max_iters: int, config: SmoothGenerationConfig):
        """
        Generate smooth text using the model

        Parameters:
        input_tokens --- input tokens (ordinary or generalized)

        returns the SmoothGenerationOutput object containing the generated tokens
        """
        toks, tokprobs = (
            input_tokens
            if isinstance(input_tokens, tuple)
            else self.generalize_tokens(input_tokens, config.topk)
        )
        cache = None

        # config.random_state = save_random_state(input_tokens.device)

        saved_states = []
        init_len = toks.shape[1]
        after_eos_mask = torch.ones((toks.shape[0], toks.shape[1] + 1), dtype=bool) # the mask which removes everything after eos
        for i in range(max_iters):
            with torch.no_grad():
                # Save the random states
                saved_random_state = save_random_state(toks.device)

                # Generate one step
                newtok, newprobs, new_saved_states, new_cache = self.forward(
                    toks, tokprobs, config, cache
                )
                # newtok of shape [batch_size, topk]
                # newprobs of shape [batch_size, topk]
                # new saved_states: tuple
                # new_cache : Optional[Cache]
                max_tok = newtok[:, 0]

                # save & update
                saved_states.append((saved_random_state,) + new_saved_states)
                toks = torch.cat((toks, newtok.unsqueeze(1)), dim=1)
                tokprobs = torch.cat((tokprobs, newprobs.unsqueeze(1)), dim=1)
                cache = new_cache

                # update the mask which removes everything after eos
                new_eos_mask = max_tok != config.eos_token_id
                after_eos_mask = torch.cat(
                    (
                        after_eos_mask,
                        torch.logical_or(new_eos_mask, after_eos_mask[:, -1]).unsqueeze(
                            -1
                        ),
                    ),
                    dim=-1,
                )

                if config.do_quit_on_eos:
                    if torch.any(new_eos_mask):
                        break

        res = SmoothGenerationOutput(
            self,
            toks,
            tokprobs,
            cache,
            after_eos_mask[:, :-1],
            saved_states,
            deepcopy(config),
            init_len,
        )
        return res

    # we have to reimplement the basic transformers generate function as it does not support batches
    def ordinary_generate(
        self, input_tokens, max_iters: int, config: SmoothGenerationConfig
    ):
        model = self.model
        toks = input_tokens.clone()
        cache = None
        after_eos_mask = torch.ones((toks.shape[0], toks.shape[1] + 1), dtype=bool)

        for i in range(max_iters):
            with torch.no_grad():

                # Generate one step
                output = model.forward(
                    toks, use_cache=config.use_kv_cache, past_key_values=cache
                )

                # Extract logits from the last token
                logits = output.logits[:, -1, :]
                cache = output.past_key_values

                # gumbel-softmax sampling trick impl
                if config.do_sample:
                    logits += (
                        torch.distributions.Gumbel(0, 1)
                        .sample(logits.shape)
                        .to(logits.device)
                        * config.temperature
                    )

                # repeat-ngram banning
                if config.ban_repeat_ngrams:
                    banned_tokens = ban_repeat_ngrams(
                        toks[:, :, 0], config.no_repeat_ngram_size
                    )
                    for batch_idx, banned in enumerate(banned_tokens):
                        logits[batch_idx, banned] = float("-inf")

                # next step computation
                newtok = torch.argmax(logits, dim=-1)

                # save & update
                toks = torch.cat((toks, newtok.unsqueeze(1)), dim=1)

                # update the mask which removes everything after eos
                new_eos_mask = newtok != config.eos_token_id
                after_eos_mask = torch.cat(
                    (
                        after_eos_mask,
                        torch.logical_or(new_eos_mask, after_eos_mask[:, -1]).unsqueeze(
                            -1
                        ),
                    ),
                    dim=-1,
                )

                if config.do_quit_on_eos:
                    if torch.any(newtok == config.eos_token_id):
                        break

        return toks, after_eos_mask[:, :-1]


class SmoothLoss:
    loss = None

    class LossValue:
        model_output: SmoothGenerationOutput
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
            saved_states = self.model_output.saved_states
            toks = self.model_output.toks
            tokprobs = self.model_output.tokprobs
            config = self.model_output.config
            kv_cache = self.model_output.kv_cache
            after_eos_mask = self.model_output.after_eos_mask

            # compute (∂𝓛/∂τ_i)
            # we store the intermediate grads in tokprobs
            tokprobs.requires_grad_(True)
            loss_val = self.loss(toks, tokprobs)
            loss_val.backward()
            tokprobs.grad *= after_eos_mask.unsqueeze(-1)
            # init_grad = tokprobs.grad.clone().detach()

            # enable accumulating the grads in the kv_cache
            if config.use_kv_cache:
                for kv in kv_cache:
                    kv[0].requires_grad_(True)
                    kv[1].requires_grad_(True)

                # storage for passing the cached KV-cache gradients to the modded backprop
                kv_cache_grad_buf = []
                for kv in kv_cache:
                    kv_cache_grad_buf.append(
                        [
                            torch.zeros_like(kv[0][:, :, -1:, :]),
                            torch.zeros_like(kv[1][:, :, -1:, :]),
                        ]
                    )

            # add hooks for passing the stored kV-cache gradients
            # hooks, when passing through the attention layer, take the gradients from the storage and add them into the backprop flow
            if config.use_kv_cache:
                model.model_modder(model.model, kv_cache_grad_buf)

            # compute the range
            batch_size, seq_len, topk = toks.shape
            init_len = self.model_output.generation_start_idx

            # we rerun the computation backwards
            for i in reversed(range(init_len, seq_len - 1)):

                cur_toks = toks[:, :i, :]
                cur_tokprobs = tokprobs[:, :i, :]

                # recall the cache at time i
                cur_kv_cache = None
                if config.use_kv_cache:
                    cur_kv_cache = ()
                    for kv, kv_grad_buf in zip(kv_cache, kv_cache_grad_buf):
                        cur_kv_cache = cur_kv_cache + (
                            (kv[0][:, :, : (i - 1), :], kv[1][:, :, : (i - 1), :]),
                        )
                        if kv[0].grad is None:
                            pass
                        else:
                            # push the gradients of the final part of the KV-cache into the backprop-storage
                            kv_grad_buf[0].copy_(kv[0].grad[:, :, (i - 1) : i, :])
                            kv_grad_buf[1].copy_(kv[1].grad[:, :, (i - 1) : i, :])

                # restore the original random state for reproducibility
                load_random_state(saved_states[i - init_len][0], toks.device)

                # rerun the model on the step i
                newtok, newprobs, _, _ = model.forward(
                    cur_toks, cur_tokprobs, config, cur_kv_cache
                )

                # check that the re-generated output is the same
                if not torch.equal(newtok, toks[:, i, :]):
                    print(f"At position {i}")
                    print("Re-generated tokens:", newtok, newprobs)
                    print("Original tokens:", toks[:, i, :].shape, tokprobs[:, i, :])
                    print("These must be equal")
                assert torch.equal(newtok, toks[:, i, :])

                # take the gradient
                last_grad = tokprobs.grad[:, i, :]

                # clip the norms to avoid gradient explosion
                if config.do_clip_norms:
                    with torch.no_grad():
                        norm = torch.linalg.vector_norm(
                            last_grad, dim=(1), keepdim=True
                        )
                        mask = norm >= config.clip_norm
                        last_grad = torch.where(mask, last_grad / norm, last_grad)

                # TRACK THIS TO UNDERSTAND GRADIENT EXPLOSION
                # this is the gradient of the i-th token, which is being backpropagated
                # essentially the i-th state in an RNN
                if config.debug:
                    print(torch.linalg.vector_norm(last_grad))

                # GRADIENT UPDATE STATE

                # on the i-th step for all j < i
                # try to update ∂𝓛/∂τ_j ← ∂𝓛/∂τ_j + ∂𝓛/∂τ_i ∂τ_i(τ_1 … τ_i-1)/∂τ_j
                # if use_kv_cache, some of this gradient will be accumulated in the kv_cache
                # however on the j-th step it will reach the kv_cache due to the GradMod
                newprobs.backward(last_grad)

            # remove the hooks and unset the gradients
            if config.use_kv_cache:
                model.model_unmodder(model.model)
                for kv in kv_cache:
                    kv[0].requires_grad_(False)
                    kv[1].requires_grad_(False)

            tokprobs.requires_grad_(False)

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def __call__(self, output):
        return self.LossValue(output, self.loss)


# estimate 𝔼𝑋 and 𝔼𝑋² by sampling from rv 𝑋
def estimate_tensor_stats(rv, iters, seed=1337):
    set_determininsm(seed)
    e_rv = torch.zeros_like(rv()).to(dtype=torch.float64)
    e_rv2 = torch.zeros_like(e_rv).to(dtype=torch.float64)
    for _ in tqdm(range(iters)):
        rvi = rv().to(dtype=torch.float64)
        e_rv += rvi
        e_rv2 += rvi**2
    return e_rv / iters, e_rv2 / iters


# sample the model gradient from the smooth estimator
def smooth_seq_grad(model, loss, prompt, max_toks, config):
    def run():
        output: SmoothGenerationOutput = model.generate(prompt, max_toks, config)
        loss_val = loss(output)
        loss_val.backwards()
        res = total_grad(model)
        model.zero_grad()
        return res

    return run


# sample the model gradient from the REINFORCE estimator
def reinforce_grad(model, loss, prompt, max_toks, cfg):
    def run():
        toks, after_eos_mask = model.ordinary_generate(
            prompt, max_iters=max_toks, config=cfg
        )
        reward = -loss(toks)
        init_len = prompt.shape[1]
        output = model.model(toks)
        log_probas = F.log_softmax(output.logits, dim=-1)  # shape [batch_size, seq_len]
        # log_probas.sum().backward()
        # return None
        # remove the tokens generated after the eos_tok appeared
        log_probas = log_probas * after_eos_mask.unsqueeze(-1)
        # remove the prompt, as we do not generate it

        log_probas = log_probas[:, init_len:-1]
        toks = toks[:, init_len + 1 :]
        # we want probas[batch_size][toks[batch_size, seq_len ± 1]]
        seq_probas = log_probas.gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        total_seq_probas = seq_probas.sum(-1)

        total_seq_probas.backward(reward)
        res = total_grad(model)
        model.zero_grad()
        return res

    return run
