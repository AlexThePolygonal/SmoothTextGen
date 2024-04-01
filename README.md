## Immediate smooth text generation

### Everything written here is non-batched. Do not use batches!!!
### Also, I do not cache attentions
### Not very optimal


### The idea

Consider autoregressive text generation:

#### Base text gen
0. $[t_1, \ldots t_n] = \text{tokenizer}(str)$
1. $p_i = \text{model.forward}([\text{emb}(t_1), \ldots, \text{emb}(t_n)])$
2. $t_{n+1} = \text{argmax}(p_i)$
3. $goto \; 1$

The process is obviously non-differentiable, which isn't good. Can we change that?

Obviously, tokens are not differentiable in any way. This means that we will generate a sequence of suitable replacements.

Instead of tokens, we generate a sequence of probability distributions over tokens. Instead of a token $t$, we consider a _generalized token_ $\tau$, which is sequence of pairs $(t_1, p_1), (t_2, t_2), \ldots$, where $t_i$ is a token and $p_i$ is its probability.

The sum of the probabilities is 1.
Notice that generalized tokens have a natural embedding vector:
$$ emb(\tau) = \sum emb(t_i) p_i $$
As we know from `word2vec`, directions in the embedding space should be meaningful.
This means that generalized tokens can be model inputs, and model outputs are obviously again generalized tokens.

We deduce the following:
#### Naive smooth text generation
0. $[\tau_1, \ldots, \tau_n] = [t_1, \ldots t_n] = \text{tokenizer}(str)$
1. $p_i = \text{model.forward}([\text{emb}(\tau_1), \ldots, \text{emb}(\tau_n)])$
2. $\tau_{n+1} = [(t_1, p_1), (t_2, p_2), \ldots]$
3. $goto \; 1$

This obviously does not work: the entropy of $\tau_i$ explodes. To correct for entropy, we choose the temperature of the softmax so that the entropy $H(p_i)$ is bounded by some small value. If this value is too large, the generation becomes incoherent, while the gradients will die for too-small values. We keep only the top k values, these are enough as the distribution is tight

#### Possible smooth text generation
0. $[\tau_1, \ldots, \tau_n] = [t_1, \ldots t_n] = \text{tokenizer}(str)$
1. $p_i = \text{model.forward}([\text{emb}(\tau_1), \ldots, \text{emb}(\tau_n)])$
2. $p_i = \text{entropy\_normalize}(p_i)$
3. $\tau_{n+1} = \text{topk}(\text{sorted}([(t_1, p_1), (t_2, p_2), \ldots]))$
4. $goto \; 1$

We can use the gumbel-softmax trick to simulate sampling.

### Use-cases

- Consider any reinforcement learning problem with explicit reward functions, for example, the second part of RLHF, or BLEU fitting. Notice that the task is fully differentiable, this means we can compute gradients directly, without resorting to the log-derivative trick, with approx-zero overhead
- This is similar to RL in the sense that we need many samples. However, the variance of such gradient estimation should be lower, since we get rid of some of the stochasticity.
- Most likely, text GANs become more stable with this setup
- Notice that we can reuse foundation models, so this modification is comparatively cheap
- The construction is architecture-agnostic, anything from an RNN to a transformer to a SSM goes

### Pesky implementation details

#### Memory usage
in the naive implementation, the computational graph will consume $N^3$ memory when generating $N$ tokens, which will OOM immediately. We can do better.

Our implementation, compared to ordinary RL, will consume a single additional pass, and maximum memory usage stays the same

Sketch of the idea: consider the loss $\mathcal{L} = \mathcal{L}(\tau_1, \ldots, \tau_n)$.
We store $\frac{\partial \mathcal{L}}{\partial \tau_i}$ and, with $m$ going from $n$ to $1$, update them as follows:
$$
\frac{\partial \mathcal{L}}{\partial \tau_i} \leftarrow  \frac{\partial \mathcal{L}}{\partial \tau_i} + \frac{\partial \mathcal{L}}{\partial \tau_m} \frac{\partial \tau_m(\tau_1, \ldots, \tau_{m-1})}{\partial \tau_i} \qquad \forall i \in [1, m-1]
$$
recomputing $\tau_m(\tau_1, \ldots, \tau_{m-1})$ at each step.

#### Gradient explosion:

uuuuuuh uuuuuuuh attention is global so gradients will stack up at key inflection tokens uuuuh trust the plan

### Practice

As a first test of this idea, we implement it for the TinyStories model below