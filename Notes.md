
In this course we will learn many things, one of them being Mixture of Experts (MoEs). Sparse MoEs were popularised in modern language models by [this paper](https://arxiv.org/abs/2101.03961). The main reason why we need them is that we want bigger models to improve accuracy, but at the same time bigger models also mean higher computation and memory costs. A commonly used estimate for the total training FLOPs on a sequence of $N$ tokens is proportional to $6NP$, where $P$ is the number of active parameters (see [this blog](https://jax-ml.github.io/scaling-book/transformers/#forward-and-reverse-flops) for a detailed explanation). In simple words, a matrix multiplication of an $m\times k$ matrix by a $k \times n$ matrix costs approximately $2mnk$ FLOPs, and the forward pass plus backward pass is roughly three times the forward pass. This gives $3 \times 2NP=6NP$, or roughly $6P$ FLOPs per token. For inference, the parameter-multiplication estimate is closer to $2P$ per token, and attention adds an additional sequence-length-dependent cost. There are two kinds of experts: routed experts and shared experts.

Another question is: how "big" should the up-projection and down-projection be for each expert? In [this paper](https://arxiv.org/abs/2402.07871) they answer this question. They introduce the "granularity", which is defined as

$$

G = \frac{\text{dim}_{FFN}}{\text{dim}_{expert}}.

$$

where $\text{dim}_{FFN}$ is the intermediate dimension of the Feed-Forward Network and $\text{dim}_{expert}$ is the intermediate dimension of a single expert. They show in the paper that, up to a certain point, greater granularity improves accuracy.

  

Now question: how many FLOPS do we need for a given model? With the function `get_nparams_and_flops` in `torchfeather/model/model_args.py` we compute an approximate answer.

### Notation

From now on, we will use the following notation:

- We will try to refer to the hidden dimension of an embedded token always with $D$.

- $H$ will denote the number of heads.

- $T$ will usually denote the sequence length.


### Rotary Position Embedding (RoPE)

  

RoPE is a way to encode the position of each token in a sequence. In the [original Transformer paper](https://arxiv.org/abs/1706.03762), they define sinusoidal positional embeddings as

$$

\begin{cases}

PE_{(pos, 2i)} = \sin(pos / 10000^{2i/D}), \\

PE_{(pos, 2i + 1)} = \cos(pos / 10000^{2i/D}).

\end{cases}

$$

Sinusoidal embeddings explicitly encode absolute positions. In [this paper](https://arxiv.org/abs/2104.09864), they propose a way to encode absolute positions through rotations while making the attention score depend on the relative position as well. More precisely, we would like the scalar product of the encoded query vector $f_q(x_m, m)$ and the encoded key vector $f_k(x_n, n)$ to depend on $x_m$, $x_n$, and $m-n$, but not independently on the absolute positions $m,n$:

$$

\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n).

$$

A simple attempt at a solution is the following. In the 2D case, identify vectors with complex numbers and define

$$

f_k(x, m) = f_q(x, m) := (W_q x) e^{im\theta},

$$

so that, using the appropriate complex inner product,

$$

\langle f_q(x_m, m), f_k(x_n, n) \rangle = (W_q x_m)^T (W_k x_n) e^{i(m-n)\theta}.

$$

More precisely we can write

$$

f(x, m) := R_{\theta, m} Wx
$$
where
$$
R_{\theta, m} := \begin{bmatrix}
\cos(m \theta) & -\sin(m\theta) \\
\sin(m \theta) & \cos(m\theta)
\end{bmatrix}
$$
This is only for the 2D case, but we can extend it by splitting the $D$ coordinates of the embedded vector into $D/2$ two-dimensional vectors and applying the same operation to each sub-vector, with perhaps different angles $\theta_1, \dots, \theta_{D/2}$. We want this because we want the model to treat two tokens that are a fixed distance from each other similarly, independently of where they appear in the text. In other words, we want the model to care about their relative position.


How do we choose the $\theta$s? Usually they range approximately from $1.0$ to a very small number, so the frequencies associated with the higher-indexed pairs are lower and those vectors rotate more slowly. In `rope.py` we define the frequencies exactly as

```python

freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

```


With this "simple" RoPE, the model can still have difficulty generalising to positions beyond the training context, because the slowest rotations complete very few cycles and the highest-frequency rotations can alias at long distances. This is a limitation of extrapolation, rather than RoPE failing to encode relative position.

We can't use the same $\theta$ across all the dimensions $i \in [1, \dots, D/2]$ because each of them has a different "job". Higher $\theta$ s help for tokens that are close from each other in the sequence, lower $\theta$ s help for those that are further away from each other.


What is the solution to teach the model to use an $L'$-token context window after it has been trained with an $L$-token context window? I could divide each position by 4 if I pretrained it with 4,000 tokens and wanted to use 16,000 tokens. But this can damage the high-frequency terms, which are important for distinguishing adjacent tokens. So what we can do is scale only part of the frequencies, leave the other part unchanged, and use a linear interpolation in between. This idea is used in [this paper](https://arxiv.org/abs/2309.00071), and it is called YaRN.


Define
$$
r(i) = L/\lambda_i,
$$
where
$$
\lambda_i := \frac{2\pi}{\theta_i}.
$$
Let
$$

\gamma = \begin{cases}

0, \quad r < \alpha, \\

1, \quad r > \beta, \\

\frac{r-\alpha}{\beta - \alpha}, \quad \text{otherwise}.

\end{cases}

$$
and define
$$
h(\theta_{i}) = \left( 1 - \gamma (r(i)) \right)\theta_{i} + \gamma(r(i))\frac{\theta_{i}}{s}.
$$
In the ramped region this replaces $\theta_i$ with $\theta_i/s$; since $s>1$, this reduces the frequency and increases the wavelength. The implementation in `rope.py` uses a linear ramp over frequency indices, with the exact direction determined by the indexing convention.

### Initialization of the weights

Assume $x$ is a vector with $p$ independent features with variance $v$ and mean $0$. Assume $W$ are weights with mean $0$ and variance $\sigma^{2}$. So one has that

$$
y = Wx
$$

has $0$ mean and variance

$$
\sum_{i=1}^{p} v \sigma^{2} = p v \sigma^{2}.
$$

If we want $Var(x) = Var(y)$ then we need the weight standard deviation $\sigma = \frac{1}{\sqrt{p}}$. This is the basic idea behind this kind of initialization.

For the residual stream case, consider

$$
x_{\ell+1} = x_{\ell} + f_{\ell}(x_{\ell})
$$

so that

$$
Var(x_{\ell+1}) = Var(x_{\ell}) + Var(f_{\ell}) + 2Cov(x_{\ell}, f_{\ell}) \sim Var(x_{\ell}) + Var(f_{\ell}).
$$

The reason why we often approximate
$$

Cov(x_{\ell}, f_{\ell}) \sim 0

$$

is that the residual branch is initially close to uncorrelated with the residual stream; it is not strictly independent because the branch receives $x_{\ell}$ as input.

So, if $Var(f_{\ell}) = v^{2}$ we have

$$

Var(x_{L}) = Var(x_{0}) + L v^{2}.

$$

If we want the variance to remain roughly constant across $L$ layers, and there are two residual branches per layer, we want each branch to have standard deviation approximately

$$

v \approx \frac{1}{\sqrt{2L}}

$$

In DeepSeek though they choose

```python

self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

```

so basically the standard deviation decreases as the layer gets deeper.

Again, we do this because variance preservation is important for neural networks training.

#### Multi-Head Latent Attention


Problem: autoregressive decoding is IO bound. A GPU has two main characteristics:  

- **Compute throughput**: how fast it computes arithmetic.

- **Memory throughput**: how fast it moves data between the memory (HBM) and the compute cores.

For example, a H100 SXM has:

- BF16 Tensor Core: 1979 teraFLOPS

- GPU Memory 80GB

- GPU Memory 3.35TB/s

As we can see here, $3.35 \ll 1979$. The relevant quantity is the arithmetic intensity:

$$

\frac{\text{FLOPS performed}}{\text{total bytes IO (read + write)}}

$$

In the H100:

$$

\text{A.I.(H100)} = \frac{1979 \cdot 10^{12}}{3.35 \cdot 10^{12}} \approx 590 \frac{\text{FLOPs}}{\text{bytes moved}}

$$

This means that, in this simplified roofline model, the computation must perform more than about 590 FLOPs for every byte moved in order to become compute-bound.

**An example:** For a matrix multiplication $Y = WX$, the GPU must read not only the weights $W$, but also the input $X$, and then write the output $Y$. If $W\in\mathbb{R}^{n\times n}$ and $X\in\mathbb{R}^{n\times B}$, using BF16 the total memory traffic is approximately $2n^2 + 4nB$ bytes, while the computation requires about $2n^2B$ FLOPs. Therefore, the arithmetic intensity is

$$\text{A.I.} \sim \frac{\underbrace{2n^2B}_{\text{compute}}} {\underbrace{2n^2}_{W\text{ read}}+\underbrace{2nB}_{X\text{ read}}+\underbrace{2nB}_{Y\text{ write}}} = \frac{nB}{n+2B}.$$

In the typical LLM regime where the hidden dimension $n$ is much larger than the batch size $B$, the weight matrix dominates the memory traffic, so $AI \approx B$. In particular, for batch size $B=1$, the arithmetic intensity is roughly $1$ FLOP/byte, far below the H100 balance point of about $590$ FLOPs/byte, which is why autoregressive decoding at small batch sizes is strongly memory-bandwidth bound.

In [this paper](https://arxiv.org/abs/1911.02150) they show that, for attention, the inverse arithmetic intensity scales as

$$

O(\frac{n}{D} + \frac{1}{B})

$$


When decoding (generating tokens autoregressively), we have 1 query and many kv tokens to load. So the GPU is always waiting for the new kv tokens to be loaded. How can we make it faster?

  

- Arithmetic operations: $O(BD^{2} + BnD)$

- Memory access for the KV cache: $O(BnD)$


So basically:

$$

\frac{1}{\text{A.I.}} = O(\frac{1}{B} + \frac{n}{D})

$$

and we want this to be very small. The cache-read term is therefore important when $n$ is large, which is usually the case during long-context decoding.

#### Vanilla MHSA

Let $h_{t} \in \mathbb{R}^{d}$ and we have some kv vectors. Define $d_{h} = \frac{D}{H}$. Simply consider, for each $i \in \{1, \dots, H\}$

$$

a_{t, j, i} = \text{Softmax}_{j} \left( q_{t, i}^{T} k_{j, i}\right)

$$

and then

$$

o_{t, i} = \sum\limits_{j \leq t} a_{t, j, i} v_{j, i}

$$

and

$$

\mu_{t} = \sum\limits_{i \leq H} W_{i}^{o}o_{t, i}

$$

  

The cache holds, for every layer, for every token:

$$

2 H d_{h}

$$

  
  

**First solution: [MQA](https://arxiv.org/abs/1911.02150).** If there are $n_{kv}$ key/value heads, we can rewrite

$$

O\left(\frac{1}{b} + \frac{n}{d}\right)= O\left(\frac{1}{b} + \frac{n}{d_{h} n_{kv}}\right)

$$

One way to make attention faster is to reduce the number of key/value heads to 1, i.e. $n_{kv}=1$. This can degrade accuracy, though.

  

**Second solution: [GQA](https://arxiv.org/abs/2305.13245).** So what we can do is reduce $n_{kv}$ from $H$ to a lower value instead of 1. Basically, we use the same KV vectors for the all the different query vectors within a group of heads.




**Third solution:** [**MLA**](https://arxiv.org/abs/2405.04434). Again, the goal is to prevent the KV cache from growing too much and eventually filling GPU memory. In MQA and GQA, we reduce the number of key/value heads. In MLA, instead, we compress the key/value information of each token into a lower-dimensional latent vector. Consider the following image from the [DeepSeek paper](https://arxiv.org/abs/2405.04434).
  
![[MLA_diagram.png]]

First, we project $h_t$ into the compressed vector $c^{kv}_t$ with

```python

self.wkv_a = nn.Linear(

self.dim,

self.kv_lora_rank + self.qk_rope_head_dim,

bias=False

)

```

The first `self.kv_lora_rank` dimensions correspond to $c^{kv}_t$, while `self.qk_rope_head_dim` corresponds to the $k^R$ component in the figure. The same RoPE dimension is used for $q^R$ on the query side. We then up-project $c^{kv}_t$ to obtain the content keys and values for all heads:

```python

self.wkv_b = nn.Linear(

self.kv_lora_rank,

self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),

bias=False

)

```

Let the superscript $d$ denote a down-projection and $u$ an up-projection. The compressed KV representation is

$$

c^{kv}_t = W^{dkv} h_t,

$$

where
$$

W^{dkv} \in \mathbb{R}^{d_c \times D},

$$

and $d_c$ is the dimension of the compressed latent space. The content key and value for head $i$ are then
$$

k^{c}_{t,i} = W^{uk}_i c_t^{kv},

\qquad

v^{c}_{t,i} = W^{uv}_i c_t^{kv},

$$
where
$$

W^{uk}_i, W^{uv}_i \in \mathbb{R}^{d_h \times d_c}.

$$
The important *inference-time trick* is that we do not need to explicitly reconstruct all the keys. For each head $i \in \{1,\dots,H\}$ and positions $t,j \in \{1,\dots,T\}$,

$$

q_{t,i}^{T}k^{c}_{j,i}

=

(W^{q}_i h_t)^T W^{uk}_i c^{kv}_j

=

\left[(W^{uk}_i)^T W^q_i h_t\right]^T c^{kv}_j.

$$
Therefore, after training, we can precompute

$$

(W^{uk}_i)^T W^q_i,

$$

and directly project the query into the latent KV space. The attention score can then be computed against the cached $c^{kv}_j$, without reconstructing $k^c_{j,i}$.

A similar absorption can be performed on the value side:

$$

o_{t,i}

=

\sum_j a_{t,j,i} v^c_{j,i}

=

\sum_j a_{t,j,i} W^{uv}_i c^{kv}_j

=

W^{uv}_i \sum_j a_{t,j,i} c^{kv}_j.

$$
Define

$$

\overline{o}_{t,i}

=

\sum_j a_{t,j,i} c^{kv}_j.

$$

Then

$$

\mu_t

=

\sum_i W^o_i o_{t,i}

=

\sum_i W^o_i W^{uv}_i \overline{o}_{t,i}.

$$
Since $W^o_i W^{uv}_i$ is fixed after training, it can also be precomputed. Thus, at inference time, attention can operate directly on the compressed latent vectors rather than repeatedly reconstructing the full keys and values.

**Why do we need the distinction between `NoPE` and `RoPE`?**

The problem is that `RoPE` prevents the absorption trick above. With RoPE, an attention score has the form

$$

q_m^T k_n

=

[W^q x_m]^T

R_{\theta,m-n}

W^k x_n.

$$
In the MLA notation, this would introduce a term such as

  

$$

h_t^T

(W^q_i)^T

R_{t,j}

W^{uk}_i

c^{kv}_j.

$$

The rotation matrix $R_{t,j}$ depends on the relative positions $t$ and $j$, so it cannot be absorbed into a single fixed projection matrix.

MLA therefore separates the query and key into a **content** component and a **positional** component:
$$

q_{t,i}

=

[q^c_{t,i}, q^r_{t,i}],

\qquad

k_{t,i}

=

[k^c_{t,i}, k^r_t].

$$
The attention dot product becomes

$$

q^T_{t,i}k_{j,i}

=

(q^c_{t,i})^T k^c_{j,i}

+

(q^r_{t,i})^T k^r_j.

$$

The first term,
$$

(q^c_{t,i})^T k^c_{j,i},

$$
contains no `RoPE`, so the absorption trick can be applied.

The second term,
$$
(q^r_{t,i})^T k^r_j,
$$

contains the positional `RoPE` information and is computed separately.

In the code, the absorption of the content-key projection into the query projection is performed by

```python

wq_abs_nope = torch.bmm(

w_uk.float().transpose(1, 2),

wq_nope.float()

).to(dtype=dtype)

```

Here, `w_uk` has shape

```python

[n_heads, qk_nope_head_dim, kv_lora_rank]

```

so after transposition it has shape

```python

[n_heads, kv_lora_rank, qk_nope_head_dim]

```

and multiplying it by `wq_nope` produces the absorbed query projection with shape

```python

[n_heads, kv_lora_rank, dim]

```

This lets each query be projected directly into the compressed KV latent space used by the cache.

### Basics of backpropagation

Let's consider the case of a NN with L layers. We have
$$
z^{(\ell)} = w^{(\ell)} a^{(\ell-1)} + b^{(\ell)}
$$
and
$$
a^{(\ell)} = \sigma(z^{(\ell)})
$$
for $\ell = 1, \dots, L$ with $a^{(0)} = x$. Now one has
$$
\frac{\partial C}{\partial a^{(\ell)}} = \frac{\partial C}{\partial a^{(\ell + 1)}} \frac{\partial a^{(\ell+1)}}{\partial z^{(\ell+1)}} \frac{\partial z^{(\ell+1)}}{\partial a^{(\ell)}} = \frac{\partial C}{\partial a^{(\ell + 1)}} \sigma^{'}(z^{(\ell)}) w^{(\ell+1)}
$$
Then once I have all
$$
\frac{\partial C}{\partial a^{(\ell)}}
$$
we do
$$
\frac{\partial C}{\partial w^{(\ell)}} = \frac{\partial C}{\partial a^{(\ell)}} \frac{\partial a^{(\ell)}}{\partial z^{(\ell)}} \frac{\partial z^{(\ell)}}{\partial w^{(\ell)}} = \frac{\partial C}{\partial a^{(\ell)}} \sigma'(z^{(\ell)})a^{(\ell-1)}
$$

Now, assume
$$
L(x) := h(\mu(x), \nu(x))
$$
then
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \mu} \frac{\partial \mu}{\partial x} + \frac{\partial L}{\partial \nu} \frac{\partial \nu}{\partial x}
$$
Now, say we have
$$
L = g(x_{1}+x_{2})
$$
where $x_{1}, x_{2}$ come from different GPUs. Then we copy the gradient of $L$ wrt $y$ on both GPUs.

![[diagram.png]]

#### PyTorch backpropagation

A quick intro on `PyTorch` implementation of backpropagation. The idea is the following. Assume now we have black-boxes:
$$
x \to f_{1}(x, w_{1})=:y_{1} \to f_{2}(y_{1}, w_{2})=:y_{2} \to \dots \to f_{n}(y_{n-1}, w_{n})=:y_{n}
$$
Then, if we want to compute the partial derivatives of the loss wrt the params
$$
\frac{\partial L}{\partial \omega_{i}} = \frac{\partial L}{\partial y_{i}} \frac{\partial y_{i}}{\partial \omega_{i}}
$$
and for
$$
\frac{\partial L}{\partial y_{i-1}} = \frac{\partial L}{\partial y_{i}} \frac{\partial y_{i}}{\partial y_{i-1}}
$$
so each black-box implements a way to do the `backward` pass, i.e. given the gradient wrt the outputs of that black-box, it gives you back the gradient wrt the inputs and weights.
$$
\frac{\partial L}{\partial y_{i}} \xrightarrow{} \frac{\partial L}{\partial y_{i-1}}
$$
Let's consider the following example from the PyTorch docs

```python
class QKVProjection(Function):
    """Projects input x into Q, K, V: q = x @ w_q, k = x @ w_k, v = x @ w_v."""
    boxed_grads_call = True

    @staticmethod
    def forward(ctx, x, w_q, w_k, w_v):
        ctx.save_for_backward(x, w_q, w_k, w_v)
        return x.mm(w_q), x.mm(w_k), x.mm(w_v)

    @staticmethod
    def backward(ctx, grads):
        x, w_q, w_k, w_v = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_weights = []

        # Process each grad independently and free it immediately.
        # Without boxed_grads_call, all three grads would stay alive
        # until backward returns, tripling peak grad memory.
        for i, w in enumerate((w_q, w_k, w_v)):
            grad_out = grads[i]
            grads[i] = None      # Release reference in the caller's list
            grad_x += grad_out.mm(w.t())
            grad_weights.append(x.t().mm(grad_out))
            del grad_out         # grad_out can now be freed by the runtime

        return grad_x, *grad_weights
```

Define `o_1, o_2, o_3 = x.mm(w_q), x.mm(w_k), x.mm(w_v)` where `o` stays for output. The `backward` pass receives `grads` which is
$$
\frac{\partial L}{\partial o_{1}}, \frac{\partial L}{\partial o_{2}}, \frac{\partial L}{\partial o_{3}}
$$

Since
```python

forward(ctx, x, w_q, w_k, w_v)

```
has four inputs, `backward` must return the gradients with respect to those four inputs, in the same order:
$$
\frac{\partial L}{\partial x},

\qquad

\frac{\partial L}{\partial W_q},

\qquad

\frac{\partial L}{\partial W_k},

\qquad

\frac{\partial L}{\partial W_v}.
$$


For one branch, for example
  
$$
q = xW_q,
$$
we have

$$
\frac{\partial L}{\partial x}\Big|_q = G_q W_q^\top,
$$
and
$$
\frac{\partial L}{\partial W_q} = x^\top G_q.
$$
Because `x` is used in all three branches,
$$
q=xW_q,\qquad

k=xW_k,\qquad

v=xW_v,
$$

its total gradient is
$$
\boxed{

\frac{\partial L}{\partial x}

=

G_qW_q^\top

+

G_kW_k^\top

+

G_vW_v^\top

}

$$

This is what

```python

grad_x = torch.zeros_like(x)

  

for i, w in enumerate((w_q, w_k, w_v)):

grad_out = grads[i]

grad_x += grad_out.mm(w.t())

```

computes.

The weight gradients are
$$

\frac{\partial L}{\partial W_q}=x^\top G_q,

\qquad

\frac{\partial L}{\partial W_k}=x^\top G_k,

\qquad

\frac{\partial L}{\partial W_v}=x^\top G_v.
$$

So after the loop,

```python

grad_weights[0] = x.T @ G_q # dL/dW_q

grad_weights[1] = x.T @ G_k # dL/dW_k

grad_weights[2] = x.T @ G_v # dL/dW_v

```

Therefore `backward` returns

$$

\left(

\frac{\partial L}{\partial x},

\frac{\partial L}{\partial W_q},

\frac{\partial L}{\partial W_k},

\frac{\partial L}{\partial W_v}

\right).

$$

## Distributed training

#### An example - DDP

Assume we have $4$ GPUs, and consider
$$
\begin{align}
& x_{0} \xrightarrow{} GPU_{0} \xrightarrow{} (y_{0}, \text{target}_{0}) \xrightarrow{} \ell_{0}, \\
& x_{1} \xrightarrow{} GPU_{1} \xrightarrow{} (y_{1}, \text{target}_{1}) \xrightarrow{} \ell_{1}, \\
& x_{2} \xrightarrow{} GPU_{2} \xrightarrow{} (y_{2}, \text{target}_{2}) \xrightarrow{} \ell_{2}, \\
& x_{3} \xrightarrow{} GPU_{3} \xrightarrow{} (y_{3}, \text{target}_{3}) \xrightarrow{} \ell_{3}, \\
\end{align}
$$
And fortunately
$$
\frac{\partial L}{\partial \theta} = \frac{1}{4} \left( \frac{\partial \ell_{0}}{\partial \theta} + \frac{\partial \ell_{1}}{\partial \theta} + \frac{\partial \ell_{2}}{\partial \theta} + \frac{\partial \ell_{3}}{\partial \theta}\right)
$$
First of all, `torchrun` sets all the enviroments variables like `LOCAL_RANK`, `WROLD_SIZE`, `RANK`, etc. We access them for example as:
```python
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
```


## Pipeline Parallelism

In [this paper](https://arxiv.org/abs/1811.06965) they talk about "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism". For example, assume that your model is too big to be stored in a single device. What we can do is to split it among say 4 gpus. We can put layers 0, 1 into device0, layers 2, 3 into device1, layers 4, 5 into device2 and finally layers 6, 7 into device3. The only issue is how we make the different devices communicate. This is is a bit inefficient, and one solution is pipeline parallelism. This is because at any given moment, there is only one device working and this means that we are wasting resources. Consider the following graph from [this paper](https://arxiv.org/abs/2104.04473)

The idea is to split the batch into 8 pieces. So once the gpu is done with the first batch it can start with the gpu 2. Same logic for the backward pass.

![[JPEG image-42D4-A78E-74-0.jpeg]]

To reduce the bubbles, we could do something like

![[JPEG image-4D18-801C-6A-0.jpeg]]

Now, why is the backward pass twice the size of the forward one? Because you need to compute $\frac{\partial L}{\partial x}$ and $\frac{\partial L}{\partial w}$ and for this we require 2 matrix multiplications (for the simplified case of a linear layer).

Each GPU arrives to a state called "steady", where it starts alternating a forward step with a backward step. An advantage of this second case wrt the first one is that, device 4, after it has done the backward step 1, it can delete the activations, so it frees memory. At each point in time, device 4 has just activations from one single group.

![[JPEG image-48B6-9398-47-0.jpeg]]

Here we interleaved groups of layers (stages). Instead a sequential group of layers, we save the 0,1,8,9 in the device 1, then 2, 3, 10, 11 in device 2. At $t_{4}$ the first minibatch is sent back to device 1 from device 4. And we improved the number of time steps necessary to finish an optimisation step. The problem is that this does not come for free, because this schedule requires extra communication.

Now, there's a very smart way to avoid bubbles completely. To proceed with the backward pass, I don't actually need to compute $\frac{\partial L}{\partial w}$, I just need $\frac{\partial L}{\partial w}$ and I can delay $\frac{\partial L}{\partial w}$ for each layer. In [this paper](https://arxiv.org/abs/2401.10241) they propose a zero bubble schedule.

See a visual diagram here. The idea is to separate the backward pass for the weights, and the backward pass for the gradient wrt the input.

![[Screenshot 2026-09-16 at 12.12.56.png]]

In the DeepSeek paper, they do this symmetrically. This is called DualPipe

![[Screenshot 2026-09-16 at 12.14.23 1.png]]

How to actually do this in practice is pretty easy. We construct a csv file with columns $t_{0}, t_{1}, t_{2}$ etc. And rows the device, and in each cell I store what that device should be performing at that time step.

But the question is: how do we make sure that they are in sync? We use: `torch.distributed.isend(tensor, dst=None)` which sends it to another device that receives with `torch.distributed.irecv(tensor, dst=None)`. For each isend there must be a irecv.

During pretraining, we don't want to use padding, but we want all the sequences to have length $2048$. So what we do is: we just add the
- BOS + sequence1 + EOS + BOS + sequence2
- sequence2 + EOS + BOS + sequence3
This is different from what happens in the post-training though.

Now, dataloaders in different devices will sample batches that are always disjoint between each other.

Let's talk about [`torchrun`](https://docs.pytorch.org/docs/2.14/elastic/run.html). The way we launch training runs is with `torchrun`. We do
```bash
torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```
For example, if we have one node with 8 gpus and we want to use all of them we set `NUM_TRAINERS=8`.


Now imagine we have 16 devices. Placing one layer in each gpu would underutilised them. So we could do the following. We could use devices $k$ for $k \text{ mod } 4=0$
for batch $B_{0}$, then devices $k$ for $k \text{ mod } 4=1$ for batch $B_{1}$ etc. So basically we use groups of 4 devices to perform optimization step for independent batches. At the end of accumulation steps then we can sum gradients across ranks (that contain the same layers of the pp). This is the `device mesh`.

With `slurm` you can allocate the resources you need. `slurm` will work with `torchrun` and will assign to each subprocess inside each node a world_size, rank, local_rank, etc.


Consider the following situation. $2$ minibatches, $8$ devices. The first two devices contain the first $8$ layers, etc. The shape has two dimensions, one for the pipeline parallelism, and one for data parallelism.

![[JPEG image-4CF9-87B2-6C-0.jpeg]]

Rule: device that need to share lots of data should be in dimensions with small strides (small jumps in the memory array).

Take a tensor with (2, 2, 4) shape (stride (8, 4, 1)), then `dp_replicate` is the first one, `dp_shard` is the second one, then `TP` is the third one. So I can compute, if I am in rank 14:
- `dp_replicate` = 14 / 8 = 1, remainder 6. I will now use the remainder to continue.
- `dp_shard` = 6 / 4 = 1, remainder 2. I will now use the remainder to continue.
- `TP` = 2 / 1 = 2, remainder 0.

Therefore, rank 14 will have index (1, 1, 2). The device mesh tells you who you should talk to and for what purpose.

Example: if I want to run TP, and I am rank 14, who should I talk to? I compute the DM index (1, 1, 2), and I should talk to all the other ranks with DM (1, 1, i) for i = 0, 1, 2, 3.

What I want to know how to find the ones for `dp_shard`? It's DM(1, 1, 2) and DM(1, 0, 2).