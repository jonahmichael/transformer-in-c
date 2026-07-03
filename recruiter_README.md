# transformer in c

a from-scratch implementation of the transformer architecture described in "attention is all you need" (vaswani et al., 2017) — written in pure c with zero ml framework dependencies.

no pytorch. no tensorflow. no numpy. no eigen. no blas.

just c, math, and manual memory management.

---

## why this exists

most people implement transformers in pytorch in about 50 lines. this project does it in c from first principles — every matrix multiplication, every softmax, every layer norm, every attention head — written by hand with full understanding of the underlying math.

the goal was to prove that i understand how transformers work at the level of memory, pointers, and floating point arithmetic — not just how to call nn.transformer().

---

## what is implemented

- tensor struct with row-major memory layout and manual malloc/free
- matrix multiplication (triple nested loop, cache-aware design)
- softmax with numerical stability (subtract max before exp)
- layer normalization (mean, variance, gamma, beta)
- gelu activation (gaussian error linear unit, derived from normal cdf approximation)
- scaled dot-product attention (qk^t / sqrt(d_k), then softmax, then v)
- causal masking (decoder cannot attend to future positions)
- multi-head attention (project, split into h heads, attend, concat, project back)
- sinusoidal positional encoding (sin/cos at different frequencies per dimension)
- encoder block (mha + residual + layernorm + ffn + residual + layernorm)
- decoder block (masked mha + residual + layernorm + cross-attention + residual + layernorm + ffn + residual + layernorm)
- full transformer (stack of n encoders + n decoders)
- random weight initialization
- complete memory management with zero leaks

---

## project structure

```
transformer/
├── src/
│   ├── tensor.c / tensor.h         -- tensor struct, create, free, get, set
│   ├── math_ops.c / math_ops.h     -- matmul, matadd, softmax, layernorm, gelu
│   ├── attention.c / attention.h   -- transpose, scaled attention, mha
│   ├── encoder.c / encoder.h       -- full encoder block
│   ├── decoder.c / decoder.h       -- full decoder block
│   ├── transformer.c / transformer.h -- stacked encoder + decoder
│   ├── utils.c / utils.h           -- positional encoding
│   └── main.c                      -- entry point and tests
├── Makefile
└── README.md
```

---

## how to build and run

requirements: gcc, make, linux (tested on ubuntu 24.04 / wsl2)

```bash
git clone https://github.com/jonahmichael/transformer-in-c
cd transformer-in-c
make build
./transformer
```

to clean build artifacts:

```bash
make clean
```

---

## sample output
```bash
positional encoding (3 x 8):
0.0000  1.0000  0.0000  1.0000  0.0000  1.0000  0.0000  1.0000
0.8415  0.5403  0.0998  0.9950  0.0100  0.9999  0.0010  1.0000
0.9093 -0.4161  0.1987  0.9801  0.0200  0.9998  0.0020  1.0000
transformer output (3 x 8):
-1.4916 -1.0593 -0.7117 -0.3170  0.2887  0.7206  1.0206  1.5497
-1.4853 -1.0576 -0.7168 -0.3307  0.2966  0.7299  1.0132  1.5508
-1.4777 -1.0552 -0.7230 -0.3469  0.3055  0.7409  1.0044  1.5520
```


each row of the output has mean = 0 and variance = 1, confirming layer normalization is working correctly.

---

## memory safety

verified with valgrind — zero memory leaks, zero errors:
```
==19911== HEAP SUMMARY:
==19911==     in use at exit: 0 bytes in 0 blocks
==19911==   total heap usage: 354 allocs, 354 frees, 32,336 bytes allocated
==19911==
==19911== All heap blocks were freed -- no leaks are possible
==19911==
==19911== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
``` 

![alt text](<carbon (2).png>)

---

## architecture details

### tensor memory layout

tensors are stored in row-major order as a flat 1d float array.

element at row i, column j is accessed as:
```c
data[row * cols + col]
```

this matters for performance. reading a full row is sequential in memory (cache friendly). reading a full column requires jumping by `cols` elements each time (cache miss). this is why the inner loop of matmul iterates over columns, not rows.

### scaled dot-product attention
```
attention(q, k, v) = softmax(qk^t / sqrt(d_k)) * v
```

the division by sqrt(d_k) prevents dot products from growing too large as dimension increases, which would push softmax into regions where gradients vanish.

softmax subtracts the row maximum before computing exp() for numerical stability. without this, large values cause expf() to overflow to infinity.

### causal masking

the decoder uses masked self-attention so that position i cannot attend to positions after it. this is implemented by setting attention scores at positions j > i to -1e9 before softmax, which causes them to become approximately zero after the exponential.

```
if (j > i) scores[i][j] = -1e9f;
```

this is what makes the decoder autoregressive — each token can only see past tokens, never future ones.

### multi-head attention

instead of one attention computation over the full d_model dimensions, multi-head attention runs h smaller attention computations in parallel, each over d_k = d_model / h dimensions.

each head learns to focus on different aspects of the input (syntax, coreference, position, semantics) through training. this specialization emerges naturally from gradient descent — not by design.

the outputs of all heads are concatenated and projected through w_o to mix information across heads.

### residual connections

every sublayer (attention, ffn) adds its input back to its output:
```
output = sublayer(input) + input
```

this creates a gradient highway that allows gradients to flow directly through deep stacks without vanishing. without residuals, training transformers with 6+ layers would be practically impossible.

### positional encoding

attention has no inherent sense of word order — the same set of tokens in different orders would produce identical attention scores. positional encoding fixes this by adding a unique position-dependent signal to each token embedding.

the encoding uses sin and cos functions at different frequencies:
```
pe(pos, 2i)   = sin(pos / 10000^(2i / d_model))
pe(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

```
each position gets a unique fingerprint across all dimensions. higher dimensions change more slowly (lower frequency), lower dimensions change faster (higher frequency).

---

## layer norm verification

after each sublayer, layer norm is applied. the output of each row should have mean = 0 and variance = 1.

verified:
```
row 0 -- mean: 0.000000, variance: 0.999990
row 1 -- mean: 0.000000, variance: 0.999990
row 2 -- mean: 0.000000, variance: 0.999990

the tiny deviation from 1.0 is normal floating point precision — pytorch produces the same deviation.

---

## things i learned building this

- row-major memory layout and why cache misses kill performance in matmul
- why softmax needs the max subtraction trick to avoid floating point overflow
- why we divide attention scores by sqrt(d_k) — dot products grow proportionally to dimension, which saturates softmax and kills gradients
- gelu comes from approximating the gaussian cdf — the 0.044715 cubic term is a taylor series correction, sqrt(2/pi) falls out of the gaussian integral
- residual connections are not an architectural choice, they are a necessity for training deep networks
- calloc vs malloc — malloc leaves memory uninitialized, calloc zeros it. using malloc for float arrays causes valgrind to correctly report uninitialized value errors
- include guards prevent the same header from being expanded multiple times, causing duplicate definition errors
- makefile tabs vs spaces — make will silently fail if recipe lines use spaces instead of tabs

---

## what the paper says vs what is implemented

| component | paper | this implementation |
|---|---|---|
| layers | n=6 | n=2 (configurable) |
| d_model | 512 | 8 (configurable) |
| heads | 8 | 2 (configurable) |
| d_ff | 2048 | 32 (configurable) |
| attention | scaled dot-product | yes |
| causal mask | yes | yes |
| positional encoding | sinusoidal | yes |
| layer norm | yes | yes |
| residual connections | yes | yes |
| weight initialization | xavier | random uniform / 10 |
| training | yes | forward pass only |

---

## what is next

-  python numerical verification (assert c output matches numpy within 1e-3)
-  copy task training (cross entropy + sgd + backprop)
-  attention visualization (heatmaps showing head specialization)
- [ ] rotary positional embeddings (rope) — used in llama, qwen, deepseek
- [ ] rmsnorm — computationally cheaper than layernorm, used in modern llms
- [ ] weight loading from pytorch (load real gpt-2 weights and run inference)

---

## reference

vaswani, a., shazeer, n., parmar, n., uszkoreit, j., jones, l., gomez, a. n., kaiser, l., & polosukhin, i. (2017). attention is all you need. advances in neural information processing systems, 30.

inspired by andrej karpathy's llama2.c and llm.c projects.

---

## author

jonah paulin joyce  
btech computer science (cyber security), national forensic sciences university  
research intern, iit hyderabad (summer 26') 
- [github](https://github.com/jonahmichael/transformer-in-c)
- [linkedin](https://linkedin.com/in/jonahpaulinjoyce)