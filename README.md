# transformer-in-c

refer docs folder please

| Phase | Module | What we build |
| --- | --- | --- |
| **1** | `tensor` + `math_ops` | Tensor struct, matmul, add, softmax, layer norm, GELU |
| **2** | `attention` | Scaled dot-product attention (single head) |
| **3** | `attention` | Multi-head attention, positional encoding |
| **4** | `encoder` | Encoder block = MHA + FFN + residuals + norm |                               ---> done till here
| **5** | `decoder` | Masked MHA + cross-attention + FFN |
| **6** | `transformer` | Stack encoder + decoder, full forward pass |
| **7** | `utils` | Weight loading, tokenizer hook, inference |

I have completed Phase 1: ( 05-06-2026)
- Implemented `Tensor` struct and basic operations in `tensor.c`
- Implemented matrix multiplication, addition, softmax, layer normalization, and GELU in `math_ops.c`

I completed Phase 2: ( 08-06-2026)
- implemented self-attention model in `attention.h` and `attention.c`

I completed Phase 3: ( 18-06-2026)
- started with defining the structure of MHA and creating MHA, etc
- will resume the rest tomorrow.. ;))
- after quite some time, i found the time to resume the learning again...
- i did do some learning, but not this work



