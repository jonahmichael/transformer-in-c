# transformer-in-c

I will be releasing a series of articles on the implementation of tranformers from first principles in C language.
One of the best things I learnt in the last 2 years of my college is this...
Never have I ever in this 2 years learnt a subject using research papers... it was always yt videos 
I have read more blogs, research papers, articles, books, podcasts, for this than i had for anything...

few of the coolest things I learnt in life is from bible and mathematics
game theory: talks about how to make decisions in a competitive environment
it isnt a zero-sum game...
by gate keeping these resources, i get nothing but a false sense of superiority and a false sense of being smart..
.
I have learnt a lot from these resources and i want to share it with the world... ( may not the best for YOU, but it is the best for me and i want to share it with the world, may be it helps someone else too, and if it does for even 1 person, i will be happy)

i will be releasing a series of articles on the implementation of transformers from first principles in substack and my perosnal blog page:
- [substack](https://learn-the-unknown.substack.com/)
- [personal blog](https://learn-the-unknown.github.io/)

also, here are my social media handles:
- [twitter](https://x.com/jonah_25_)
- [linkedin](https://linkedin.com/in/jonahpaulinjoyce/)
- [github](https://github.com/jonahmichael)
- [instagram](https://www.instagram.com/j218448/)
- [youtube](https://www.youtube.com/@jonah_Mi)
- [medium](https://medium.com/@jonah.paulin2596)
- [gmail](mailto:jonah.paulin2596@gmail.com)


refer docs folder please for phase wise learning & implementation of transformer in C language from scratch.

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

I have started Phase 4: ( 22-06-2026)
- started with defining the structure of EncoderBlock and creating EncoderBlock, etc
