# Decoder 


Q. Why does the decoder need masking?

During training, the decoder sees the entire output sequence at once. Without masking, position 3 could cheat and look at positions 4, 5, 6... which it shouldn't know yet.

(Here, i cant explain everything so read about it from my blog...
the link of it is in the main README.md file of this repo)

Masking blocks future positions by setting their attention scores to -infinity before softmax — so they become 0 after softmax. The decoder can only attend to past and current positions.

in case u didnt know:
      softmax(-infinity) = 0

```
Position:  1    2    3    4    5
           ↓    ↓    ↓    ↓    ↓
pos 1:  [ ✅  ❌  ❌  ❌  ❌ ]
pos 2:  [ ✅  ✅  ❌  ❌  ❌ ]
pos 3:  [ ✅  ✅  ✅  ❌  ❌ ]
pos 4:  [ ✅  ✅  ✅  ✅  ❌ ]
pos 5:  [ ✅  ✅  ✅  ✅  ✅ ]

```

This is called a causal mask or look-ahead mask.

---

The decoder has 3 sub-layers instead of 2:

```
Target Input
     ↓
Masked Self-Attention    ← can't see future tokens
     ↓
Add & Norm
     ↓
Cross-Attention          ← attends to encoder output
     ↓
Add & Norm
     ↓
FFN
     ↓
Add & Norm
     ↓
Output

```
The decoder struct needs everything the encoder has, plus one extra MHA for cross-attention.

I sometimes think of what i will write in the blog if i wrtie everthing here...

in a blog, i can explain everything in detail with diagrams and stuff, but here i will just give the code and some brief explanation.

so bare with me if i dont explain everything here, but i will try to explain as much as i can.

the 2 new concepts in the decoder are that you havent seen before are:
1. Masked self-attention
2. Cross-attention

The rest of the decoder is similar to the encoder, with 3 layer norms instead of 2.

You know the intution behind the decoder is so so beautiful that it makes it mathematically complex, but uk in science you dont say "its mathematically complex" you say "its mathematically poetic" and that is what the decoder is, mathematically poetic.

but the core idea is fairly very logical and simple, and that is what makes it beautiful.

```c

typedef struct {
    MultiHeadAttention* masked_mha;   // masked self-attention
    MultiHeadAttention* cross_mha;    // cross-attention

    // 3 layer norms instead of 2
    float* gamma1; float* beta1;      // after masked MHA
    float* gamma2; float* beta2;      // after cross attention
    float* gamma3; float* beta3;      // after FFN

    // FFN weights (same as encoder)
    Tensor* W1;
    Tensor* W2;
    float* b1;
    float* b2;

    int d_model;
    int d_ff;
} DecoderBlock;

```

## decoder_create

```c
  DecoderBlock* decoder_create(int d_model, int d_ff, int num_heads) {
    DecoderBlock* decoder = malloc(sizeof(DecoderBlock));
    
    // Two MHA modules
    decoder->masked_mha = mha_create(num_heads, d_model);
    decoder->cross_mha  = mha_create(num_heads, d_model);
    
    // 3 layer norms
    decoder->gamma1 = malloc(d_model * sizeof(float));
    decoder->beta1  = malloc(d_model * sizeof(float));
    decoder->gamma2 = malloc(d_model * sizeof(float));
    decoder->beta2  = malloc(d_model * sizeof(float));
    decoder->gamma3 = malloc(d_model * sizeof(float));
    decoder->beta3  = malloc(d_model * sizeof(float));
    
    // FFN weights
    decoder->W1 = tensor_create(d_model, d_ff);
    decoder->W2 = tensor_create(d_ff, d_model);
    
    // FFN biases
    decoder->b1 = malloc(d_ff * sizeof(float));
    decoder->b2 = malloc(d_model * sizeof(float));
    
    // Config
    decoder->d_model = d_model;
    decoder->d_ff    = d_ff;
    
    return decoder;
}

```

## decoder_free 

```c
void decoder_free(DecoderBlock* decoder) {
    mha_free(decoder->masked_mha);
    mha_free(decoder->cross_mha);
    
    free(decoder->gamma1);
    free(decoder->beta1);
    free(decoder->gamma2);
    free(decoder->beta2);
    free(decoder->gamma3);
    free(decoder->beta3);
    
    tensor_free(decoder->W1);
    tensor_free(decoder->W2);
    
    free(decoder->b1);
    free(decoder->b2);
    
    free(decoder);
}
```

## decoder_forward

```c
 Tensor* decoder_forward(DecoderBlock* dec, Tensor* target, Tensor* enc_output) {
    // Step 1: Masked self-attention + residual + norm
    Tensor* mha1_out = mha_forward(dec->masked_mha, target, target, target);
    Tensor* add1 = mat_add(target, mha1_out);
    tensor_free(mha1_out);
    for (int i = 0; i < add1->rows; i++)
        layer_norm(add1->data + i * add1->cols, dec->gamma1, dec->beta1, add1->cols);

    // Step 2: Cross-attention + residual + norm
    // Q comes from decoder, K and V come from encoder output
    Tensor* mha2_out = mha_forward(dec->cross_mha, add1, enc_output, enc_output);
    Tensor* add2 = mat_add(add1, mha2_out);
    tensor_free(mha2_out);
    tensor_free(add1);
    for (int i = 0; i < add2->rows; i++)
        layer_norm(add2->data + i * add2->cols, dec->gamma2, dec->beta2, add2->cols);

    // Step 3: FFN + residual + norm
    Tensor* ff1 = mat_mul(add2, dec->W1);
    for (int i = 0; i < ff1->rows; i++)
        for (int j = 0; j < ff1->cols; j++)
            ff1->data[i * ff1->cols + j] += dec->b1[j];
    for (int i = 0; i < ff1->rows * ff1->cols; i++)
        ff1->data[i] = gelu(ff1->data[i]);
    Tensor* ff2 = mat_mul(ff1, dec->W2);
    tensor_free(ff1);
    for (int i = 0; i < ff2->rows; i++)
        for (int j = 0; j < ff2->cols; j++)
            ff2->data[i * ff2->cols + j] += dec->b2[j];
    Tensor* add3 = mat_add(add2, ff2);
    tensor_free(ff2);
    tensor_free(add2);
    for (int i = 0; i < add3->rows; i++)
        layer_norm(add3->data + i * add3->cols, dec->gamma3, dec->beta3, add3->cols);

    return add3;
}
```

`Notice step 2 — cross-attention uses add1 as Q but enc_output as K and V. That's the key difference from the encoder.`

---

I will write a file to tell you how to navigate through the codebase and understand it better, but for now, just read the code and try to understand it.