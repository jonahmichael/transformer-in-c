# Encoder

What is a residual connection?

Residual connections help prevent the information from getting distorted as it moves deeper.
Just like in a chinese whisper game where the message slowly gets blurred as it is passed to the next person...

```
output = input + MHA(input)
```

**Technical**:

As networks get deeper (6, 12, 24 layers), gradients vanish during backpropagation — early layers stop learning.

```
without residual:    with residual:

input                input
  ↓                   ↓  ↘
 MHA                 MHA  |
  ↓                   ↓   |
output              add ←-+
                      ↓
                    output
```
Even if MHA learns nothing useful, the gradient can still flow back through the shortcut. This is why Transformers can be trained 6, 12, even 96 layers deep.

The entire ENCODER block looks like this:

```
        input
          ↓
        MHA(input)
          ↓
        Add(input, MHA_out)    ← residual
          ↓
        LayerNorm
          ↓
        FFN
          ↓
        Add(prev, FFN_out)     ← residual again
          ↓
        LayerNorm
          ↓
        output
```

The encoder block needs to store:

- A MultiHeadAttention
- Two layer norms (one after MHA, one after FFN)
- FFN weights — two linear layers W1 and W2  
- FFN also needs bias vectors b1 and b2



```c
// Refer to the block diagram above to see how these pieces fit together.
typedef struct {
    MultiHeadAttention* mha;

    // Layer norm 1 (after MHA)
    float* gamma1;
    float* beta1;
    

    // Feed forward network weights
    Tensor* W1;   // (d_model, d_ff)
    Tensor* W2;   // (d_ff, d_model)

    float* b1;    // bias for layer 1
    float* b2;    // bias for layer 2

    // Layer norm 2 (after FFN)
    float* gamma2;
    float* beta2;

    int d_model;
    int d_ff;

} EncoderBlock;

```

**the signature for encoder_create. What parameters does it need?**

So the signature is:

```c
EncoderBlock* encoder_create(int d_model, int d_ff, int h);
```

We also need h — number of attention heads for the MHA inside.


the body. It needs to:

- Malloc the struct
- Create the MHA with mha_create
- Malloc gamma1, beta1, gamma2, beta2 as float arrays of size d_model
- Create W1, W2 with tensor_create
- Malloc b1, b2 as float arrays
- Set d_model and d_ff fields
- Return

### step 1:
```c

    // step 1: allocate the space to encoder block, enc
    EncoderBlock* enc = malloc(sizeof(EncoderBlock));
```
### step 2: 
```c
    // step 2: create the MHA and store it in enc->mha
    // h is the no of heads, d_model is the embedding size where h * d_k = d_model, so d_k = d_model / h
        enc->mha = mha_create(h, d_model);
```

### step 3:
```c
    // srep 3:  malloc gamma1 as a float array of size d_model
    // gamma 1, beta1, gamma2, beta2 are the parameters for layer norm. 
    // why layer norm over batch norm? check my fture blogs...
    enc->gamma1 = malloc(sizeof(float) * d_model); 


```