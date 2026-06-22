# Encoder

What is a residual connection?

Residual connections help prevent the information from getting distorted as it moves deeper.
Just like in a chinese whisper game where the message slowly gets blurred as it is passed to the next person...

```
output = input + MHA(input)
```

**Technical**:

As networks get deeper (6, 12, 24 layers), gradients vanish during backpropagation - early layers stop learning.

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
- FFN weights - two linear layers W1 and W2  
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

We also need h - number of attention heads for the MHA inside.


the body. It needs to:

1. Malloc the struct
2. Create the MHA with mha_create
3. Malloc gamma1, beta1, gamma2, beta2 as float arrays of size d_model
4. Create W1, W2 with tensor_create
5. Malloc b1, b2 as float arrays
6. Set d_model and d_ff fields
7. Return

### step 1: Malloc the struct

```c

    // step 1: allocate the space to encoder block, enc
    EncoderBlock* enc = malloc(sizeof(EncoderBlock));
```
### step 2: Create the MHA with mha_create

```c
    // step 2: create the MHA and store it in enc->mha
    // h is the no of heads, d_model is the embedding size where h * d_k = d_model, so d_k = d_model / h
        enc->mha = mha_create(h, d_model);
```

### step 3: Malloc gamma1, beta1, gamma2, beta2 as float arrays of size d_model

```c
    // srep 3:  malloc gamma1 as a float array of size d_model
    // gamma 1, beta1, gamma2, beta2 are the parameters for layer norm. 
    // why layer norm over batch norm? check my fture blogs...
    enc->gamma1 = malloc(sizeof(float) * d_model); 

    // Now do the same for beta1, gamma2, beta2 - three more lines, same pattern!

    enc->beta1 = malloc(sizeof(float) * d_model);
    enc->gamma2 = malloc(sizeof(float) * d_model);
    enc->beta2 = malloc(sizeof(float) * d_model);
```

### step 4: Create W1, W2 with tensor_create

```c
  // create W1 and W2 using tensor_create.

  // W1 shape is (d_model, d_ff)
  // W2 shape is (d_ff, d_model)
    enc->W1=tensor_create(d_model, d_ff);
    enc->W2=tensor_create(d_ff, d_model);
```

### step 5: Malloc b1, b2 as float arrays

```c
//  malloc b1 and b2:

// b1 size is d_ff (output of first FFN layer)
// b2 size is d_model (output of second FFN layer)

    enc->b1=malloc(sizeof(float) * d_ff);
    enc->b2=malloc(sizeof(float) * d_model);

```

### step 6: Set d_model and d_ff fields

```c
    enc->d_model = d_model;
    enc->d_ff = d_ff;
```
#### step 7: Return

```c
    return enc;
```

REFER       for the full code>>>

Now write encoder_free - free everything in the right order. You know the rule - free the insides before the container. Try it!

```c
void encoder_free(EncoderBlock* enc) {
    mha_free(enc->mha);
    
    free(enc->gamma1);
    free(enc->beta1);
    free(enc->gamma2);
    free(enc->beta2);
    
    tensor_free(enc->W1);
    tensor_free(enc->W2);
    
    free(enc->b1);
    free(enc->b2);
    
    free(enc);
}

```

Notice the order:
  - Free MHA first (it has its own internal mallocs)
  - Free all float arrays
  - Free tensors
  - Free the struct itself last

---

Good! Just fix the type name spelling and write all the frees. Here's the complete function:

```c
void encoder_free(EncoderBlock* enc) {
    mha_free(enc->mha);
    
    free(enc->gamma1);
    free(enc->beta1);
    free(enc->gamma2);
    free(enc->beta2);
    
    tensor_free(enc->W1);
    tensor_free(enc->W2);
    
    free(enc->b1);
    free(enc->b2);
    
    free(enc);
}
```

Notice the order:
- Free MHA first (it has its own internal mallocs)
- Free all float arrays
- Free tensors
- Free the struct itself last

---

Now the big one - `encoder_forward`. This is the full encoder pass:

this is the Figure 1: The Transformer - model architecture. in the attention is all you need paper. The encoder block is the left half of the figure, starting from the input and ending at the output of the encoder block.
(refer the pdf in the root folder- pg 3)

```
input → MHA → Add & Norm → FFN → Add & Norm → output
```

Let's go step by step. The signature is:

```c
Tensor* encoder_forward(EncoderBlock* enc, Tensor* input);
```

### 1. **Step 1** 

` run MHA. In the encoder, Q, K, and V all come from the same input (self-attention). `

```c
Tensor* mha_forward(MultiHeadAttention* mha, Tensor* Q, Tensor* K, Tensor* V);
```

use the above

```c
    // step 1: run MHA. In the encoder, Q, K, and V all come from the same input (self-attention).
    Tensor* mha_out = mha_forward(enc->mha, input, input, input);
```
### Step 2: Residual COnnection and Layer Norm

`Add the MHA output to the original input (residual connection), then apply layer normalization.`

```c
  // Residual connection
  Tensor* add1 = mat_add(input, mha_out);
  tensor_free(mha_out);

  // Layer norm applied row by row in place
  for (int i = 0; i < add1->rows; i++) {
      layer_norm(add1->data + i * add1->cols,
                enc->gamma1, enc->beta1, add1->cols);
  }


```

### Step 3: Feed Forward Network

``` 
FFN(x) = GELU(x * W1 + b1) * W2 + b2
```

Let's break down into smaller steps...
3.1-  first linear layer: x * W1 + b1
3.2. GELU activation
3.3. second linear layer: GELU_out * W2 + b2

1 and 3 are simple linear layers. 
2 is the GELU activation function.

let's start...


Two linear layers with GELU in between. Try writing the first linear layer — x * W1 + b1. You have mat_mul and mat_add, and you know how to add a bias vector to each row of the output.

heyy!! don't forget BODMAS :)) - do the matmul first, then add the bias. If you do it the other way, you'll be adding the bias to the input instead of the output of the linear layer, which is not correct.

```c
  // step 3: FFN
  // 3.1 first linear layer: x * W1 + b1
  Tensor* ff1 = mat_mul(add1, enc->W1); // (batch_size, d_ff)
  
  // add bias b1 to each row of ff1
  for (int i = 0; i < ff1->rows; i++) {
      for (int j = 0; j < ff1->cols; j++) {
          ff1->data[i * ff1->cols + j] += enc->b1[j];
      }
  }
```
---

```c
// 3.2 GELU activation - apply to every element
  for (int i = 0; i < ff1->rows * ff1->cols; i++) {
    ff1->data[i] = gelu(ff1->data[i]);
}

  // 3.3 second linear layer: GELU_out * W2 + b2
  Tensor* ff2 = mat_mul(ff1, enc->W2); // (batch_size, d_model)
  
  // add bias b2 to each row of ff2
  for (int i = 0; i < ff2->rows; i++) {
      for (int j = 0; j < ff2->cols; j++) {
          ff2->data[i * ff2->cols + j] += enc->b2[j];
      }
  }

  tensor_free(ff1);


```

### 4. Residual Connection and Layer Normalisation

```c
<!-- Same pattern as step 2 but:

Add add1 + ff2
Apply gamma2 and beta2 -->

Tensor* add2 = mat_add(add1, ff2);
  tensor_free(ff2);
  tensor_free(add1);

  // Layer norm applied row by row in place
  for (int i = 0; i < add2->rows; i++) {
      layer_norm(add2->data + i * add2->cols,
                enc->gamma2, enc->beta2, add2->cols);
  }

  return add2;



```


Now the full encoder_forwatd is in `encoder.c` file. Check it out.


For encoder.h — same pattern as before. Include guards, include attention.h, struct definition, and 3 function declarations. Try writing it!

