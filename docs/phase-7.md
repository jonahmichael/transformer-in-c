# Phase 7- utils.c

What is Positional Encoding?

Self attention works parallelly, which is great for speed, but it doesn't know the order of the tokens.
So for a self attention model:
"Jonah Killed Anxiety" and "Anxiety Killed Jonah" would be the same, which is not what we want.(because in reality Jonah did kill Anxiety with God's grace!! Praise the Lord!!)

So to maintain the order of the tokens, we add positional encoding to the input embeddings. This is done by adding a vector to each input embedding that encodes the position of the token in the sequence.

[I will literally teach you teh full amths, the inutution in a way a 5th grade can undertsnad .. seriously!! I can PROMISE you will understand it!! but not here... Check my blog for that!!]

The formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- pos is the position of the token in the sequence (0-indexed)
- i is the dimension (0-indexed)
- d_model is the dimension of the model

In plain English:

* Every position gets a unique pattern of sin/cos waves
* Even dimensions use sin, odd dimensions use cos (imp**)
* Different frequencies for different dimensions

`PE injects position information into the embeddings.`

the function takes a seq_len and d_model and returns a Tensor of shape (seq_len, d_model) where each row is the positional encoding for that position.

```c
Tensor* positional_encoding(int seq_len, int d_model) {
    Tensor* out = tensor_create(seq_len, d_model);
    
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float divisor = powf(10000.0f, (2.0f * i) / d_model);
            
            // even dimension → sin
            tensor_set(out, pos, 2 * i,     sinf(pos / divisor));
            // odd dimension → cos
            tensor_set(out, pos, 2 * i + 1, cosf(pos / divisor));
        }
    }
    
    return out;
}
```

Read the key lines:

* 2 * i → even column → sin
* 2 * i + 1 → odd column → cos
* Same divisor used for both

