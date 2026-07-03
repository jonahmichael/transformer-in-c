# Casula Masking

we have added a new portion to our `attention.c` called the mask
What is masking?
(i wont say, read my blog y'all ;)) but seriously, i will explain it here in a simple way)

i wrote the below code but claude recommended me :
"no need for a separate mask tensor. Just modify scores directly"

so if u see in the codebase i would have direcly added the -INFINITY to the scores tensor instead of creating a new mask tensor and then adding it to the scores tensor.

its the same thing, but i just wanted to show you guys how to create a mask tensor and then add it to the scores tensor.. so that you can understand the concept of masking in a better way.
```c
// welcome back guyss!! we are in phase 8 and we are here again to add masking to the attention mechanism.. so that the decoder can not see the future tokens while predicting the next token in the sequence.. so let's do it
    Tensor* mask = tensor_create(scores->rows, scores->cols);
    for (int i = 0; i < scores->rows; i++) {
        for (int j = 0; j < scores->cols; j++) {
            if(j > i) {
                tensor_set(mask, i, j, -INFINITY);
            } else {
                tensor_set(mask, i, j, 0.0f);
            }
        }
    }

    // now add them

    for (int i = 0; i < scores->rows; i++) {
        for (int j = 0; j < scores->cols; j++) {
            tensor_set(scores, i, j,
                tensor_get(scores, i, j) + tensor_get(mask, i, j));
        }
    }
```

also we need to change the signature our current attention function always applies the mask. But the encoder doesn't need masking, only the decoder's self-attention does.
So we need to make masking optional. Change the signature to:

```c
Tensor* attention(Tensor* Q, Tensor* K, Tensor* V, int causal);
```

causal=1 → apply mask, causal=0 → skip mask.