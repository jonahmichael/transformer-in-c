# Attention Mechanism

( PLease read the Attention Is All You Need.pdf before starting Phase 2 because you may not be able to follow
with the concepts w/o a clear understanding of the concepts involved.

I will soon be writing a blog named "Tranformer Archietecture explained for 5 y/o"
)

-------

In the Attention Is All You Need paper, the attention formula is:

```text
Attention(Q, K, V) = softmax(QK^T / √d_k) * V            ( hope u are aware? )
```

Q. Before writing any code tell me what Q, K, and V stand for and what they represent conceptually??

```text
Q = Query, K = Key, V = Value
```

Side story: When i watched CampusX videos to understand tranformer architecture, he actually used a example to explain
these 3: Q,K,V ... I'll add it in the blog .. it is by far the best analogy used to explain a CS concept

back to the formula:

```text
softmax(QK^T / √d_k) * V
```

let us divide them into portions as below:

```text
Step 1: QK^T                     → how much does each query match each key?
Step 2: divide by √d_k           → scale down to prevent huge values
Step 3: softmax                  → convert scores to probabilities (sum to 1)
Step 4: multiply by V            → get weighted sum of values
```

Jsut a quick look at Step 2: why do we divide by √d_k? What problem does it solve?

as d_k gets large, the dot products QK^T get very large too (d_k is the dimensionality of Key(K))
which pushes softmax into regions where gradients are extremely small so dividing them by by √d_k keeps the values in a stable range.

LET US GO STEP BY STEP FROM NOW ON:

## Step1 : QK^T

Q, K, V are all shape (seq_len, d_k).

Q. What shape does K need to be transposed to, and what will the output shape of QK^T be?

(please refer LA before coming here, it is the base.. )
(uk i really really wish at times, that i could teach everything here but it would become way too cluttered)

So we need a transpose function first.

Q. Given our Tensor struct, write the signature for a transpose function — it takes a Tensor and returns a new transposed Tensor:

```c
Tensor* transpose ( Tensor* K);
```

Q. Now write the body.
The logic is simple:

- Output shape is (K->cols, K->rows) — flipped
- For every element: out[j][i] = K[i][j]

```c
Tensor* transpose(Tensor* K) {
    Tensor* out = tensor_create(K->cols, K->rows);
    for (int i = 0; i < K->rows; i++) {
        for (int j = 0; j < K->cols; j++) {
            tensor_set(out, j, i, tensor_get(K, i, j));
        }
    }
    return out;
}
```

Now we have transpose. Next ste of Step 1 of attention, compute QK^T

Q. We already have mat_mul. So computing QK^T is just one line. What would it be?You said:

```c
Tensor* atten = tensor_create( Q-> rows, Q->rows);
```

```c
Tensor* scores = mat_mul(Q, transpose(K));
```

--

## Step 2: divide by √d_k

We need to divide every element in scores by √d_k.
d_k is the number of columns in K — K->cols.

```c
for( i = 0; i < scores->rows; i++){
      for( j=0 ; j < scores-> cols ; j++){
          tensor_set(scores, i, j,  tensor_get(scores, i, j) / sqrtf(K->cols));
      }}
```

--

## Step 3 — softmax

We need to apply softmax to each row of scores.
Each row represents one query's attention scores over all keys.

We already have softmax(float* x, int len). ( scores->data is a flat array. Where does row i start in that flat array?)

```c
for (int i = 0; i < scores->rows; i++) {
    softmax(scores->data + i * scores->cols, scores->cols);
}
```

[See how scores->data + i * scores->cols is just the row-major formula]

--

## Step 4: multiply by V

Q. We have scores of shape (seq_len, seq_len) and V of shape (seq_len, d_k). What is the output shape and how do we compute it?

--

That's it, now refer to attention.c u'll undertsand everything

wait, i forgot to mention smtg

Notice I added tensor_free(Kt) and tensor_free(scores) — we created them inside the function so we must free them before returning. Otherwise memory leak!

--

Also about what is .h files?

declarations, not definitions...

So attention.h should have:

- Include guards
- The #include for tensor.h
- The function declarations for transpose and attention