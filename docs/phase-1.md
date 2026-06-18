# Phase 1 — tensor.h

A Tensor is just a multi-dimensional array of floats.

In PyTorch we do this:
```python
x = torch.zeros(3, 4)  # 2D tensor, 3 rows, 4 cols
```

in C, we dont have such luxury, lets start designing it

tho tensor is multi dimentional, in the ram it gets stored as a single line

eg: a 3×4 matrix

```
[0,0] [0,1] [0,2] [0,3]
[1,0] [1,1] [1,2] [1,3]
[2,0] [2,1] [2,2] [2,3]     total = 3 rows and 4 columns
```

but in RAM:

```
index: 0     1     2     3     4     5     6     7     8     9     10    11
data: [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] [1,2] [1,3] [2,0] [2,1] [2,2] [2,3]
```

so if we were to design a struct

it would look like this:

```c
struct Tensor {
    int rows;
    int cols;
    float *data;
};
```

- data is a pointer using which we will dynamically allocate memory in the future.
- the need for rows and cols is so that we can access in ram..

now the general formula to access lets say [1,2] (1st row , 3rd col)??

it is in 6th pos in RAM: that is

```
--------------------------
| data[row * cols + col] |
--------------------------
```

```
[1,2] → 1 * 4 + 2 = 6   where 4 is total number of cols.
[2,0] → 2 * 4 + 0 = 8
[0,3] → 0 * 4 + 3 = 3
```

But what if you're reading an entire column? Where are those elements in memory? Are they consecutive or spread out?

They are spread out...

And why does that matter?
modern CPUs don't fetch memory one float at a time. When you access data[6], the CPU automatically pulls a whole chunk of nearby memory into cache.

So:

- Reading a row → consecutive → all in cache →is fast
- Reading a column → spread out → cache keeps missing → its slow

This is called a CACHE MISS and it's one of the biggest hidden performance killers in ML systems.

---

Q: write a function signature called tensor_create that:

- Takes rows and cols as input
- Returns a Tensor

```c
Tensor* tensor_create(int rows, int cols);
```

let's write the body of this function together.
It needs to do 3 things:

1. Allocate memory for the Tensor struct itself
2. Allocate memory for the float data
3. Set the rows and cols fields, then return the pointer

```c
Tensor* tensor_create(int rows, int cols) {
    Tensor* t = malloc(sizeof(Tensor));
    t->data = malloc(rows * cols * sizeof(float));
    t->rows = rows;
    t->cols = cols;
    return t;
}
```

What should tensor_free do, and what should its signature look like?

```c
void tensor_free(Tensor* t) {
    free(t->data);
    free(t);
}
```

SO this is our Tensor.h

```c
// #ifndef / #define / #endif — these are include guards
```

GO CHECK IT TO UNDERSTAND

Tensor.c

Now tensor.c. We have tensor_create and tensor_free done.
We need one more utility function — tensor_get and tensor_set to read and write values using row-major indexing.

```c
float tensor_get(Tensor* t, int row, int col) {
    return t->data[row * t->cols + col];
}

void tensor_set(Tensor* t, int row, int col, float val) {
    t->data[row * t->cols + col] = val;
}
```

CHECKOUT COMPLETE tensor.c

---

math_ops is where the real Transformer math lives.

We shall inpletemt in the following order:

- mat_mul - matrix multiplication
- mat_add - element-wise addition
- softmax - attention weights
- layer_norm - normalization
- gelu - feedforward activation

-----

## 1. mat_mul:

Q: If matrix A is shape (2, 3) and matrix B is shape (3, 4), what is the shape of the output matrix C?

it is 2 x 4 because

If A is (M, K) and B is (K, N), output C is (M, N).
(notice K must match — that's the shared inner dimension)

Q: how do you compute a single element C[i][j]? What's the formula?

It's the dot product of the entire i-th row of A with the entire j-th column of B.

```text
A = [1, 2, 3]   ← row i
B = [4, 5, 6]   ← col j (transposed for clarity)

C[i][j] = 1*4 + 2*5 + 3*6 = 32
```

```text
C[i][j] = Σ A[i][k] * B[k][j]   for k = 0 to K-1
```

( That k loops over the shared inner dimension.)

Q; Now try writing the signature for mat_mul first. It takes two input tensors and returns the output tensor

```c
Tensor* mat_mul( Tensor*A , Tensor* B)
```

Q; Now try writing the body. You know:

- Output shape is (A->rows, B->cols)
- You need 3 nested loops
- Inside the innermost loop: C[i][j] += A[i][k] * B[k][j]

First, just write step 1 — create the output tensor C with the right shape:

```c
Tensor* mat_mul(Tensor* A, Tensor* B) {
    Tensor* C = ___(___, ___);  // fill this in
```

so it is

```c
Tensor* C = tensor_create(A->rows, B->cols);
```

Q: Now write the 3 nested loops. You already told me what they are:

- i from 0 to A->rows
- j from 0 to B->cols
- k from 0 to ?

What does k go up to?
It's the shared inner dimension — the cols of A (which equals rows of B). So k goes from 0 to A->cols.

Q: Now try writing just the 3 nested for loops

```c
for (int i = 0; i < A->rows; i++) {
    for (int j = 0; j < B->cols; j++) {
        for (int k = 0; k < A->cols; k++) {
```

Q: Now the body of the innermost loop. We need to accumulate the sum:

But we can't use [][] syntax — we have to use tensor_get and tensor_set.

```c
float sum = 0.0f;
for (int k = 0; k < A->cols; k++) {
    sum += tensor_get(A, i, k) * tensor_get(B, k, j);
}
tensor_set(C, i, j, sum);
```

REFER CODE of mat_mul function in maths_ops.c

-----

## 3. SoftMax:

idk to explain

The softmax function is a mathematical function used primarily in machine learning and deep learning. It takes a vector of raw, unnormalized scores (often called logits) and converts them into a probability distribution where each value ranges between 0 and 1, and all values sum exactly to 1 ( Google)

For a given input vector z, the softmax function for the i-th element is calculated as:

( idk well.. i'll revise and add here)

REFER CODE SoftMax fucntion

One thing to notice — why do we subtract max before calling expf?

This is a numerical stability trick. expf of large numbers overflows to infinity. Subtracting max prevents that without changing the result mathematically.

-----

## 4. layer_norm

You studied it theoretically so let's connect theory to code.
Layer norm normalizes a vector to have mean 0 and variance 1, then scales it.

The formula is:

```text
output[i] = gamma * (x[i] - mean) / sqrt(variance + epsilon) + beta
```

Where:

- mean = average of all elements
- variance = average of squared differences from mean
- gamma, beta = learnable scale and shift parameters
- epsilon = tiny number to avoid division by zero = 1e-5f

### 4[a]: mean

```c
float compute_mean(float* x, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += x[i];
    }
    return sum / len;
}
```

### 4[b]; variance

```text
variance = average of (x[i] - mean)²
```

```c
float compute_variance(float* x, int len, float mean) {
    float avg = 0.0f;
    for (int i = 0; i < len; i++) {
        float val = x[i] - mean;
        avg += val * val;
    }
    return avg / len;
}
```

layer_norm function:

Signature is:

```c
void layer_norm(float* x, float* gamma, float* beta, int len)
```

Try writing just the body — compute mean, compute variance, then loop and apply the formula:

```text
x[i] = gamma[i] * (x[i] - mean) / sqrt(variance + epsilon) + beta[i]
```

```c
void layer_norm(float* x, float* gamma, float* beta, int len) {
    float mean = compute_mean(x, len);
    float variance = compute_variance(x, len, mean);
    float epsilon = 1e-5f;

    for (int i = 0; i < len; i++) {
        x[i] = gamma[i] * (x[i] - mean) / sqrtf(variance + epsilon) + beta[i];
    }
}
```

Read each line:

- First compute mean and variance using our helpers
- sqrtf is the float version of sqrt — always use sqrtf for floats in C
- We modify x in place — no new array needed

## 5. gelu

gelu activation used in the Transformer's feedforward network.

The formula is:

```text
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

```c
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
```

0.7978845608 is just sqrt(2/π) precomputed.