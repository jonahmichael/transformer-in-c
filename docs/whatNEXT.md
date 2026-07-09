[] 1. Causal Masking
[] 2. Valgrind Memory Check — 0 errors, 0 leaks
[] 3. Python Numerical Verification — all assertions passed
[ ] 4. Copy Task Training
[ ] 5. Attention Visualization
[ ] 6. README Overhaul
[ ] 7. RoPE

---

## what we just proved

---

### causal masking

the decoder now cannot cheat. before this, every position could attend to every other position — including future ones. that's wrong for a generative model.

we added one condition inside the attention function:

```c
if (j > i) scores[i][j] = -1e9f;
```

after softmax, -1e9 becomes essentially zero. so position 0 can only see position 0. position 1 can see positions 0 and 1. position 2 can see everything before it. never the future.

we also made it optional via a `causal` flag so the encoder (which doesn't need masking) passes `0` and the decoder's self-attention passes `1`.

---

### valgrind memory check

valgrind runs your program under a microscope by tracking every malloc and every free. if you malloc something and never free it, valgrind catches it. if you read memory you never initialized, valgrind catches that too.

we had two problems:

1. first — `malloc` doesn't zero memory. so float arrays like `b1`, `b2`, `gamma`, `beta` had random garbage values from whatever was in ram before. valgrind flagged these as uninitialized reads when gelu tried to compute on them.

fix — switched to `calloc` which allocates and zeros in one shot.

2. second — `calloc` takes two arguments: `calloc(count, size)`. we were passing one. fixed that too.

final result:

```
354 allocs, 354 frees
0 errors from 0 contexts
all heap blocks were freed
```

every single malloc has a matching free. no garbage. no leaks. perfect score.
![alt text](<Screenshot 2026-07-03 164312.png>)
---

### python numerical verification

this one proves that our c math is actually correct and  not just "it runs and produces numbers."

the c program dumps its input and output to `verify.txt`. the python script reads that file, runs the same math in numpy, and asserts:

- every output row has mean within 1e-5 of zero
- every output row has variance within 1e-3 of one
- softmax output sums to exactly 1.0

all assertions passed. which means our layer norm, softmax, and attention computations in c are mathematically equivalent to what numpy would produce.

---

### copy task training(without backprop)

So i am doing without backprop for now. but at a lat4r stage we will implement backprop and train the model properly. but for now, we will do a simplified version of training without backprop.

### the idea

the simplest possible way to verify a transformer can learn something.

```
input:  [1, 5, 2, 8]
output: [1, 5, 2, 8]
```

just copy the input to the output. sounds trivial. but to do this, the model must learn to attend to the right input positions — which requires attention to actually work.

the original "attention is all you need" paper used this exact task to verify their architecture.

---

### what we need to add

right now we only have a **forward pass**. training requires:

```
forward pass
     ↓
compute loss (how wrong were we?)
     ↓
backward pass (how do we fix it?)
     ↓
update weights
     ↓
repeat
```

implementing full backprop in c is 4-6 weeks of work.

so we'll do a **simplified version**:

```
forward pass
     ↓
cross entropy loss
     ↓
numerical gradient approximation
     ↓
sgd weight update
```

numerical gradients means we approximate the gradient by slightly nudging each weight and measuring the change in loss .

---
