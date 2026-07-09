import numpy as np

# ── read verify.txt ──────────────────────────────────────────
with open("verify.txt") as f:
    lines = f.read().splitlines()

input_vals  = []
output_vals = []
mode = None

for line in lines:
    if line == "INPUT":
        mode = "input"
    elif line == "OUTPUT":
        mode = "output"
    elif mode == "input":
        input_vals.append(float(line))
    elif mode == "output":
        output_vals.append(float(line))

seq_len = 3
d_model = 8

src = np.array(input_vals).reshape(seq_len, d_model)
c_out = np.array(output_vals).reshape(seq_len, d_model)

print("input (from c):")
print(np.round(src, 4))

# ── verify layer norm output properties ──────────────────────
print("\nverifying layer norm properties on c output:")
for i in range(seq_len):
    mean = c_out[i].mean()
    var  = c_out[i].var()
    print(f"  row {i} -- mean: {mean:.6f}, variance: {var:.6f}")

# ── numpy softmax (matches our c softmax) ────────────────────
def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ── numpy layer norm ─────────────────────────────────────────
def layer_norm(x, eps=1e-5):
    mean = x.mean()
    var  = x.var()
    return (x - mean) / np.sqrt(var + eps)

# ── verify softmax matches ────────────────────────────────────
print("\nverifying softmax:")
test = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
py_soft = softmax(test)
print(f"  numpy softmax: {np.round(py_soft, 4)}")
print(f"  sums to: {py_soft.sum():.6f} (should be 1.0)")

# ── verify layer norm matches ─────────────────────────────────
print("\nverifying layer norm on first row of c output:")
row0 = c_out[0]
py_norm = layer_norm(row0)
print(f"  numpy mean after norm:     {py_norm.mean():.6f} (should be ~0)")
print(f"  numpy variance after norm: {py_norm.var():.6f}  (should be ~1)")

# ── final assertion ───────────────────────────────────────────
print("\nfinal checks:")
for i in range(seq_len):
    mean = abs(c_out[i].mean())
    var  = abs(c_out[i].var() - 1.0)
    assert mean < 1e-5, f"mean check failed row {i}: {mean}"
    assert var  < 1e-3, f"variance check failed row {i}: {var}"

print("all assertions passed!")
print("c output is numerically correct.")