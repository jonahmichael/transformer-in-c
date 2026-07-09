# Attention Visualization 

The plan:

- modify `attention.c` to optionally save attention weights to a global buffer
- dump them to `attention_weights.csv` after a forward pass
- plot heatmaps in Python

This is what papers like "Attention Is All You Need" use to show head specialization — one head attends to nearby words, another to the subject, etc...

a simple approach. After running a forward pass, we want to capture what the attention scores looked like after softmax for each head.