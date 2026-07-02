# Full Transformer with N encoder and M decoder layers

```
Input
  ↓
[Encoder 1] → [Encoder 2] → ... → [Encoder N]
                                        ↓
                                   enc_output
                                        ↓
[Decoder 1] → [Decoder 2] → ... → [Decoder N]
                                        ↓
                                      Output
```

The struct needs to store an array of encoders and an array of decoders.

```c
typedef struct {
    // array of encoder blocks
    // array of decoder blocks
    // number of layers
    // d_model, d_ff, h
} Transformer;

```

Ler's give it a try:

```c
typedef struct {
    // array of encoder blocks
    EncoderBlock** encoders;  

    // array of decoder blocks
    DecoderBlock** decoders;

    // number of layers
    int N;

    // d_model, d_ff, h
    int d_model;
    int d_ff;
    int h;
} Transformer;
```

## 1. transformer_create

Now write the signature for transformer_create. It needs to take N, d_model, d_ff, h and return a Transformer*.

```c
Transformer* transformer_create(int N, int d_model, int d_ff, int h);
```

for the body of the vody:
1. Malloc the struct
2. Malloc the array of encoder pointers — malloc(N * sizeof(EncoderBlock*))
3. Loop and create each encoder with encoder_create
4. Same for decoders
5. Set N, d_model, d_ff, h
6. Return

```c
Transformer* transformer_create(int N, int d_model, int d_ff, int h){
    // 1. Malloc the struct
    Transformer* t = malloc(sizeof(Transformer));
    // 2. Malloc the array of encoder pointers — malloc(N * sizeof(EncoderBlock*))
    t->encoders = malloc(N * sizeof(EncoderBlock*));
    // 3. Loop and create each encoder with encoder_create
    for(int i = 0; i < N; i++){
        t->encoders[i] = encoder_create(d_model, d_ff, h);
    }
    // 4. Same for decoders
    t->decoders = malloc(N * sizeof(DecoderBlock*));
    for(int i = 0; i < N; i++){
        t->decoders[i] = decoder_create(d_model, d_ff, h);
    }
    // 5. Set N, d_model, d_ff, h
    t->N = N;
    t->d_model = d_model;
    t->d_ff = d_ff;
    t->h = h;
    // 6. Return
    return t;
}
```

## 2. transformer_free

you need to:

1. Loop and free each encoder
2. Free the encoders array
3. Loop and free each decoder
4. Free the decoders array
5. Free the struct
```c
void transformer_free(Transformer* t){
  // encoders
    for(int i = 0; i < t->N; i++){
        encoder_free(t->encoders[i]);
    }
    free(t->encoders);

    // decoders
    for(int i = 0; i < t->N; i++){
        decoder_free(t->decoders[i]);
    }
    free(t->decoders);
    free(t);
}
```

## 3. transformer_forward

it needss:
1. Pass input through all N encoders in sequence
2. Pass target + final encoder output through all N decoders in sequence
3. Return final decoder output

```c

Tensor* transformer_forward(Transformer* t, Tensor* src, Tensor* tgt) {
    // 1. Pass through N encoders in sequence
    Tensor* enc_out = src;
    for (int i = 0; i < t->N; i++) {
        Tensor* next = encoder_forward(t->encoders[i], enc_out);
        if (i > 0) tensor_free(enc_out);  // free intermediate (not original src)
        enc_out = next;
    }

    // 2. Pass through N decoders in sequence
    Tensor* dec_out = tgt;
    for (int i = 0; i < t->N; i++) {
        Tensor* next = decoder_forward(t->decoders[i], dec_out, enc_out);
        if (i > 0) tensor_free(dec_out);  // free intermediate (not original tgt)
        dec_out = next;
    }

    // free final encoder output
    tensor_free(enc_out);

    // 3. Return final decoder output
    return dec_out;
}

```
