# DecGR — (run on the FDCompCN dataset)

A runnable implementation of DecGR (decoupled graph representation with
universal–private fusion).

```
DecGR/
  data.py        FDCompCN loader
  model.py       HiRS / HeRS / SMF / classifier, with Eq.(1)-(25) in the comments
  train.py       training, validation, test, checkpoint
  config.yaml    the hyperparameters used for FDCompCN
```


The split is the one given in the paper (per-class counts, seeded):

| part | nodes | benign | fraud |
|---|---|---|---|
| train | 2,658 | 2,379 | 279 |
| val | 1,064 | 953 | 111 |
| test | 1,595 | 1,426 | 169 |

