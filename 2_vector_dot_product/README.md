# Vector Dot Product

This example shows adding a custom parallelization to a dot product between two vectors.

Since the doct product needs a global reduction, we see independant processing until the last step of an allreduce.

Run the baseline example:

```
python vector_add_baseline.py
```

And run the sharded example:

```
torchrun --nproc-per-node 8 vector_add_sharded.py
```

