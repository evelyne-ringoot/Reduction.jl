compilation:

```
nvcc --std=c++17 --expt-relaxed-constexpr --usefastmath -O3 -arch=sm_89 -o red red.cu -lcurand
```

 On RTX4060 we get:
| size_i         | time (ms) |
|----------------|-----------|
| 1024           | 0.005     |
| 32x1024        | 0.008     |
| 1024x1024      | 0.010     |
| 32x1024x1024   | 0.581     |
| 1024x1024x1024 | 17.407    |

For future reference the 2D timing:

| number of segments | length of segments | time (ms) |
|--------------------|--------------------|-----------|
| 32x1024            | 1024               | 0.591     |
| 1024               | 32x1024            | 0.586     |
| 32                 | 1024x1024          | 0.572     |
| 1024x1024          | 32                 | 6.047     |
| 1024x1024          | 1024               | 17.664    |
| 1024               | 1024x1024          | 17.431    |
| 32                 | 32x1024x1024       | 17.414    |
| 32x1024x1024       | 32                 | 193.412   |   