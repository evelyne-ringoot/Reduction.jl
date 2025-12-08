Set up project as: 

   ```
   julia --project=.
   ]instantiate
   exit()
   ```
The run as

```julia
julia --project=. run.jl
```

Results on RTX4060:

| size_i         | time (ms) |
|----------------|-----------|
| 1024           | 0.034     |
| 32x1024        | 0.034     |
| 1024x1024      | 0.055     |
| 32x1024x1024   | 0.652     |
| 1024x1024x1024 | 18.204    |

For future reference the 2D timing:

| number of segments | length of segments | time (ms) |
|--------------------|--------------------|-----------|
| 32x1024            | 1024               | 85.985    |
| 1024               | 32x1024            | 85.515    |
| 32                 | 1024x1024          | 86.966    |
| 1024x1024          | 32                 | 74.421    |

