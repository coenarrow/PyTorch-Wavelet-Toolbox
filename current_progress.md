# Progress

## MPS Compatibility Status

| Module | Status | Notes |
|--------|--------|-------|
| continuous_transform | [x] MPS compatible | Standard wavelets work; differentiable wavelets need float64 (MPS limitation) |
| conv_transform | [ ] | |
| conv_transform_2 | [ ] | |
| conv_transform_3 | [ ] | |
| matmul_transform | [ ] | |
| matmul_transform_2 | [ ] | |
| matmul_transform_3 | [ ] | |
| separable_conv_transform | [ ] | |
| sparse_math | [ ] | |
| stationary_transform | [ ] | |
| wavelets_learnable | [ ] | Experimental |

## Future Optimizations

### CWT Scale Loop Vectorization
The CWT currently iterates over scales in a Python loop, limiting GPU parallelism.
Recommended approach: Group scales by FFT size and batch operations within groups.
See analysis in plan file: `docs/plans/2026-01-30-mps-continuous-transform.md`