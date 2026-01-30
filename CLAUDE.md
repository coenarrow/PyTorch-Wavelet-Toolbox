# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

All development tasks are managed through **nox**. Install dependencies with `uv add <package>` (not `pip install`).

| Command | Purpose |
|---------|---------|
| `uv run nox -s test` | Run full test suite |
| `uv run nox -s fast-test` | Run tests excluding `@pytest.mark.slow` |
| `uv run nox -s format` | Auto-format with black and isort |
| `uv run nox -s lint` | Run flake8 and doc8 checks |
| `uv run nox -s typing` | Run mypy type checker |
| `uv run nox -s doctests` | Run docstring examples (requires matplotlib, pygments) |
| `uv run nox -s docs` | Build Sphinx documentation |
| `uv run nox -s coverage` | Generate HTML coverage report in htmlcov/ |

**Development workflow (before creating a PR):**
1. Ensure all unit tests pass: `uv run nox -s test`
2. Auto-fix linting issues: `uv run nox -s format`
3. Check remaining style issues: `uv run nox -s lint`

**Run a single test:**
```bash
uv run pytest tests/test_convolution_fwt.py::test_name -v
```

## Architecture

**Package:** `src/ptwt/` - Differentiable, GPU-enabled fast wavelet transforms in PyTorch.

### Transform Implementations

Two implementation strategies exist for each transform type:

1. **Convolution-based** (default): Uses `torch.nn.functional.conv*d` operations. Fast, supports various boundary modes via padding.
   - 1D: `wavedec()`, `waverec()` in `conv_transform.py`
   - 2D: `wavedec2()`, `waverec2()` in `conv_transform_2.py`
   - 3D: `wavedec3()`, `waverec3()` in `conv_transform_3.py`
   - 2D/3D separable: `fswavedec*()`, `fswaverec*()` in `separable_conv_transform.py`

2. **Sparse matrix-based**: Uses `torch.sparse.mm` with boundary filters to minimize padding. Required for boundary wavelets.
   - 1D: `MatrixWavedec`, `MatrixWaverec` in `matmul_transform.py`
   - 2D: `MatrixWavedec2`, `MatrixWaverec2` in `matmul_transform_2.py`
   - 3D: `MatrixWavedec3`, `MatrixWaverec3` in `matmul_transform_3.py`

### Other Transforms

- **Continuous wavelet transform**: `cwt()` in `continuous_transform.py` (FFT-based port of pywt.cwt)
- **Stationary wavelet transform**: `swt()`, `iswt()` in `stationary_transform.py`
- **Wavelet packets**: `WaveletPacket`, `WaveletPacket2D` in `packets.py`

### Key Types (constants.py)

- `Wavelet`: Protocol for wavelet objects (must have `dec_lo`, `dec_hi`, `rec_lo`, `rec_hi` filter attributes)
- `BoundaryMode`: Literal type for "constant", "zero", "reflect", "periodic", "symmetric"
- `WaveletCoeff1d`, `WaveletCoeff2d`, `WaveletCoeff3d`: Coefficient tuple types

### Utilities (_util.py)

Centralized helper functions:
- `_as_wavelet()`: Convert string to pywt Wavelet object
- `_get_filter_tensors()`: Extract filter tensors from wavelet
- `_get_pad()`, `_pad_symmetric()`: Padding calculations and operations
- `_check_same_device_dtype()`: Validate tensor consistency
- `_preprocess_tensor_*d()`, `_postprocess_tensor_*d()`: Batch dimension handling

## Code Style

- Line length: 80 characters (black, flake8)
- Docstrings: Google style (Napoleon)
- Type hints: Strict mypy enforcement
- All public functions need docstrings validated by darglint

## MPS Support Development Workflow

**Goal:** Update ptwt transforms to work efficiently on Apple Silicon (MPS backend).

### Branch Strategy

Each source file being modified gets its own branch:
```bash
git checkout -b mps/<module_name>   # e.g., mps/continuous_transform
# ... make changes ...
# All tests must pass before merging to main
git checkout main && git merge mps/<module_name>
git branch -d mps/<module_name>
```

### Progress Tracking

Track completion status in [current_progress.md](current_progress.md). Update the checkbox for each module as work completes.

### Requirements for Each Module

1. **Test coverage**: Ensure tests in `tests/test_<module>.py` are comprehensive enough to validate MPS changes. Add tests if needed.

2. **Performance validation**: Changes must result in actual timing improvements on MPS. Profile before/after to confirm GPU is being used efficiently, not just used.

3. **All tests passing**: Run `nox -s test` (or at minimum the relevant `pytest tests/test_<module>.py`) before merging.

### Module Files to Update

| Module | Source File | Test File |
|--------|-------------|-----------|
| continuous_transform | `src/ptwt/continuous_transform.py` | `tests/test_cwt.py` |
| conv_transform | `src/ptwt/conv_transform.py` | `tests/test_convolution_fwt.py` |
| conv_transform_2 | `src/ptwt/conv_transform_2.py` | `tests/test_convolution_fwt.py` |
| conv_transform_3 | `src/ptwt/conv_transform_3.py` | `tests/test_convolution_fwt_3.py` |
| matmul_transform | `src/ptwt/matmul_transform.py` | `tests/test_matrix_fwt.py` |
| matmul_transform_2 | `src/ptwt/matmul_transform_2.py` | `tests/test_matrix_fwt_2.py` |
| matmul_transform_3 | `src/ptwt/matmul_transform_3.py` | `tests/test_matrix_fwt_3.py` |
| separable_conv_transform | `src/ptwt/separable_conv_transform.py` | `tests/test_separable_conv_fwt.py` |
| sparse_math | `src/ptwt/sparse_math.py` | `tests/test_sparse_math.py` |
| stationary_transform | `src/ptwt/stationary_transform.py` | `tests/test_swt.py` |
| wavelets_learnable | `src/ptwt/wavelets_learnable.py` | (experimental) |
