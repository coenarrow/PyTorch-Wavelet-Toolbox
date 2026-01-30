# MPS Support for continuous_transform.py Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the continuous wavelet transform (CWT) work efficiently on Apple Silicon MPS devices while maintaining CUDA and CPU compatibility.

**Architecture:** Replace device-specific code (`.cuda()`, `.cpu()`, `is_cuda`) with device-agnostic patterns using `.to(device)`. Convert numpy scalar operations to torch equivalents to keep computations on-device. Add MPS tests and performance benchmarks.

**Tech Stack:** PyTorch, pytest, torch.backends.mps

**Environment:** All commands must be run within the project's virtual environment. Use one of:

- `source .venv/bin/activate` then run commands, OR
- Prefix commands with `uv run`, e.g., `uv run pytest ...`

---

## Task 1: Create Branch and Baseline Tests

**Files:**
- Test: `tests/test_cwt.py`

**Step 1: Create the feature branch**

```bash
git checkout -b mps/continuous_transform
```

**Step 2: Run existing tests to establish baseline**

Run: `uv run pytest tests/test_cwt.py -v`
Expected: All existing tests PASS

**Step 3: Commit branch creation**

```bash
git add -A
git commit -m "chore: create mps/continuous_transform branch"
```

---

## Task 2: Add Device-Agnostic Test Infrastructure

**Files:**
- Modify: `tests/test_cwt.py`

**Step 1: Write failing test for MPS device support**

Add this test after `test_cwt_cuda` (around line 66):

```python
@pytest.mark.parametrize("wavelet", ["mexh", "morl", "cgau6", "gaus4"])
def test_cwt_device_agnostic(wavelet: str) -> None:
    """Test CWT works on CPU, CUDA, and MPS devices."""
    # Determine available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        pytest.skip("No GPU device available")

    t = np.linspace(-2, 2, 200, endpoint=False)
    sig = np.sin(2 * np.pi * 7 * t)
    data_cpu = torch.from_numpy(sig.astype(np.float32))
    data_device = data_cpu.to(device)
    scales = np.arange(1, 16)

    # Run on device
    coefs_device, freqs = ptwt.cwt(data_device, scales, wavelet)

    # Verify output is on correct device
    assert coefs_device.device.type == device.type, (
        f"Output on {coefs_device.device}, expected {device}"
    )

    # Verify numerical correctness vs CPU
    coefs_cpu, _ = ptwt.cwt(data_cpu, scales, wavelet)
    torch.testing.assert_close(
        coefs_device.cpu(), coefs_cpu, rtol=1e-4, atol=1e-5
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cwt.py::test_cwt_device_agnostic -v`
Expected: FAIL (on MPS) or SKIP (on CPU-only machine) - if on Mac with MPS, expect device placement errors

**Step 3: Commit the failing test**

```bash
git add tests/test_cwt.py
git commit -m "test: add device-agnostic CWT test for MPS support"
```

---

## Task 3: Fix Device Handling for Differentiable Wavelets

**Files:**
- Modify: `src/ptwt/continuous_transform.py:82-84`
- Modify: `src/ptwt/continuous_transform.py:152-154`

**Step 1: Fix first `.cuda()` call (lines 82-84)**

Replace:
```python
    if isinstance(wavelet, torch.nn.Module):
        if data.is_cuda:
            wavelet.cuda()
```

With:
```python
    if isinstance(wavelet, torch.nn.Module):
        wavelet = wavelet.to(data.device)
```

**Step 2: Fix second `.cuda()` call (lines 152-154)**

Replace:
```python
    if isinstance(wavelet, _DifferentiableContinuousWavelet):
        if data.is_cuda:
            wavelet.cuda()
```

With:
```python
    if isinstance(wavelet, _DifferentiableContinuousWavelet):
        wavelet = wavelet.to(data.device)
```

**Step 3: Fix `.cpu()` call (line 142)**

Replace:
```python
        wavelet.cpu()
```

With:
```python
        pass  # wavelet stays on original device; moved back at end if needed
```

Note: The wavelet is moved to data.device at line 83 and again at line 153, so no explicit cpu() needed.

**Step 4: Run tests to check progress**

Run: `uv run pytest tests/test_cwt.py::test_cwt_device_agnostic -v`
Expected: May still fail due to numpy operations, but device movement errors should be resolved

**Step 5: Commit device handling fixes**

```bash
git add src/ptwt/continuous_transform.py
git commit -m "fix: use device-agnostic .to(device) instead of .cuda()/.cpu()"
```

---

## Task 4: Fix numpy/torch Mixed Operations

**Files:**
- Modify: `src/ptwt/continuous_transform.py:127`
- Modify: `src/ptwt/continuous_transform.py:132`

**Step 1: Fix np.sqrt in coefficient calculation (line 127)**

Replace:
```python
        coef = -np.sqrt(scale) * torch.diff(conv, dim=-1)
```

With:
```python
        coef = -torch.sqrt(torch.tensor(scale, device=data.device, dtype=data.dtype)) * torch.diff(conv, dim=-1)
```

**Step 2: Fix np.floor/np.ceil in slicing (line 132)**

Replace:
```python
            coef = coef[..., int(np.floor(d)) : -int(np.ceil(d))]
```

With:
```python
            coef = coef[..., int(d // 1) : -int(-(-d // 1))]
```

Note: `d // 1` is floor for positive d, `-(-d // 1)` is ceiling. Since d > 0 in this branch, this is equivalent.

Alternative (clearer):
```python
            d_floor = int(d)
            d_ceil = int(d) + (1 if d % 1 else 0)
            coef = coef[..., d_floor : -d_ceil]
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_cwt.py::test_cwt_device_agnostic -v`
Expected: PASS (or closer to passing)

**Step 4: Run full CWT test suite**

Run: `uv run pytest tests/test_cwt.py -v`
Expected: All tests PASS

**Step 5: Commit numpy fixes**

```bash
git add src/ptwt/continuous_transform.py
git commit -m "fix: replace numpy ops with torch/python for MPS compatibility"
```

---

## Task 5: Add Performance Benchmark Test

**Files:**
- Modify: `tests/test_cwt.py`

**Step 1: Write performance benchmark test**

Add at end of file:

```python
@pytest.mark.slow
def test_cwt_performance_benchmark() -> None:
    """Benchmark CWT performance on available devices."""
    import time

    # Determine available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_name = "MPS"
    else:
        pytest.skip("No GPU device available for benchmark")

    # Large signal for meaningful benchmark
    t = np.linspace(-2, 2, 8000, endpoint=False)
    sig = np.sin(2 * np.pi * 7 * t) + np.sin(2 * np.pi * 13 * t)
    data_cpu = torch.from_numpy(sig.astype(np.float32))
    data_device = data_cpu.to(device)
    scales = np.arange(1, 64)
    wavelet = "morl"

    # Warmup
    for _ in range(3):
        _ = ptwt.cwt(data_device, scales, wavelet)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Benchmark GPU
    n_runs = 10
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = ptwt.cwt(data_device, scales, wavelet)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    gpu_time = (time.perf_counter() - start) / n_runs

    # Benchmark CPU
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = ptwt.cwt(data_cpu, scales, wavelet)
    cpu_time = (time.perf_counter() - start) / n_runs

    speedup = cpu_time / gpu_time
    print(f"\n{device_name} benchmark: {gpu_time*1000:.2f}ms vs CPU: {cpu_time*1000:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    # Assert GPU provides benefit (at least not significantly slower)
    # Note: For small inputs, GPU may be slower due to overhead
    assert speedup > 0.5, f"{device_name} is more than 2x slower than CPU"
```

**Step 2: Run benchmark test**

Run: `uv run pytest tests/test_cwt.py::test_cwt_performance_benchmark -v -s`
Expected: PASS with timing output

**Step 3: Commit benchmark**

```bash
git add tests/test_cwt.py
git commit -m "test: add CWT performance benchmark for GPU devices"
```

---

## Task 6: Add Differentiable Wavelet MPS Test

**Files:**
- Modify: `tests/test_cwt.py`

**Step 1: Write test for differentiable wavelets on MPS**

Add after `test_nn_cwt`:

```python
@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, torch.arange(1, 16)])
@pytest.mark.parametrize("samples", [31, 32])
def test_nn_cwt_device_agnostic(scales: Any, samples: int) -> None:
    """Test differentiable CWT on GPU devices."""
    from ptwt.continuous_transform import _ShannonWavelet

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        pytest.skip("No GPU device available")

    wavelet = _ShannonWavelet("shan1-1")
    data = torch.randn(1, samples, dtype=torch.float64, device=device)

    coefs, freqs = ptwt.cwt(data, scales, wavelet)

    assert coefs.device.type == device.type
    assert coefs.shape[-1] == samples
```

**Step 2: Run test**

Run: `uv run pytest tests/test_cwt.py::test_nn_cwt_device_agnostic -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_cwt.py
git commit -m "test: add differentiable wavelet test for MPS/CUDA"
```

---

## Task 7: Run Full Test Suite and Lint

**Files:**
- All modified files

**Step 1: Run full CWT test suite**

Run: `uv run pytest tests/test_cwt.py -v`
Expected: All tests PASS

**Step 2: Run lint**

Run: `uv run nox -s lint`
Expected: PASS (or note any issues to fix)

**Step 3: Run type checking**

Run: `uv run nox -s typing`
Expected: PASS

**Step 4: Fix any lint/typing issues**

If issues found, fix and commit:
```bash
git add -A
git commit -m "style: fix lint and typing issues"
```

---

## Task 8: Update Progress and Merge

**Files:**
- Modify: `current_progress.md`

**Step 1: Update progress file**

Change:
```markdown
[] continuous_transform
```

To:
```markdown
[x] continuous_transform
```

**Step 2: Commit progress update**

```bash
git add current_progress.md
git commit -m "docs: mark continuous_transform as MPS-compatible"
```

**Step 3: Run final full test**

Run: `uv run nox -s fast-test`
Expected: All tests PASS

**Step 4: Merge to main**

```bash
git checkout main
git merge mps/continuous_transform
git branch -d mps/continuous_transform
```

---

## Verification Checklist

- [ ] All existing tests still pass
- [ ] New `test_cwt_device_agnostic` passes on MPS
- [ ] New `test_nn_cwt_device_agnostic` passes on MPS
- [ ] Performance benchmark shows GPU is being utilized (not slower than 2x CPU)
- [ ] `nox -s lint` passes
- [ ] `nox -s typing` passes
- [ ] `current_progress.md` updated
- [ ] Branch merged to main
