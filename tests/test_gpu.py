"""
GPU tests for DEMNE — CUDA (NVIDIA) and ROCm (AMD).

Skip architecture
-----------------
Each GPU test is decorated with a `pytest.mark` (cuda or rocm) AND a
`pytest.mark.skipif` based on actual backend availability.

  - On a standard CPU runner (GitHub Actions ubuntu/windows/macos):
    torch.cuda.is_available() == False → automatic skip, no error.
  - On a self-hosted GPU runner:
    the skip does not trigger → the test actually runs.

The same file therefore covers both infrastructures without modification.

Test scope
----------
Tests here do NOT test DEMNE business logic (already covered by
test_pipeline_integration.py).  They verify that:

  1. torch sees the GPU and can allocate tensors on it.
  2. HuggingFace models load on the correct device without immediate OOM.
  3. The DEMNE NER pipeline (transformers) runs on GPU and produces
     correctly-shaped outputs.
  4. CPU→GPU→CPU transfers are coherent (no data corruption).
"""


import pytest

# eco2ai is mocked by the session-scoped fixture in conftest.py (autouse=True)
# no need to repeat it here.

# ---------------------------------------------------------------------------
# Module-level GPU detection — evaluated ONCE at collection time
# ---------------------------------------------------------------------------

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
    # ROCm exposes the same API as CUDA via torch.cuda on AMD
    # torch.version.hip is not None if the build is ROCm
    _ROCM_AVAILABLE = _CUDA_AVAILABLE and (getattr(torch.version, "hip", None) is not None)
    # Pure CUDA = CUDA available AND not ROCm
    _CUDA_ONLY = _CUDA_AVAILABLE and not _ROCM_AVAILABLE
except ImportError:
    torch = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False
    _ROCM_AVAILABLE = False
    _CUDA_ONLY = False

# Skip reasons reused across decorators
_NO_TORCH = "torch not installed"
_NO_CUDA = "no CUDA GPU detected (NVIDIA required)"
_NO_ROCM = "no ROCm GPU detected (AMD required)"

# Shortcuts for decorators
skip_no_torch = pytest.mark.skipif(torch is None, reason=_NO_TORCH)
skip_no_cuda = pytest.mark.skipif(not _CUDA_ONLY, reason=_NO_CUDA)
skip_no_rocm = pytest.mark.skipif(not _ROCM_AVAILABLE, reason=_NO_ROCM)
skip_no_any_gpu = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="no GPU detected")


# ===========================================================================
# BLOC 1 — Sanity checks torch / device
# ===========================================================================


@pytest.mark.cuda
@skip_no_cuda
def test_cuda_device_accessible():
    """torch.cuda.is_available() + basic allocation on NVIDIA GPU."""
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    # Minimal tensor allocation: if the GPU is OOM or misconfigured,
    # it fails here rather than in the middle of a more complex test.
    t = torch.zeros(4, 4, device=device)
    assert t.device.type == "cuda"


@pytest.mark.rocm
@skip_no_rocm
def test_rocm_device_accessible():
    """Same test for AMD GPU (ROCm exposes torch.cuda.*)."""
    assert torch.cuda.is_available()
    assert torch.version.hip is not None, "This torch build is not ROCm"
    device = torch.device("cuda:0")  # ROCm reuses the cuda alias
    t = torch.zeros(4, 4, device=device)
    assert t.device.type == "cuda"


@pytest.mark.cuda
@pytest.mark.rocm
@skip_no_any_gpu
def test_gpu_count_positive():
    """At least one GPU visible to torch."""
    assert torch.cuda.device_count() >= 1


@pytest.mark.cuda
@pytest.mark.rocm
@skip_no_any_gpu
def test_device_name_not_empty():
    """torch.cuda.get_device_name() returns a non-empty string."""
    name = torch.cuda.get_device_name(0)
    assert isinstance(name, str) and len(name) > 0


# ===========================================================================
# BLOC 2 — Cohérence CPU ↔ GPU (pas de corruption de données)
# ===========================================================================


@pytest.mark.cuda
@pytest.mark.rocm
@skip_no_any_gpu
def test_cpu_to_gpu_to_cpu_round_trip():
    """
    A tensor transferred CPU→GPU→CPU must be numerically identical.

    Detects memory-copy bugs and mixed-precision issues
    (float32 on CPU, float16 on GPU, etc.).
    """
    import torch

    cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    gpu_tensor = cpu_tensor.to("cuda")
    back_to_cpu = gpu_tensor.cpu()
    assert torch.allclose(cpu_tensor, back_to_cpu), (
        f"Round-trip CPU→GPU→CPU altered values: "
        f"{cpu_tensor.tolist()} → {back_to_cpu.tolist()}"
    )


@pytest.mark.cuda
@pytest.mark.rocm
@skip_no_any_gpu
def test_batch_matrix_multiply_on_gpu():
    """
    Basic matmul operation on GPU.

    NER pipelines perform massive matrix products inside attention layers.
    This test verifies that the cuBLAS/rocBLAS backend is functional
    before loading a full model.
    """
    import torch

    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 64, device="cuda")
    c = torch.matmul(a, b)
    assert c.shape == (64, 64)
    assert not torch.isnan(c).any(), "GPU matmul produced NaN"


# ===========================================================================
# BLOC 3 — HuggingFace Transformers sur GPU
# ===========================================================================


@pytest.mark.cuda
@pytest.mark.rocm
@pytest.mark.slow
@skip_no_any_gpu
def test_tokenizer_loads_on_any_device():
    """
    The tokenizer is CPU-only but must load without error even when a GPU
    is present (some misconfigured builds raise CUDA exceptions during
    tokenizer initialisation).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoded = tok("HER2 positif 3+", return_tensors="pt")
    assert "input_ids" in encoded
    assert encoded["input_ids"].shape[1] > 0


@pytest.mark.cuda
@pytest.mark.slow
@skip_no_cuda
def test_ner_model_inference_on_cuda():
    """
    Loads a small NER model and runs inference on an NVIDIA GPU.

    Uses distilbert-base-uncased (< 270 MB) to stay fast in CI.
    Checks logit shape and absence of NaN/Inf — not business-logic accuracy
    (covered by CPU unit tests).
    """
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    device = torch.device("cuda:0")
    model_name = "elastic/distilbert-base-uncased-finetuned-conll03-english"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    model.eval()

    inputs = tok("HER2 positive 3+ confirmed by FISH", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    assert logits.device.type == "cuda", "logits are not on GPU"
    assert not torch.isnan(logits).any(), "logits contain NaN"
    assert not torch.isinf(logits).any(), "logits contain Inf"
    # shape: (batch=1, seq_len, num_labels)
    assert logits.ndim == 3
    assert logits.shape[0] == 1


@pytest.mark.rocm
@pytest.mark.slow
@skip_no_rocm
def test_ner_model_inference_on_rocm():
    """
    Same test as CUDA but on AMD GPU (ROCm).

    ROCm exposes the same cuda API — the only difference is that
    torch.version.hip is not None, checked in the skip fixture.
    """
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    device = torch.device("cuda:0")  # ROCm reuses the cuda alias
    model_name = "elastic/distilbert-base-uncased-finetuned-conll03-english"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    model.eval()

    inputs = tok("HER2 positive 3+ confirmed by FISH", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    assert not torch.isnan(logits).any()
    assert logits.ndim == 3


# ===========================================================================
# BLOC 4 — Mixed precision (AMP) — CUDA seulement
# ===========================================================================


@pytest.mark.cuda
@pytest.mark.slow
@skip_no_cuda
def test_amp_autocast_does_not_crash():
    """
    torch.amp.autocast (automatic mixed precision) must not raise an exception.

    NER fine-tuning with DrBERT on the oncological corpus uses AMP to halve
    VRAM usage. This test verifies that the CUDA build supports AMP before
    launching a full fine-tuning run.
    """
    device = torch.device("cuda:0")
    a = torch.randn(32, 64, device=device, dtype=torch.float32)
    b = torch.randn(64, 32, device=device, dtype=torch.float32)

    with torch.amp.autocast(device_type="cuda"):
        c = torch.matmul(a, b)

    # Under AMP the result can be float16 or bfloat16 depending on the GPU
    assert c.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert not torch.isnan(c).any()


# ===========================================================================
# BLOC 5 — Rapport de configuration GPU (informatif, toujours exécuté)
# ===========================================================================


def test_gpu_environment_report():
    """
    Prints GPU information in the pytest output (-v).
    Always executed (even on CPU): useful for diagnosing runners.
    Never fails.
    """
    if torch is None:
        print("\n[GPU Report] torch not installed — all GPU tests skipped.")
        return

    print(f"\n[GPU Report] torch={torch.__version__}")
    if hasattr(torch.version, "hip") and torch.version.hip:
        print(f"  Backend : ROCm {torch.version.hip}")
    else:
        cuda_ver = torch.version.cuda or "N/A"
        print(f"  Backend : CUDA {cuda_ver}")

    n = torch.cuda.device_count()
    print(f"  GPU count : {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(f"  GPU {i} : {props.name} — {vram_gb:.1f} GB VRAM")

    if n == 0:
        print("  → No GPU detected. cuda/rocm tests will be skipped.")
