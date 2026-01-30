"""
MPS Flash Attention Benchmarking Suite

Comprehensive benchmarking for mps-flash-attention vs PyTorch SDPA.
Supports various configurations, generates reports, and compares performance.

Usage:
    # Run benchmark suite from Python
    from mps_flash_attn import benchmark
    results = benchmark.run_suite()
    benchmark.compare_vs_sdpa()
    benchmark.generate_report("report.html")

    # Run from command line
    python -m mps_flash_attn.benchmark --suite full --output report.html
"""

import torch
import torch.nn.functional as F
import time
import json
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Try to import MFA
try:
    from . import flash_attention, is_available, __version__
    _HAS_MFA = is_available()
except ImportError:
    _HAS_MFA = False
    __version__ = "unknown"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch_size: int = 1
    num_heads: int = 8
    seq_len: int = 1024
    head_dim: int = 64
    dtype: str = "float16"  # float32, float16, bfloat16
    is_causal: bool = False
    window_size: int = 0  # 0 = full attention
    warmup_runs: int = 5
    benchmark_runs: int = 20
    backward: bool = False  # Whether to include backward pass


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    mfa_time_ms: Optional[float] = None
    sdpa_time_ms: Optional[float] = None
    mfa_memory_mb: Optional[float] = None
    sdpa_memory_mb: Optional[float] = None
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None
    max_diff: Optional[float] = None
    rel_diff_percent: Optional[float] = None
    mfa_error: Optional[str] = None
    sdpa_error: Optional[str] = None


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_str.lower()]


def get_memory_mb() -> float:
    """Get current MPS memory usage in MB."""
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, use driver API
        torch.mps.synchronize()
        try:
            # This is an approximation based on allocated tensors
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except AttributeError:
            return 0.0
    return 0.0


def benchmark_mfa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: BenchmarkConfig,
) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor]]:
    """Benchmark MFA forward (and optionally backward) pass."""
    if not _HAS_MFA:
        return None, None, None

    try:
        # Warmup
        for _ in range(config.warmup_runs):
            if config.backward:
                q_grad = q.clone().requires_grad_(True)
                out = flash_attention(q_grad, k, v, is_causal=config.is_causal, window_size=config.window_size)
                out.sum().backward()
            else:
                _ = flash_attention(q, k, v, is_causal=config.is_causal, window_size=config.window_size)
        torch.mps.synchronize()

        # Memory before
        mem_before = get_memory_mb()

        # Benchmark
        start = time.perf_counter()
        for _ in range(config.benchmark_runs):
            if config.backward:
                q_grad = q.clone().requires_grad_(True)
                out = flash_attention(q_grad, k, v, is_causal=config.is_causal, window_size=config.window_size)
                out.sum().backward()
            else:
                out = flash_attention(q, k, v, is_causal=config.is_causal, window_size=config.window_size)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        # Memory after
        mem_after = get_memory_mb()

        time_ms = (elapsed / config.benchmark_runs) * 1000
        mem_mb = max(0, mem_after - mem_before)

        # Get output for correctness check
        final_out = flash_attention(q, k, v, is_causal=config.is_causal, window_size=config.window_size)
        torch.mps.synchronize()

        return time_ms, mem_mb, final_out

    except Exception as e:
        return None, None, str(e)


def benchmark_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: BenchmarkConfig,
) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor]]:
    """Benchmark PyTorch SDPA forward (and optionally backward) pass."""
    try:
        # Warmup
        for _ in range(config.warmup_runs):
            if config.backward:
                q_grad = q.clone().requires_grad_(True)
                out = F.scaled_dot_product_attention(q_grad, k, v, is_causal=config.is_causal)
                out.sum().backward()
            else:
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=config.is_causal)
        torch.mps.synchronize()

        # Memory before
        mem_before = get_memory_mb()

        # Benchmark
        start = time.perf_counter()
        for _ in range(config.benchmark_runs):
            if config.backward:
                q_grad = q.clone().requires_grad_(True)
                out = F.scaled_dot_product_attention(q_grad, k, v, is_causal=config.is_causal)
                out.sum().backward()
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=config.is_causal)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        # Memory after
        mem_after = get_memory_mb()

        time_ms = (elapsed / config.benchmark_runs) * 1000
        mem_mb = max(0, mem_after - mem_before)

        # Get output for correctness check
        final_out = F.scaled_dot_product_attention(q, k, v, is_causal=config.is_causal)
        torch.mps.synchronize()

        return time_ms, mem_mb, final_out

    except Exception as e:
        return None, None, str(e)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark with the given configuration."""
    result = BenchmarkResult(config=config)

    # Create tensors
    dtype = get_dtype(config.dtype)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(42)
    q = torch.randn(
        config.batch_size, config.num_heads, config.seq_len, config.head_dim,
        device=device, dtype=dtype
    )
    k = torch.randn(
        config.batch_size, config.num_heads, config.seq_len, config.head_dim,
        device=device, dtype=dtype
    )
    v = torch.randn(
        config.batch_size, config.num_heads, config.seq_len, config.head_dim,
        device=device, dtype=dtype
    )

    # Benchmark MFA
    mfa_result = benchmark_mfa(q, k, v, config)
    if isinstance(mfa_result[2], str):
        result.mfa_error = mfa_result[2]
    else:
        result.mfa_time_ms = mfa_result[0]
        result.mfa_memory_mb = mfa_result[1]
        mfa_out = mfa_result[2]

    # Benchmark SDPA (only if no window_size, SDPA doesn't support sliding window)
    if config.window_size == 0:
        sdpa_result = benchmark_sdpa(q, k, v, config)
        if isinstance(sdpa_result[2], str):
            result.sdpa_error = sdpa_result[2]
        else:
            result.sdpa_time_ms = sdpa_result[0]
            result.sdpa_memory_mb = sdpa_result[1]
            sdpa_out = sdpa_result[2]

        # Calculate speedup and correctness
        if result.mfa_time_ms and result.sdpa_time_ms:
            result.speedup = result.sdpa_time_ms / result.mfa_time_ms

        if result.mfa_memory_mb and result.sdpa_memory_mb and result.sdpa_memory_mb > 0:
            result.memory_reduction = result.sdpa_memory_mb / max(result.mfa_memory_mb, 0.1)

        # Correctness check
        if mfa_out is not None and sdpa_out is not None:
            diff = (mfa_out.float() - sdpa_out.float()).abs()
            result.max_diff = diff.max().item()
            result.rel_diff_percent = (diff / (sdpa_out.float().abs() + 1e-6)).mean().item() * 100

    return result


def run_suite(
    seq_lengths: List[int] = None,
    dtypes: List[str] = None,
    batch_sizes: List[int] = None,
    head_dims: List[int] = None,
    num_heads_list: List[int] = None,
    include_causal: bool = True,
    include_window: bool = True,
    include_backward: bool = False,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """
    Run a comprehensive benchmark suite.

    Args:
        seq_lengths: List of sequence lengths to test. Default: [512, 1024, 2048, 4096, 8192]
        dtypes: List of dtypes to test. Default: ["float16", "bfloat16"]
        batch_sizes: List of batch sizes. Default: [1, 4]
        head_dims: List of head dimensions. Default: [64, 128]
        num_heads_list: List of num_heads. Default: [8]
        include_causal: Include causal attention tests. Default: True
        include_window: Include sliding window tests. Default: True
        include_backward: Include backward pass tests. Default: False
        verbose: Print progress. Default: True

    Returns:
        List of BenchmarkResult objects
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]
    if dtypes is None:
        dtypes = ["float16", "bfloat16"]
    if batch_sizes is None:
        batch_sizes = [1, 4]
    if head_dims is None:
        head_dims = [64, 128]
    if num_heads_list is None:
        num_heads_list = [8]

    results = []
    total_configs = (
        len(seq_lengths) * len(dtypes) * len(batch_sizes) * len(head_dims) * len(num_heads_list)
        * (2 if include_causal else 1)
        * (2 if include_backward else 1)
    )
    if include_window:
        total_configs += len([s for s in seq_lengths if s >= 2048]) * len(dtypes) * len(batch_sizes)

    current = 0

    for dtype in dtypes:
        for batch_size in batch_sizes:
            for num_heads in num_heads_list:
                for head_dim in head_dims:
                    for seq_len in seq_lengths:
                        # Non-causal
                        config = BenchmarkConfig(
                            batch_size=batch_size,
                            num_heads=num_heads,
                            seq_len=seq_len,
                            head_dim=head_dim,
                            dtype=dtype,
                            is_causal=False,
                        )

                        if verbose:
                            current += 1
                            print(f"[{current}/{total_configs}] B={batch_size} H={num_heads} N={seq_len} D={head_dim} {dtype}", end="", flush=True)

                        result = run_benchmark(config)
                        results.append(result)

                        if verbose:
                            if result.speedup:
                                print(f" -> MFA: {result.mfa_time_ms:.2f}ms, SDPA: {result.sdpa_time_ms:.2f}ms, Speedup: {result.speedup:.2f}x")
                            elif result.mfa_time_ms:
                                print(f" -> MFA: {result.mfa_time_ms:.2f}ms")
                            else:
                                print(f" -> Error: {result.mfa_error or result.sdpa_error}")

                        # Causal
                        if include_causal:
                            config = BenchmarkConfig(
                                batch_size=batch_size,
                                num_heads=num_heads,
                                seq_len=seq_len,
                                head_dim=head_dim,
                                dtype=dtype,
                                is_causal=True,
                            )

                            if verbose:
                                current += 1
                                print(f"[{current}/{total_configs}] B={batch_size} H={num_heads} N={seq_len} D={head_dim} {dtype} causal", end="", flush=True)

                            result = run_benchmark(config)
                            results.append(result)

                            if verbose:
                                if result.speedup:
                                    print(f" -> MFA: {result.mfa_time_ms:.2f}ms, SDPA: {result.sdpa_time_ms:.2f}ms, Speedup: {result.speedup:.2f}x")
                                elif result.mfa_time_ms:
                                    print(f" -> MFA: {result.mfa_time_ms:.2f}ms")
                                else:
                                    print(f" -> Error: {result.mfa_error or result.sdpa_error}")

                        # Backward pass tests
                        if include_backward:
                            for is_causal in [False, True] if include_causal else [False]:
                                config = BenchmarkConfig(
                                    batch_size=batch_size,
                                    num_heads=num_heads,
                                    seq_len=seq_len,
                                    head_dim=head_dim,
                                    dtype=dtype,
                                    is_causal=is_causal,
                                    backward=True,
                                )

                                if verbose:
                                    current += 1
                                    causal_str = " causal" if is_causal else ""
                                    print(f"[{current}/{total_configs}] B={batch_size} H={num_heads} N={seq_len} D={head_dim} {dtype}{causal_str} backward", end="", flush=True)

                                result = run_benchmark(config)
                                results.append(result)

                                if verbose:
                                    if result.speedup:
                                        print(f" -> MFA: {result.mfa_time_ms:.2f}ms, SDPA: {result.sdpa_time_ms:.2f}ms, Speedup: {result.speedup:.2f}x")
                                    elif result.mfa_time_ms:
                                        print(f" -> MFA: {result.mfa_time_ms:.2f}ms")
                                    else:
                                        print(f" -> Error: {result.mfa_error or result.sdpa_error}")

    # Sliding window tests (only for longer sequences where it makes sense)
    if include_window:
        window_sizes = [2048, 4096]
        for dtype in dtypes:
            for batch_size in batch_sizes[:1]:  # Just first batch size
                for seq_len in [s for s in seq_lengths if s >= 4096]:
                    for window_size in window_sizes:
                        if window_size < seq_len:
                            config = BenchmarkConfig(
                                batch_size=batch_size,
                                num_heads=8,
                                seq_len=seq_len,
                                head_dim=64,
                                dtype=dtype,
                                is_causal=True,  # Sliding window usually with causal
                                window_size=window_size,
                            )

                            if verbose:
                                current += 1
                                print(f"[{current}/{total_configs}] B={batch_size} H=8 N={seq_len} D=64 {dtype} window={window_size}", end="", flush=True)

                            result = run_benchmark(config)
                            results.append(result)

                            if verbose:
                                if result.mfa_time_ms:
                                    print(f" -> MFA: {result.mfa_time_ms:.2f}ms (sliding window, no SDPA comparison)")
                                else:
                                    print(f" -> Error: {result.mfa_error}")

    return results


def compare_vs_sdpa(
    seq_lengths: List[int] = None,
    dtype: str = "float16",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quick comparison of MFA vs SDPA at various sequence lengths.

    Returns dict with summary statistics.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    results = run_suite(
        seq_lengths=seq_lengths,
        dtypes=[dtype],
        batch_sizes=[1],
        head_dims=[64],
        num_heads_list=[8],
        include_causal=False,
        include_window=False,
        include_backward=False,
        verbose=verbose,
    )

    # Compute summary
    speedups = [r.speedup for r in results if r.speedup is not None]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    max_speedup = max(speedups) if speedups else 0
    min_speedup = min(speedups) if speedups else 0

    crossover_seq = None
    for r in results:
        if r.speedup and r.speedup > 1.0 and crossover_seq is None:
            crossover_seq = r.config.seq_len

    return {
        "average_speedup": avg_speedup,
        "max_speedup": max_speedup,
        "min_speedup": min_speedup,
        "crossover_seq_len": crossover_seq,
        "num_tests": len(results),
        "results": results,
    }


def generate_report(
    results: List[BenchmarkResult],
    output_path: str,
    title: str = "MPS Flash Attention Benchmark Report",
) -> None:
    """
    Generate an HTML report from benchmark results.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path to write HTML report
        title: Report title
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .speedup-good {{ color: #2e7d32; font-weight: bold; }}
        .speedup-bad {{ color: #c62828; }}
        .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .error {{ color: #c62828; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated with mps-flash-attention v{__version__}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total benchmarks: {len(results)}</p>
"""

    # Calculate summary stats
    speedups = [r.speedup for r in results if r.speedup is not None]
    if speedups:
        html += f"""
        <p>Average speedup: <strong>{sum(speedups)/len(speedups):.2f}x</strong></p>
        <p>Max speedup: <strong>{max(speedups):.2f}x</strong></p>
        <p>Min speedup: <strong>{min(speedups):.2f}x</strong></p>
"""

    html += """
    </div>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Config</th>
            <th>MFA (ms)</th>
            <th>SDPA (ms)</th>
            <th>Speedup</th>
            <th>Max Diff</th>
            <th>Rel Diff %</th>
        </tr>
"""

    for r in results:
        config_str = f"B={r.config.batch_size} H={r.config.num_heads} N={r.config.seq_len} D={r.config.head_dim} {r.config.dtype}"
        if r.config.is_causal:
            config_str += " causal"
        if r.config.window_size > 0:
            config_str += f" win={r.config.window_size}"
        if r.config.backward:
            config_str += " backward"

        mfa_str = f"{r.mfa_time_ms:.2f}" if r.mfa_time_ms else (f'<span class="error">{r.mfa_error}</span>' if r.mfa_error else "-")
        sdpa_str = f"{r.sdpa_time_ms:.2f}" if r.sdpa_time_ms else (f'<span class="error">{r.sdpa_error}</span>' if r.sdpa_error else "-")

        if r.speedup:
            speedup_class = "speedup-good" if r.speedup >= 1.0 else "speedup-bad"
            speedup_str = f'<span class="{speedup_class}">{r.speedup:.2f}x</span>'
        else:
            speedup_str = "-"

        max_diff_str = f"{r.max_diff:.6f}" if r.max_diff is not None else "-"
        rel_diff_str = f"{r.rel_diff_percent:.2f}%" if r.rel_diff_percent is not None else "-"

        html += f"""
        <tr>
            <td>{config_str}</td>
            <td>{mfa_str}</td>
            <td>{sdpa_str}</td>
            <td>{speedup_str}</td>
            <td>{max_diff_str}</td>
            <td>{rel_diff_str}</td>
        </tr>
"""

    html += """
    </table>
</body>
</html>
"""

    Path(output_path).write_text(html)
    print(f"Report written to {output_path}")


def main():
    """Command-line interface for benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="MPS Flash Attention Benchmark Suite")
    parser.add_argument("--suite", choices=["quick", "standard", "full"], default="standard",
                        help="Benchmark suite size (quick/standard/full)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for HTML report")
    parser.add_argument("--json", type=str, default=None,
                        help="Output file for JSON results")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None,
                        help="Sequence lengths to test")
    parser.add_argument("--dtypes", type=str, nargs="+", default=None,
                        help="Data types to test (float16, bfloat16, float32)")
    parser.add_argument("--backward", action="store_true",
                        help="Include backward pass tests")

    args = parser.parse_args()

    # Configure suite
    if args.suite == "quick":
        seq_lengths = args.seq_lengths or [1024, 4096]
        dtypes = args.dtypes or ["float16"]
        batch_sizes = [1]
        head_dims = [64]
        include_window = False
    elif args.suite == "standard":
        seq_lengths = args.seq_lengths or [512, 1024, 2048, 4096]
        dtypes = args.dtypes or ["float16", "bfloat16"]
        batch_sizes = [1, 4]
        head_dims = [64, 128]
        include_window = True
    else:  # full
        seq_lengths = args.seq_lengths or [256, 512, 1024, 2048, 4096, 8192]
        dtypes = args.dtypes or ["float32", "float16", "bfloat16"]
        batch_sizes = [1, 2, 4, 8]
        head_dims = [32, 64, 128]
        include_window = True

    print(f"MPS Flash Attention Benchmark Suite v{__version__}")
    print(f"Suite: {args.suite}")
    print(f"MFA available: {_HAS_MFA}")
    print()

    results = run_suite(
        seq_lengths=seq_lengths,
        dtypes=dtypes,
        batch_sizes=batch_sizes,
        head_dims=head_dims,
        include_window=include_window,
        include_backward=args.backward,
    )

    # Output results
    if args.output:
        generate_report(results, args.output)

    if args.json:
        json_results = [
            {
                "config": asdict(r.config),
                "mfa_time_ms": r.mfa_time_ms,
                "sdpa_time_ms": r.sdpa_time_ms,
                "speedup": r.speedup,
                "max_diff": r.max_diff,
                "rel_diff_percent": r.rel_diff_percent,
            }
            for r in results
        ]
        Path(args.json).write_text(json.dumps(json_results, indent=2))
        print(f"JSON results written to {args.json}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    speedups = [r.speedup for r in results if r.speedup is not None]
    if speedups:
        print(f"Average speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")

        # Find crossover point
        for r in sorted(results, key=lambda x: x.config.seq_len):
            if r.speedup and r.speedup > 1.0:
                print(f"MFA faster than SDPA starting at seq_len={r.config.seq_len}")
                break

    errors = [r for r in results if r.mfa_error or r.sdpa_error]
    if errors:
        print(f"\nErrors: {len(errors)} benchmarks failed")


if __name__ == "__main__":
    main()
