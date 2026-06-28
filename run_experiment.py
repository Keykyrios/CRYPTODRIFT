"""
CryptoDrift Experiment Launcher.

Downloads Llama 3.1 8B (4-bit), runs iterative refinement
across all corpus samples × all strategies, captures attention
weights, and stores results in SQLite.

Usage:
    python run_experiment.py
"""

import os
import sys

# Use CUDA PyTorch from D: drive (C: has no space)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".local_packages"))

# Download model cache to D: drive
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TEMP"] = r"D:\tmp"
os.environ["TMP"] = r"D:\tmp"

import logging
import time

# Setup logging — force UTF-8 so Unicode chars (Δ, ×) work on Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("data/experiment.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("cryptodrift")


def main():
    # Ensure data dirs exist
    os.makedirs("data/snapshots", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

    logger.info("=" * 60)
    logger.info("CryptoDrift Experiment")
    logger.info("=" * 60)

    # Step 1: Check CUDA
    logger.info("Checking GPU...")
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("No CUDA GPU detected. Cannot run experiment.")
            logger.info("CUDA available: %s", torch.cuda.is_available())
            return 1
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        logger.info("GPU: %s (%.0f MB VRAM)", gpu_name, vram_mb)
    except Exception as e:
        logger.error("GPU check failed: %s", e)
        return 1

    # Step 2: Load model
    logger.info("Loading Llama 3.1 8B (4-bit quantized)...")
    logger.info("First run will download ~5GB. Subsequent runs use cache.")
    start = time.time()

    from src.attention.model_loader import get_model_and_tokenizer, generate_refinement
    from src.attention.probe import AttentionProbe

    model, tokenizer = get_model_and_tokenizer()
    load_time = time.time() - start
    logger.info("Model loaded in %.1fs", load_time)

    # Step 3: Set up attention probe
    logger.info("Attaching attention hooks to %d layers...", 32)
    probe = AttentionProbe(model)
    probe.enable()
    logger.info("Probe attached: %d hooks active", len(probe._hooks))

    # Step 4: Set up experiment runner
    from src.experiment.runner import ExperimentRunner, RunConfig
    from src.experiment.corpus import get_corpus

    config = RunConfig(
        max_iterations=5,
        strategies=["EF", "FF", "SF", "AI"],
        db_path="data/cryptodrift.db",
        snapshot_dir="data/snapshots",
        use_gpu=True,
        max_new_tokens=1024,
        temperature=0.7,
    )

    runner = ExperimentRunner(config)
    runner.set_refinement_fn(generate_refinement)
    runner.set_tokenizer(tokenizer)
    runner.setup()

    # Step 5: Run
    samples = get_corpus()
    total_sessions = len(samples) * len(config.strategies)
    logger.info(
        "Starting experiment: %d samples x %d strategies = %d sessions, %d iterations each",
        len(samples),
        len(config.strategies),
        total_sessions,
        config.max_iterations,
    )
    logger.info("Estimated time: %d-%d hours on RTX 4050", 
                total_sessions * 2, total_sessions * 5)

    experiment_start = time.time()

    try:
        results = runner.run_full_experiment(samples, probe=probe)
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user (Ctrl+C)")
        results = []
    finally:
        runner.teardown()

    elapsed = time.time() - experiment_start
    hours = elapsed / 3600

    logger.info("=" * 60)
    logger.info("Experiment complete!")
    logger.info("Sessions completed: %d/%d", len(results), total_sessions)
    logger.info("Total time: %.1f hours", hours)
    logger.info("Results stored in: data/cryptodrift.db")
    logger.info("Snapshots in: data/snapshots/")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
