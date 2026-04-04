import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def convert_safetensors_to_fp8(input_path, output_path):
    """Convert a safetensors file to FP8 precision."""
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False

    print(f"\n🔄 Converting: {os.path.basename(input_path)}")
    print(f"   Source size: {get_file_size_mb(input_path):.2f} MB")

    start_time = time.time()
    fp8_tensors = {}

    try:
        with safe_open(input_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for key in tqdm(keys, desc="Converting tensors", unit="tensor"):
                tensor = f.get_tensor(key)

                if tensor.is_floating_point():
                    fp8_tensors[key] = tensor.to(torch.float8_e4m3fn)
                else:
                    fp8_tensors[key] = tensor

        print(f"💾 Saving to: {output_path}")
        save_file(fp8_tensors, output_path)

        elapsed = time.time() - start_time
        original_size = get_file_size_mb(input_path)
        new_size = get_file_size_mb(output_path)
        savings = (1 - new_size / original_size) * 100

        print(f"✅ Conversion complete in {elapsed:.2f}s")
        print(f"   New size: {new_size:.2f} MB (Saved {savings:.1f}%)")
        return True

    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        return False


def main():
    print("🔥 OmniWeaving FP8 Checkpoint Converter")
    print("=" * 50)

    ckpts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")

    files_to_convert = [
        {
            "input": os.path.join(
                ckpts_dir, "text_encoder", "ckpt", "text_encoder_model.safetensors"
            ),
            "output": os.path.join(
                ckpts_dir, "text_encoder", "ckpt", "text_encoder_model_fp8.safetensors"
            ),
        },
        {
            "input": os.path.join(
                ckpts_dir, "transformer", "diffusion_pytorch_model.safetensors"
            ),
            "output": os.path.join(
                ckpts_dir, "transformer", "diffusion_pytorch_model_fp8.safetensors"
            ),
        },
    ]

    if len(sys.argv) > 1:
        files_to_convert = []
        for i in range(1, len(sys.argv), 2):
            if i + 1 < len(sys.argv):
                files_to_convert.append(
                    {"input": sys.argv[i], "output": sys.argv[i + 1]}
                )

    success_count = 0
    for file_cfg in files_to_convert:
        if convert_safetensors_to_fp8(file_cfg["input"], file_cfg["output"]):
            success_count += 1

    print("\n" + "=" * 50)
    print(
        f"🎉 Conversion finished: {success_count}/{len(files_to_convert)} files converted."
    )
    print("\n💡 Usage Tip:")
    print("   To use these FP8 checkpoints, rename them to replace the originals,")
    print("   or update your config to point to the _fp8 files.")


if __name__ == "__main__":
    main()
