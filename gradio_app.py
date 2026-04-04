import os
import sys
import datetime
import tempfile
import shutil
import subprocess
import re
import json
from pathlib import Path

import gradio as gr

DEFAULT_MODEL_PATH = "./ckpts"
DEFAULT_OUTPUT_DIR = "./outputs"
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gradio_temp")

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "2.39:1"]
TASK_LABELS = {
    "t2v": "Text-to-Video (T2V)",
    "i2v": "First-Frame-to-Video (I2V)",
    "interpolation": "Key-Frames-to-Video",
    "reference2v": "Multi-Image-to-Video",
    "editing": "Video-to-Video Editing",
    "tiv2v": "Text-Image-Video-to-Video",
}
TASK_DESCRIPTIONS = {
    "t2v": "Generate videos from text prompts.",
    "i2v": "Animate a static image into a video guided by text.",
    "interpolation": "Generate a video conditioned on start and end frames.",
    "reference2v": "Single- or multi-subject reference-driven video generation (1-4 images).",
    "editing": "Instruction-based video manipulation and stylization.",
    "tiv2v": "Generate a video conditioned on text, image, and video inputs.",
}
CACHE_TYPES = ["deepcache", "teacache", "taylorcache"]
QUANT_TYPES = ["fp8-per-token-sgl", "fp8-per-tensor-weight-only", "fp8-per-tensor"]
INCLUDE_PATTERNS = ["double_blocks", "single_blocks", "all"]


def ensure_dirs():
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def clean_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)


def save_uploaded_files(files):
    if not files:
        return []
    clean_temp()
    paths = []
    for i, f in enumerate(files):
        if isinstance(f, str):
            src = f
        else:
            src = f
        ext = os.path.splitext(src)[1] if isinstance(src, str) else ".png"
        dst = os.path.join(TEMP_DIR, f"upload_{i}{ext}")
        shutil.copy2(src, dst)
        paths.append(dst)
    return paths


def save_uploaded_video(video):
    if video is None:
        return None
    clean_temp()
    if isinstance(video, str):
        src = video
    else:
        src = video
    dst = os.path.join(TEMP_DIR, "condition_video.mp4")
    shutil.copy2(src, dst)
    return dst


def parse_range(value):
    if not value or not value.strip():
        return []
    value = value.strip()
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start.strip()), int(end.strip()) + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def format_deepstack(value):
    if not value:
        return [8, 16, 24]
    return parse_range(value)


def build_command(args_dict, ref_image_paths, condition_video_path):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py"),
    ]

    env = os.environ.copy()
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    arg_map = [
        ("task", "--task"),
        ("prompt", "--prompt"),
        ("negative_prompt", "--negative_prompt"),
        ("model_path", "--model_path"),
        ("pipeline_config", "--pipeline_config"),
        ("aspect_ratio", "--aspect_ratio"),
        ("num_inference_steps", "--num_inference_steps"),
        ("video_length", "--video_length"),
        ("dtype", "--dtype"),
        ("seed", "--seed"),
        ("output_path", "--output_path"),
        ("fps", "--fps"),
        ("sage_blocks_range", "--sage_blocks_range"),
        ("cache_type", "--cache_type"),
        ("no_cache_block_id", "--no_cache_block_id"),
        ("cache_start_step", "--cache_start_step"),
        ("cache_end_step", "--cache_end_step"),
        ("total_steps", "--total_steps"),
        ("cache_step_interval", "--cache_step_interval"),
        ("checkpoint_path", "--checkpoint_path"),
        ("lora_path", "--lora_path"),
        ("quant_type", "--quant_type"),
        ("include_patterns", "--include_patterns"),
    ]

    for key, flag in arg_map:
        val = args_dict.get(key)
        if val is not None and val != "" and val != "None":
            if isinstance(val, bool):
                continue
            cmd.extend([flag, str(val)])

    if args_dict.get("think"):
        cmd.append("--think")
    if args_dict.get("use_sageattn"):
        cmd.extend(["--use_sageattn", "true"])
    if args_dict.get("sparse_attn"):
        cmd.extend(["--sparse_attn", "true"])
    if not args_dict.get("offloading", True):
        cmd.extend(["--offloading", "false"])
    if args_dict.get("group_offloading") is not None:
        cmd.extend(["--group_offloading", str(args_dict["group_offloading"]).lower()])
    if not args_dict.get("overlap_group_offloading", True):
        cmd.extend(["--overlap_group_offloading", "false"])
    if args_dict.get("enable_torch_compile"):
        cmd.extend(["--enable_torch_compile", "true"])
    if args_dict.get("enable_cache"):
        cmd.extend(["--enable_cache", "true"])
    if args_dict.get("only_give_text") is not None and not args_dict["only_give_text"]:
        cmd.extend(["--only_give_text", "false"])
    if args_dict.get("setclip") is not None and not args_dict["setclip"]:
        cmd.extend(["--setclip", "false"])
    if args_dict.get("use_fp8_gemm"):
        cmd.extend(["--use_fp8_gemm", "true"])
    if (
        args_dict.get("save_generation_config") is not None
        and not args_dict["save_generation_config"]
    ):
        cmd.extend(["--save_generation_config", "false"])
    if args_dict.get("quantize_4bit"):
        cmd.extend(["--quantize_4bit", "true"])

    deepstack = args_dict.get("deepstack")
    if deepstack and isinstance(deepstack, list):
        cmd.append("--deepstack")
        cmd.extend(str(x) for x in deepstack)

    if ref_image_paths:
        cmd.append("--ref_image_paths")
        cmd.extend(ref_image_paths)

    if condition_video_path:
        cmd.extend(["--condition_video_paths", condition_video_path])

    return cmd, env


def run_generation(cmd, env):
    log_lines = []
    output_video = None
    error_output = []

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                log_lines.append(line)
                if "Saved video to:" in line:
                    match = re.search(r"Saved video to:\s*(.+?)(?:\s*\(fps|$)", line)
                    if match:
                        output_video = match.group(1).strip()
                if "Traceback" in line or "Error" in line or "ERROR" in line:
                    error_output.append(line)
                yield "\n".join(log_lines), None, False

        process.wait()

        if process.returncode != 0 and not output_video:
            error_msg = f"Process exited with code {process.returncode}"
            if error_output:
                error_msg += "\n".join(error_output[-20:])
            yield "\n".join(log_lines) + f"\n\n❌ {error_msg}", None, False
        else:
            yield (
                "\n".join(log_lines) + "\n\n✅ Generation complete!",
                output_video,
                True,
            )

    except Exception as e:
        yield f"❌ Error: {str(e)}\n\n" + "\n".join(log_lines), None, False


def generate_video(
    task,
    prompt,
    negative_prompt,
    think,
    ref_images,
    condition_video,
    aspect_ratio,
    video_length,
    num_inference_steps,
    seed,
    pipeline_config,
    output_fps,
    attention_mode,
    dtype,
    offloading,
    group_offloading,
    overlap_group_offloading,
    enable_torch_compile,
    enable_cache,
    cache_type,
    no_cache_block_id,
    cache_start_step,
    cache_end_step,
    cache_step_interval,
    model_path,
    checkpoint_path,
    lora_path,
    deepstack_str,
    setclip,
    only_give_text,
    use_fp8_gemm,
    quant_type,
    include_patterns,
    save_generation_config,
    quantize_4bit,
    progress=gr.Progress(),
):
    ensure_dirs()

    if not prompt or not prompt.strip():
        yield "❌ Please enter a prompt.", None, False
        return

    ref_image_paths = save_uploaded_files(ref_images) if ref_images else []
    condition_video_path = (
        save_uploaded_video(condition_video) if condition_video else None
    )

    if task == "i2v" and not ref_image_paths:
        yield "❌ I2V task requires at least 1 reference image.", None, False
        return
    if task == "interpolation" and len(ref_image_paths) < 2:
        yield "❌ Interpolation task requires at least 2 reference images.", None, False
        return
    if task == "reference2v" and not ref_image_paths:
        yield "❌ Reference2V task requires at least 1 reference image.", None, False
        return
    if task == "editing" and not condition_video_path:
        yield "❌ Editing task requires a condition video.", None, False
        return
    if task == "tiv2v" and not condition_video_path:
        yield "❌ TIV2V task requires a condition video.", None, False
        return
    if task == "tiv2v" and not ref_image_paths:
        yield "❌ TIV2V task requires at least 1 reference image.", None, False
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{task}_{timestamp}.mp4"
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, output_filename)

    args_dict = {
        "task": task,
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip() if negative_prompt else "",
        "think": think,
        "model_path": model_path.strip() if model_path else DEFAULT_MODEL_PATH,
        "pipeline_config": pipeline_config,
        "aspect_ratio": aspect_ratio,
        "video_length": video_length,
        "num_inference_steps": num_inference_steps if num_inference_steps else None,
        "seed": seed,
        "dtype": dtype,
        "output_path": output_path,
        "fps": output_fps if output_fps else None,
        "use_sageattn": attention_mode == "SageAttention",
        "sparse_attn": attention_mode == "Sparse Attention",
        "offloading": offloading,
        "group_offloading": group_offloading,
        "overlap_group_offloading": overlap_group_offloading,
        "enable_torch_compile": enable_torch_compile,
        "enable_cache": enable_cache,
        "cache_type": cache_type,
        "no_cache_block_id": no_cache_block_id,
        "cache_start_step": cache_start_step,
        "cache_end_step": cache_end_step,
        "cache_step_interval": cache_step_interval,
        "checkpoint_path": checkpoint_path.strip() if checkpoint_path else None,
        "lora_path": lora_path.strip() if lora_path else None,
        "deepstack": format_deepstack(deepstack_str),
        "setclip": setclip,
        "only_give_text": only_give_text,
        "use_fp8_gemm": use_fp8_gemm,
        "quant_type": quant_type,
        "include_patterns": include_patterns,
        "save_generation_config": save_generation_config,
        "quantize_4bit": quantize_4bit,
    }

    cmd, env = build_command(args_dict, ref_image_paths, condition_video_path)

    yield from run_generation(cmd, env)


def update_media_inputs(task):
    show_ref_images = task in ("i2v", "interpolation", "reference2v", "tiv2v")
    show_condition_video = task in ("editing", "tiv2v")
    show_think = task in ("t2v", "i2v", "interpolation")

    ref_max = {"i2v": 1, "interpolation": 10, "reference2v": 4, "tiv2v": 1}.get(task, 4)
    ref_min = {"i2v": 1, "interpolation": 2, "reference2v": 1, "tiv2v": 1}.get(task, 0)

    ref_label = f"Reference Images ({ref_min}-{ref_max})"
    if task == "interpolation":
        ref_label = "Reference Images (first frame, last frame, optional intermediates)"
    elif task == "i2v":
        ref_label = "Reference Image (first frame)"

    return (
        gr.update(
            visible=show_ref_images,
            label=ref_label,
            file_count="multiple" if ref_max > 1 else "single",
        ),
        gr.update(visible=show_condition_video),
        gr.update(visible=show_think),
    )


def update_cache_section(enable_cache):
    return gr.update(visible=enable_cache)


def create_ui():
    ensure_dirs()

    with gr.Blocks(title="OmniWeaving Video Generation") as app:
        gr.Markdown(
            """
# 🎬 OmniWeaving Video Generation

Unified video generation with free-form composition and reasoning. Built upon [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5).

**Fork & Windows Support:** [benjiyaya/OmniWeaving](https://github.com/benjiyaya/OmniWeaving) | Updated by Benji
"""
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                with gr.Accordion("📋 Task & Prompt", open=True):
                    task = gr.Dropdown(
                        choices=list(TASK_LABELS.keys()),
                        value="t2v",
                        label="Task Type",
                        info="Select the generation task",
                    )
                    task_desc = gr.Markdown(value=TASK_DESCRIPTIONS["t2v"])
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the video you want to generate...",
                        lines=3,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                        lines=2,
                        value="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                    )
                    think = gr.Checkbox(
                        label="🧠 Think Mode",
                        value=False,
                        info="Activate MLLM reasoning to generate an enhanced prompt before video generation (t2v/i2v/interpolation only)",
                    )

                with gr.Accordion("🎨 Media Inputs", open=True):
                    ref_images = gr.File(
                        label="Reference Images (0-4)",
                        file_types=["image"],
                        type="filepath",
                        file_count="multiple",
                    )
                    condition_video = gr.File(
                        label="Condition Video",
                        file_types=["video"],
                        type="filepath",
                        visible=False,
                    )

                with gr.Accordion("⚙️ Generation Settings", open=True):
                    aspect_ratio = gr.Dropdown(
                        choices=ASPECT_RATIOS,
                        value="16:9",
                        label="Aspect Ratio",
                    )
                    with gr.Row():
                        video_length = gr.Slider(
                            minimum=33,
                            maximum=161,
                            value=81,
                            step=1,
                            label="Video Length (frames)",
                            info="33-161 frames",
                        )
                        num_inference_steps = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=30,
                            step=1,
                            label="Inference Steps",
                        )
                    with gr.Row():
                        seed = gr.Number(
                            value=42,
                            label="Seed",
                            precision=0,
                        )
                        random_seed = gr.Button("🎲 Random", size="sm")
                    pipeline_config = gr.Dropdown(
                        choices=["omniweaving", "omniweaving2"],
                        value="omniweaving",
                        label="Guidance Preset",
                        info="omniweaving: guidance_scale=6.0, flow_shift=7.0 | omniweaving2: guidance_scale=6.0, flow_shift=5.0",
                    )
                    output_fps = gr.Number(
                        value=None,
                        label="Output FPS (auto if empty)",
                        info="Default: 16 for ≤81 frames, 24 for >81 frames",
                    )

                with gr.Accordion("🚀 Performance", open=True):
                    attention_mode = gr.Radio(
                        choices=["Torch SDPA", "SageAttention", "Sparse Attention"],
                        value="SageAttention",
                        label="Attention Mode",
                        info="SageAttention is recommended for speed on supported GPUs",
                    )
                    dtype = gr.Radio(
                        choices=["bf16", "fp32"],
                        value="bf16",
                        label="Data Type",
                        info="bf16: faster, lower memory | fp32: better quality, slower",
                    )
                    offloading = gr.Checkbox(
                        label="Enable Offloading",
                        value=True,
                        info="Move models between CPU and GPU to save VRAM",
                    )
                    group_offloading = gr.Checkbox(
                        label="Enable Group Offloading",
                        value=False,
                        info="Advanced offloading strategy (auto-enabled for low VRAM)",
                    )
                    overlap_group_offloading = gr.Checkbox(
                        label="Overlap Group Offloading",
                        value=True,
                        info="Speeds up inference but increases CPU memory usage",
                    )
                    enable_torch_compile = gr.Checkbox(
                        label="Enable Torch Compile",
                        value=True,
                        info="Compile transformer for faster inference (first step will be slower)",
                    )

                with gr.Accordion(
                    "💾 Cache / Acceleration", open=False
                ) as cache_section:
                    enable_cache = gr.Checkbox(
                        label="Enable Cache",
                        value=False,
                        info="Skip computation for certain steps to speed up inference",
                    )
                    cache_type = gr.Dropdown(
                        choices=CACHE_TYPES,
                        value="deepcache",
                        label="Cache Type",
                    )
                    with gr.Row():
                        cache_start_step = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=11,
                            step=1,
                            label="Cache Start Step",
                        )
                        cache_end_step = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=45,
                            step=1,
                            label="Cache End Step",
                        )
                    cache_step_interval = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=4,
                        step=1,
                        label="Cache Step Interval",
                    )
                    no_cache_block_id = gr.Textbox(
                        value="53",
                        label="No-Cache Block IDs",
                        info="Blocks to exclude from caching (e.g., 0-5 or 0,1,2,3,4,5)",
                    )

                with gr.Accordion("🔧 Advanced", open=False):
                    model_path = gr.Textbox(
                        value=DEFAULT_MODEL_PATH,
                        label="Model Path",
                        info="Path to the OmniWeaving checkpoint directory",
                    )
                    checkpoint_path = gr.Textbox(
                        value="",
                        label="Checkpoint Path (optional)",
                        info="Path to a custom transformer checkpoint directory",
                    )
                    lora_path = gr.Textbox(
                        value="",
                        label="LoRA Path (optional)",
                        info="Path to a LoRA adapter directory",
                    )
                    deepstack_str = gr.Textbox(
                        value="8, 16, 24",
                        label="Deepstack Layers",
                        info="Layer indices for multi-level semantic guidance (e.g., 0-5 or 8,16,24)",
                    )
                    with gr.Row():
                        setclip = gr.Checkbox(
                            label="SetCLIP",
                            value=True,
                            info="Enable CLIP-based features in text encoder",
                        )
                        only_give_text = gr.Checkbox(
                            label="Text-Only Conditioning",
                            value=False,
                            info="Ignore visual inputs in text encoder (text-only)",
                        )
                    use_fp8_gemm = gr.Checkbox(
                        label="Use FP8 GEMM",
                        value=False,
                        info="Enable FP8 matrix multiplication for faster inference",
                    )
                    quant_type = gr.Dropdown(
                        choices=QUANT_TYPES,
                        value="fp8-per-token-sgl",
                        label="Quantization Type",
                        visible=False,
                    )
                    include_patterns = gr.Dropdown(
                        choices=INCLUDE_PATTERNS,
                        value="double_blocks",
                        label="FP8 Include Patterns",
                        visible=False,
                    )
                    save_generation_config = gr.Checkbox(
                        label="Save Generation Config",
                        value=True,
                        info="Save a JSON config file alongside the output video",
                    )
                    quantize_4bit = gr.Checkbox(
                        label="4-bit Quantization (NF4)",
                        value=False,
                        info="Reduce VRAM usage by ~4x using bitsandbytes NF4 quantization. May slightly reduce quality.",
                    )

            with gr.Column(scale=1, min_width=500):
                gr.Markdown("### 🎬 Output")
                output_video = gr.Video(
                    label="Generated Video",
                    autoplay=True,
                )
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown("### 📊 Generation Log")
                log_output = gr.Textbox(
                    label="Live Log",
                    lines=20,
                    max_lines=50,
                    interactive=False,
                )
                status_indicator = gr.State(value=False)

        def on_task_change(task_val):
            desc = TASK_DESCRIPTIONS.get(task_val, "")
            return desc

        def on_fp8_toggle(val):
            return gr.update(visible=val), gr.update(visible=val)

        task.change(
            fn=on_task_change,
            inputs=[task],
            outputs=[task_desc],
        ).then(
            fn=update_media_inputs,
            inputs=[task],
            outputs=[ref_images, condition_video, think],
        )

        enable_cache.change(
            fn=update_cache_section,
            inputs=[enable_cache],
            outputs=[cache_section],
        )

        use_fp8_gemm.change(
            fn=on_fp8_toggle,
            inputs=[use_fp8_gemm],
            outputs=[quant_type, include_patterns],
        )

        def randomize_seed():
            import random

            return random.randint(0, 2**32 - 1)

        random_seed.click(fn=randomize_seed, outputs=[seed])

        generate_btn.click(
            fn=generate_video,
            inputs=[
                task,
                prompt,
                negative_prompt,
                think,
                ref_images,
                condition_video,
                aspect_ratio,
                video_length,
                num_inference_steps,
                seed,
                pipeline_config,
                output_fps,
                attention_mode,
                dtype,
                offloading,
                group_offloading,
                overlap_group_offloading,
                enable_torch_compile,
                enable_cache,
                cache_type,
                no_cache_block_id,
                cache_start_step,
                cache_end_step,
                cache_step_interval,
                model_path,
                checkpoint_path,
                lora_path,
                deepstack_str,
                setclip,
                only_give_text,
                use_fp8_gemm,
                quant_type,
                include_patterns,
                save_generation_config,
                quantize_4bit,
            ],
            outputs=[log_output, output_video, status_indicator],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.queue()
    app.launch(
        server_name="127.0.0.1",
        server_port=7777,
        share=False,
        show_error=True,
    )
