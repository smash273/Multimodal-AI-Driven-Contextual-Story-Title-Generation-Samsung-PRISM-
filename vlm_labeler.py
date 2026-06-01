"""
PRISM - VLM-based ground truth title generation using Qwen2-VL.

Uses Qwen2-VL to watch each video and generate a descriptive title.
These titles serve as ground truth for training the multimodal transformer.
"""
import os
import gc
import re
import torch
from pathlib import Path


class VLMLabeler:
    """
    Generates video titles using Qwen2-VL (video + optional OCR/ASR text).

    Aligns with PRISM’s multimodal training setup (same signals as the
    transformer and GPT pseudo-label path): pixels from the file plus, when
    present, `ocr_text/{id}.txt` and `audio_text/{id}.txt` under PRISM_DATA_DIR.
    """

    _QWEN_VIDEO_MIN_PIXELS = 128 * 28 * 28   # 100_352
    _QWEN_IMAGE_FACTOR = 28

    def __init__(
        self,
        model_name: str = None,
        max_pixels: int = None,
        video_fps: float = None,
        max_new_tokens: int = None,
        temperature: float = None,
        prompt: str = None,
        device: str = None,
        quantization: str = "none",
        use_aux_text: bool = None,
        max_ocr_chars: int = None,
        max_asr_chars: int = None,
    ):
        from config import (
            VLM_MODEL_NAME, VLM_MAX_PIXELS, VLM_VIDEO_FPS,
            VLM_MAX_NEW_TOKENS, VLM_TEMPERATURE, VLM_PROMPT,
            VLM_USE_AUX_TEXT, VLM_MAX_OCR_CHARS, VLM_MAX_ASR_CHARS,
        )

        self.model_name = model_name or VLM_MODEL_NAME
        self.max_pixels = max_pixels or VLM_MAX_PIXELS
        self.video_fps = video_fps or VLM_VIDEO_FPS
        self.max_new_tokens = max_new_tokens or VLM_MAX_NEW_TOKENS
        self.temperature = temperature or VLM_TEMPERATURE
        self.prompt = prompt or VLM_PROMPT
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = (quantization or "none").lower()
        self.use_aux_text = VLM_USE_AUX_TEXT if use_aux_text is None else bool(use_aux_text)
        self.max_ocr_chars = max_ocr_chars if max_ocr_chars is not None else VLM_MAX_OCR_CHARS
        self.max_asr_chars = max_asr_chars if max_asr_chars is not None else VLM_MAX_ASR_CHARS

        self.model = None
        self.processor = None

    def _resolve_model_dtype(self):
        """CUDA dtype: bf16 > fp16 for stability; fp32 if PRISM_VLM_FP32=1."""
        if os.environ.get("PRISM_VLM_FP32", "").strip().lower() in ("1", "true", "yes"):
            return torch.float32
        if self.device != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _resolve_processor_pixels(self):
        """Resolve safe min/max pixel bounds for AutoProcessor."""
        raw = int(self.max_pixels)
        if raw < self._QWEN_IMAGE_FACTOR * self._QWEN_IMAGE_FACTOR:
            max_px = raw * self._QWEN_IMAGE_FACTOR * self._QWEN_IMAGE_FACTOR
        else:
            max_px = raw
        max_px = max(max_px, self._QWEN_VIDEO_MIN_PIXELS)
        min_px = min(self._QWEN_VIDEO_MIN_PIXELS, max_px)
        return min_px, max_px

    def _is_small_model(self) -> bool:
        """Heuristic: model name contains '2B' or param count <= 3B."""
        name = self.model_name.lower()
        return "2b" in name or "1b" in name or "0.5b" in name

    def load_model(self):
        """Load Qwen2-VL model and processor."""
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        # Auto-disable quantization for small models — they fit in fp16 and
        # quantization destroys their logits.
        if self.quantization != "none" and self._is_small_model():
            print(f"   ⚠️  Ignoring --quantization {self.quantization} for small model "
                  f"{self.model_name} (fits in fp16, quantization breaks output).")
            self.quantization = "none"

        dtype = self._resolve_model_dtype()
        print(f"🧠 Loading VLM: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   torch_dtype: {dtype} (set PRISM_VLM_FP32=1 for float32 if output is garbage)")
        print(f"   Quantization: {self.quantization}")
        # Eager attention is slower but avoids occasional NaN logits from
        # flash-attn + long video sequences, which break torch.multinomial in sampling.
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "attn_implementation": "eager",
        }

        if self.quantization in {"4bit", "8bit"} and self.device == "cuda":
            if self.quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            load_kwargs["attn_implementation"] = "eager"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs,
        )
        self.model.eval()

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        min_px, max_px = self._resolve_processor_pixels()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_px,
            max_pixels=max_px,
            use_fast=False,
        )
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
            if getattr(tok, "pad_token_id", None) is None and getattr(
                tok, "eos_token_id", None
            ) is not None:
                tok.pad_token_id = tok.eos_token_id

        aux = "on (OCR/ASR appended when `ocr_text/` & `audio_text/` .txt exist)" if self.use_aux_text else "off"
        print(f"   Multimodal hints: {aux}")
        print("   ✅ Model loaded successfully")
        return self

    def _compose_user_text(self, video_id: str) -> str:
        """Base prompt plus OCR/ASR from PRISM_DATA_DIR (same layout as training)."""
        if not self.use_aux_text:
            return self.prompt
        from data.data_utils import load_ocr, load_asr

        ocr = load_ocr(video_id).strip()
        asr = load_asr(video_id).strip()
        if self.max_ocr_chars > 0:
            ocr = ocr[: self.max_ocr_chars]
        if self.max_asr_chars > 0:
            asr = asr[: self.max_asr_chars]
        if not ocr and not asr:
            return self.prompt
        parts = [
            self.prompt,
            "",
            "Automated hints from the same video (OCR/ASR; may be partial or wrong):",
            "",
        ]
        if ocr:
            parts.append(f"On-screen text (OCR):\n{ocr}")
        if asr:
            if ocr:
                parts.append("")
            parts.append(f"Spoken / transcribed audio (ASR):\n{asr}")
        return "\n".join(parts)

    def _safe_pixels(self, requested_max: int):
        mp = max(int(requested_max), self._QWEN_VIDEO_MIN_PIXELS)
        return self._QWEN_VIDEO_MIN_PIXELS, mp

    def _build_messages(
        self, video_path: str, max_pixels: int, video_fps: float, video_id: str
    ):
        min_px, max_px = self._safe_pixels(max_pixels)
        user_text = self._compose_user_text(video_id)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "min_pixels": min_px,
                        "max_pixels": max_px,
                        "fps": video_fps,
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ]

    def _run_generate(
        self, video_path: str, max_pixels: int, video_fps: float, video_id: str
    ) -> str:
        """Tokenize -> generate -> decode (no OOM handling)."""
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(video_path, max_pixels, video_fps, video_id)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        tok = getattr(self.processor, "tokenizer", None)
        pad_id = getattr(tok, "pad_token_id", None) if tok else None
        eos_id = getattr(tok, "eos_token_id", None) if tok else None
        cfg = self.model.config
        if pad_id is None:
            pad_id = getattr(cfg, "pad_token_id", None)
        if eos_id is None:
            eos_id = getattr(cfg, "eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id

        # Greedy only (no sampling — avoids CUDA multinomial asserts).
        # Do NOT use repetition_penalty / no_repeat_ngram here: with greedy they
        # push the model into rare tokens (!"#$%...) and produce gibberish.
        gen_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "min_new_tokens": 5,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        title = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0].strip()

        del inputs, output_ids, generated_ids
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        return title

    def generate_title(self, video_path: str, video_id: str = None) -> str:
        """Generate a title for a single video file."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        vid = video_id or Path(video_path).stem

        base_mp = int(self.max_pixels)
        base_fps = float(self.video_fps)
        floor = self._QWEN_VIDEO_MIN_PIXELS

        # Escalating fallbacks: reduce resolution / FPS for OOM recovery.
        attempts = [
            (base_mp, base_fps),
            (max(min(base_mp, 180 * 320), floor), max(min(base_fps, 0.35), 0.15)),
            (floor, max(min(base_fps, 0.25), 0.15)),
            (floor, 0.15),
        ]
        seen = set()
        unique = []
        for pair in attempts:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)

        last_err = None
        title = ""
        for mp, fps in unique:
            try:
                title = self._run_generate(video_path, mp, fps, vid)
                last_err = None
                if title and os.environ.get("PRISM_VLM_DEBUG", "").strip() in (
                    "1", "true", "yes",
                ):
                    print(f"     [debug] raw='{title}' (px={mp}, fps={fps})")
                if not self._is_degenerate_title(title):
                    break
            except RuntimeError as e:
                err = str(e).lower()
                last_err = e
                oom = "out of memory" in err
                cuda_err = "cuda" in err or "assert" in err or "probability" in err
                if oom or cuda_err:
                    gc.collect()
                    if self.device == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                raise

        if last_err is not None:
            raise last_err

        if self._is_degenerate_title(title) and os.environ.get(
            "PRISM_VLM_DEBUG", ""
        ).strip() in ("1", "true", "yes"):
            print(f"     [debug] all attempts degenerate, last raw='{title}'")
            title = ""
        elif self._is_degenerate_title(title):
            title = ""

        return self._clean_title(title) if title else "untitled video"

    @staticmethod
    def _is_degenerate_title(title: str) -> bool:
        if not title or not title.strip():
            return True
        t = title.strip()
        if re.fullmatch(r"[\W_]+", t):
            return True
        alpha = sum(ch.isalpha() for ch in t)
        if len(t) >= 12 and alpha < 3:
            return True
        if len(set(t)) <= 2 and len(t) >= 8:
            return True
        # Mostly punctuation / symbols (e.g. !!!"!!#!!$...)
        if len(t) >= 10:
            alnum = sum(1 for ch in t if ch.isalnum())
            if alnum < max(3, len(t) // 4):
                return True
        return False

    @staticmethod
    def _clean_title(title: str) -> str:
        title = title.strip().strip('"').strip("'").strip()
        if "\n" in title:
            title = title.split("\n")[0].strip()
        prefixes = [
            "Title:", "title:", "Video Title:", "video title:",
            "The title is:", "Here is the title:",
        ]
        for prefix in prefixes:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix):].strip()
        return title

    def generate_titles_batch(
        self,
        video_paths: list,
        video_ids: list = None,
        save_path: str = None,
        resume: bool = True,
    ) -> dict:
        """Generate titles for a batch of videos with progress saving."""
        import csv
        from tqdm import tqdm

        if video_ids is None:
            video_ids = [Path(p).stem for p in video_paths]

        results = {}

        if resume and save_path and os.path.exists(save_path):
            print(f"📂 Resuming from {save_path}")
            with open(save_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results[row["video_id"]] = row["title"]
            print(f"   Found {len(results)} previously generated titles")

        remaining = [
            (vid, path) for vid, path in zip(video_ids, video_paths)
            if vid not in results
        ]

        if not remaining:
            print("✅ All titles already generated!")
            return results

        print(f"\n🎬 Generating titles for {len(remaining)} videos...")

        for i, (vid, path) in enumerate(tqdm(remaining, desc="Generating titles")):
            try:
                title = self.generate_title(str(path), video_id=vid)
                results[vid] = title

                if (i + 1) % 5 == 0:
                    tqdm.write(f"  [{vid}] → {title}")

            except Exception as e:
                tqdm.write(f"  ⚠️  Error on {vid}: {e}")
                results[vid] = ""

            if save_path and (i + 1) % 10 == 0:
                self._save_csv(results, save_path)

            if self.device == "cuda" and (i + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if save_path:
            self._save_csv(results, save_path)

        return results

    @staticmethod
    def _save_csv(results: dict, path: str):
        import csv
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "title"])
            for vid, title in sorted(results.items()):
                writer.writerow([vid, title])

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ CUDA cache clear skipped: {e}")
        print("🗑️  VLM model unloaded")
