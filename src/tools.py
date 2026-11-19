# src/tools.py

import os
import re
import gc
import tempfile
from datetime import datetime
import torch
import numpy as np
import cv2
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from langchain_core.tools import tool
from PIL import Image, ImageOps


VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 576
VIDEO_FPS = 7.0


def _slugify(text: str) -> str:
    """Converte um texto livre em um slug limpo para nomes de arquivos."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:40].strip("-") or "seamless-loop"


def _extract_slug(visual_specification: str, fallback_prompt: str) -> str:
    """Tenta extrair o slug sugerido na ficha técnica; caso falhe, gera a partir do prompt."""
    match = re.search(r"slug[:\s*]+([a-z0-9-]+)", visual_specification.lower())
    if match:
        return match.group(1)[:40].strip("-")
    return _slugify(fallback_prompt)


def _clear_vram():
    """Força coleta de lixo e limpeza de cache CUDA para evitar OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _generate_initial_image(
    conditioning_prompt: str, device: str
) -> Image.Image:
    """Gera uma imagem-base visível para condicionar o pipeline image-to-video."""
    model_id = os.getenv(
        "IMAGE_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = None

    try:
        print(f"🖼️ [Imagem-base] Carregando '{model_id}' no dispositivo: {device.upper()}...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        )
        if device == "cuda" and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)

        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        prompt = (
            f"{conditioning_prompt.strip()}. Wide 16:9 composition, "
            "locked-off tripod camera, fixed viewpoint, static framing, "
            "clear detailed subject, balanced exposure, natural colors, "
            "well-lit scene, no text or watermark."
        )
        generator = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                negative_prompt=(
                    "black screen, blank image, empty frame, underexposed, "
                    "unintelligible, text, watermark, camera movement, "
                    "pan, tilt, zoom, dolly, camera shake"
                ),
                width=768,
                height=432,
                num_inference_steps=25,
                guidance_scale=7.0,
                generator=generator,
            ).images[0]

        image = ImageOps.fit(
            image,
            (VIDEO_WIDTH, VIDEO_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if float(pixels.mean()) < 35 and float(pixels.std()) < 8:
            raise RuntimeError(
                "O modelo de imagem retornou uma imagem quase preta ou sem detalhes."
            )
        return image.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Falha ao gerar a imagem-base: {e}") from e
    finally:
        if pipeline is not None:
            del pipeline
        _clear_vram()


def _stabilize_camera(frames: list[Image.Image]) -> list[Image.Image]:
    """Compensa pan, tilt, zoom e rotação globais entre frames consecutivos."""
    if not frames:
        raise ValueError("O modelo de vídeo não retornou nenhum frame.")

    stabilized = [frames[0].convert("RGB")]
    previous_gray = cv2.cvtColor(
        np.asarray(stabilized[0]), cv2.COLOR_RGB2GRAY
    )

    for frame in frames[1:]:
        current = np.asarray(frame.convert("RGB"))
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        transform = None
        if points is not None and len(points) >= 8:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if tracked is not None and status is not None:
                valid = status.reshape(-1) == 1
                previous_points = points.reshape(-1, 2)[valid]
                current_points = tracked.reshape(-1, 2)[valid]
                if len(previous_points) >= 8:
                    transform, inliers = cv2.estimateAffinePartial2D(
                        previous_points,
                        current_points,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.99,
                    )
                    if transform is not None and (
                        inliers is None or int(inliers.sum()) < 8
                    ):
                        transform = None

        if transform is not None:
            inverse_transform = cv2.invertAffineTransform(transform)
            height, width = current.shape[:2]
            current = cv2.warpAffine(
                current,
                inverse_transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        stabilized_frame = Image.fromarray(current, mode="RGB")
        stabilized.append(stabilized_frame)
        previous_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)

    return stabilized


def _build_ping_pong_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Concatena o vídeo ao contrário sem duplicar os quadros dos extremos."""
    if len(frames) < 2:
        raise ValueError("São necessários pelo menos dois frames para criar o loop ping-pong.")
    return list(frames) + list(frames[-2:0:-1])


def inspect_video(filepath: str) -> dict[str, float | int]:
    """Decodifica o MP4 e retorna métricas reais, rejeitando saídas quase pretas."""
    capture = cv2.VideoCapture(filepath)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível abrir o vídeo exportado: {filepath}")

    frame_count = 0
    total_brightness = 0.0
    total_contrast = 0.0
    first_frame = None
    last_frame = None
    width = 0
    height = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_height, frame_width = frame.shape[:2]
            if frame_count == 0:
                width, height = frame_width, frame_height
                first_frame = frame
            elif (frame_width, frame_height) != (width, height):
                raise RuntimeError("O vídeo exportado contém frames com dimensões diferentes.")

            last_frame = frame
            total_brightness += float(frame.mean())
            total_contrast += float(frame.std())
            frame_count += 1
    finally:
        capture.release()

    if frame_count == 0 or first_frame is None or last_frame is None:
        raise RuntimeError(f"O vídeo exportado não contém frames decodificáveis: {filepath}")

    mean_brightness = total_brightness / frame_count
    mean_contrast = total_contrast / frame_count
    if mean_brightness < 35 and mean_contrast < 8:
        raise RuntimeError(
            "A validação detectou um vídeo quase preto ou sem conteúdo visual "
            f"(brilho médio {mean_brightness:.1f}, contraste médio {mean_contrast:.1f})."
        )

    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "mean_brightness": mean_brightness,
        "mean_contrast": mean_contrast,
        "boundary_mae": float(
            np.abs(first_frame.astype(np.float32) - last_frame.astype(np.float32)).mean()
        ),
    }


@tool("generate_video_tool")
def generate_video_tool(
    visual_specification: str, user_prompt: str, conditioning_prompt: str
) -> str:
    """Gera uma imagem-base textual, anima-a com SVD e exporta um MP4 validado."""
    print("🎬 [SVD] Iniciando renderização de vídeo em seamless loop...")

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _extract_slug(visual_specification, user_prompt)
    mp4_filename = f"{timestamp}_{slug}.mp4"
    mp4_filepath = os.path.join(output_dir, mp4_filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.getenv("VIDEO_MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt")
    temporary_filepath = None

    _clear_vram()

    pipeline = None

    try:
        image_prompt = conditioning_prompt.strip()
        if not image_prompt:
            raise ValueError("O engenheiro de vídeo não produziu um prompt para a imagem-base.")

        pil_image = _generate_initial_image(image_prompt, device)

        print(f"⚙️ [Hardware] Carregando '{model_id}' no dispositivo: {device.upper()}...")
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16" if device == "cuda" else None
        )
        pipeline.to(device)

        # Habilita otimizações de memória se disponíveis
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        print("🎞️ [Render] Gerando quadros com câmera fixa...")
        with torch.no_grad():
            frames = pipeline(
                pil_image,
                decode_chunk_size=8,
                motion_bucket_id=1,
                noise_aug_strength=0.0,
                generator=torch.Generator(device=device).manual_seed(42),
            ).frames[0]

        print("🧭 [Estabilização] Removendo movimento global da câmera...")
        stabilized_frames = _stabilize_camera(frames)
        print("🔁 [Loop] Concatenando o vídeo estabilizado com a sequência reversa...")
        output_frames = _build_ping_pong_frames(stabilized_frames)
        width, height = output_frames[0].size

        print(f"💾 [Exportação] Salvando vídeo em {mp4_filepath}...")
        temporary_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", dir=output_dir, delete=False
        )
        temporary_filepath = temporary_file.name
        temporary_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temporary_filepath, fourcc, VIDEO_FPS, (width, height))
        if not out.isOpened():
            out.release()
            raise RuntimeError("O OpenCV não conseguiu inicializar o encoder MP4.")

        try:
            for frame in output_frames:
                frame_np = np.asarray(frame.convert("RGB"))
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_np)
        finally:
            out.release()

        metrics = inspect_video(temporary_filepath)
        if metrics["frame_count"] != len(output_frames):
            raise RuntimeError(
                "A quantidade de frames decodificados difere da quantidade exportada "
                f"({metrics['frame_count']} != {len(output_frames)})."
            )
        os.replace(temporary_filepath, mp4_filepath)
        temporary_filepath = None

        print(f"✅ [Concluído] Vídeo em seamless loop salvo em: {mp4_filepath}")
        return mp4_filepath

    except Exception as e:
        raise RuntimeError(f"Falha na geração do vídeo via Stable Video Diffusion: {str(e)}") from e

    finally:
        if pipeline is not None:
            del pipeline
        _clear_vram()
        if temporary_filepath and os.path.exists(temporary_filepath):
            os.remove(temporary_filepath)