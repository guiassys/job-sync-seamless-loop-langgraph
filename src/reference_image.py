import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


MAX_REFERENCE_IMAGE_SIZE = 1024
REFERENCE_IMAGE_JPEG_QUALITY = 90


def encode_reference_image(image_path: str | Path) -> str:
    """Converte a imagem de referência em JPEG base64 para a API multimodal do Ollama."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Imagem de referência não encontrada: {path}")

    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail(
                (MAX_REFERENCE_IMAGE_SIZE, MAX_REFERENCE_IMAGE_SIZE),
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=REFERENCE_IMAGE_JPEG_QUALITY,
                optimize=True,
            )
    except Exception as error:
        raise ValueError(f"Não foi possível ler a imagem de referência {path}: {error}") from error

    return base64.b64encode(buffer.getvalue()).decode("ascii")
