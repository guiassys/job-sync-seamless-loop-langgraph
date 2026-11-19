# src/nodes.py

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import VideoState
from src.reference_image import encode_reference_image
from src.tools import generate_video_tool, inspect_video

# Carrega variáveis de ambiente do arquivo .env (caso existam)
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:30b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE_PATH = Path(
    os.getenv(
        "REFERENCE_IMAGE_PATH",
        str(PROJECT_ROOT / "input" / "lofi-picture-style.png"),
    )
).expanduser()
if not REFERENCE_IMAGE_PATH.is_absolute():
    REFERENCE_IMAGE_PATH = PROJECT_ROOT / REFERENCE_IMAGE_PATH

# Carrega as configurações de prompts e tarefas do YAML
CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "prompts.yaml")
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# Instancia o modelo local via Ollama
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)
vision_llm = ChatOllama(
    model=OLLAMA_VISION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)


def _build_system_prompt(agent_key: str) -> str:
    """Monta a instrução de sistema (SystemMessage) a partir do YAML."""
    cfg = PROMPTS[agent_key]
    return (
        f"Papel: {cfg['role']}\n"
        f"Objetivo: {cfg['goal']}\n"
        f"Contexto e Especialidade: {cfg['backstory']}"
    )


def director_node(state: VideoState) -> dict:
    """Nó 1: Analisa a imagem de referência e cria a ficha técnica visual."""
    print("\n--- [1/4] EXECUTANDO: Diretor de Arte e Especialista em Conteúdo Seamless Loop ---")
    if not REFERENCE_IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Imagem de referência do diretor não encontrada: {REFERENCE_IMAGE_PATH}"
        )

    system_prompt = _build_system_prompt("seamless_loop_director_agent")
    user_prompt = PROMPTS["seamless_loop_director_agent"]["task_template"].format(
        user_prompt=state["user_prompt"],
        reference_image_path=REFERENCE_IMAGE_PATH,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image",
                    "base64": encode_reference_image(REFERENCE_IMAGE_PATH),
                    "mime_type": "image/jpeg",
                },
            ]
        ),
    ]

    response = vision_llm.invoke(messages)
    visual_specification_output = response.content

    return {"visual_specification": visual_specification_output}


def video_engineer_node(state: VideoState) -> dict:
    """Nó 2: Engenheiro de Vídeo (Gerencia recursos e executa a síntese do vídeo via Stable Video Diffusion)."""
    print("\n--- [2/4] EXECUTANDO: Engenheiro de Vídeo Sênior e Otimizador de Renderização ---")

    system_prompt = _build_system_prompt("video_engineer_agent")
    user_prompt = PROMPTS["video_engineer_agent"]["task_template"].format(
        visual_specification=state["visual_specification"]
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    conditioning_prompt = llm.invoke(messages).content
    if not isinstance(conditioning_prompt, str) or not conditioning_prompt.strip():
        raise RuntimeError("O engenheiro de vídeo não produziu um prompt visual para a imagem-base.")

    video_path = generate_video_tool.invoke({
        "visual_specification": state["visual_specification"],
        "user_prompt": state["user_prompt"],
        "conditioning_prompt": conditioning_prompt,
    })

    return {"video_filepath": video_path}


def validator_node(state: VideoState) -> dict:
    """Nó 3: Validador de Loop (Verifica a paridade e a suavidade da transição do vídeo)."""
    print("\n--- [3/4] EXECUTANDO: Validador de Seamless Loop ---")

    video_path = state.get("video_filepath", "")
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado para validação: {video_path}")

    video = inspect_video(video_path)
    metrics = (
        f"Vídeo decodificado e validado: {video['frame_count']} frames, "
        f"{video['width']}x{video['height']} a {video['fps']:.2f} fps. "
        f"Brilho médio: {video['mean_brightness']:.1f}/255; "
        f"contraste médio: {video['mean_contrast']:.1f}/255; "
        f"diferença média entre primeiro e último frame: "
        f"{video['boundary_mae']:.2f}/255 (menor indica maior semelhança)."
    )

    return {"loop_validation_metrics": metrics}


def editor_node(state: VideoState) -> dict:
    """Nó 4: Editor Técnico (Compila metadados e gera o relatório final em Markdown)."""
    print("\n--- [4/4] EXECUTANDO: Editor Técnico e Documentador de Pipeline ---")

    system_prompt = _build_system_prompt("editor_agent")
    user_prompt = PROMPTS["editor_agent"]["task_template"].format(
        user_prompt=state["user_prompt"],
        visual_specification=state["visual_specification"],
        loop_validation_metrics=state["loop_validation_metrics"],
        video_filepath=state["video_filepath"]
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return {"final_report": response.content}