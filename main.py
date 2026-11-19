# main.py

import os
from datetime import datetime
from dotenv import load_dotenv
from src.graph import create_workflow
from src.flow_strategy import OriginalVideoGenerationStrategy, FastVideoGenerationStrategy

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "output")),
)
DEFAULT_PROMPT = os.getenv(
    "USER_PROMPT",
    "Seamless Loop com gatinho dormindo e respirando profundamente deitado em uma cama de um quarto aconchegante, com composição perfeitamente contínua, na qual o primeiro e o último quadro se conectam sem qualquer interrupção visual, criando um ciclo infinito e fluido de respiração.",
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    user_prompt = DEFAULT_PROMPT
    if not user_prompt:
        user_prompt = input("Digite o prompt descritivo para o vídeo em seamless loop: ")

    # Ask which flow to use
    print("\nEscolha o fluxo de geração:")
    print("1. Original (Pipeline Completo com Validação de Loop)")
    print("2. Rápido (Geração Direta)")

    choice = input("Digite sua escolha (1 ou 2): ").strip()

    # Initialize strategies
    strategies = {
        "1": OriginalVideoGenerationStrategy(),
        "2": FastVideoGenerationStrategy()
    }

    selected_strategy = strategies.get(choice, OriginalVideoGenerationStrategy())

    print(f"\n## Iniciando Pipeline Job Sync Seamless Loop (LangGraph + Ollama + SVD) ##")
    print(f"Prompt: {user_prompt}")
    print(f"Fluxo selecionado: {selected_strategy.get_name()}\n")

    # Estado inicial injetado no grafo
    initial_state = {
        "user_prompt": user_prompt,
        "visual_specification": "",
        "loop_validation_metrics": "",
        "video_filepath": "",
        "final_report": "",
    }

    # Execute using selected strategy
    try:
        result = selected_strategy.execute(initial_state)

        final_content = result.get(
            "final_report", "Nenhum relatório foi gerado pela pipeline."
        )
        video_filepath = result.get("video_filepath", "")

        # Deriva o nome do arquivo markdown usando o mesmo timestamp/slug do vídeo gerado, se disponível
        if video_filepath and os.path.exists(video_filepath):
            base_name = os.path.splitext(os.path.basename(video_filepath))[0]
            markdown_filename = f"{base_name}.md"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            markdown_filename = f"{timestamp}_seamless_loop.md"

        filepath = os.path.join(OUTPUT_DIR, markdown_filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str(final_content))

        print("\n\n##################################################")
        print(f"## RELATÓRIO SALVO COM SUCESSO: {filepath} ##")
        if video_filepath:
            print(f"## VÍDEO GERADO: {video_filepath} ##")
        print("##################################################\n")
        print(final_content)

    except Exception as e:
        print(f"\nErro durante a execução do fluxo: {str(e)}")
        raise

if __name__ == "__main__":
    main()