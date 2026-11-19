# 🤖 Job Sync Seamless Loop LangGraph (LangGraph + Ollama Local)

Pipeline de Inteligência Artificial autônoma construída com **LangGraph** para criar vídeos curtos em *Seamless Loop* (loop infinito e fluido) a partir de prompts descritivos em linguagem natural, seguindo uma arquitetura modular baseada em grafos de estados. O sistema opera 100% local utilizando modelos executados via **Ollama** (`qwen3-coder:30b`) e o modelo de geração de vídeo HuggingFace / Diffusers (`stabilityai/stable-video-diffusion-img2vid-xt`) em ambiente acelerado por GPU.

---

## 🏗️ Como o Projeto Funciona (Arquitetura)

O projeto utiliza um pipeline sequencial controlado por um Grafo de Estados (*StateGraph*), composto por quatro nós especializados cujos papéis e diretrizes são desacoplados no arquivo de configuração `config/prompts.yaml`:

1. **Diretor de Arte e Especialista em Conteúdo Seamless Loop (`seamless_loop_director_agent`):**
- Analisa o prompt descritivo inserido pelo usuário (ex: *"Crie uma imagem em estilo Seamless Loop..."*).
- Traduz conceitos abstratos em parâmetros visuais precisos (direção de arte, consistência de bordas, paleta de cores e iluminação uniforme).
- Gera um slug descritivo e define as especificações de transição e ciclo para o vídeo.

2. **Engenheiro de Vídeo Sênior e Otimizador de Renderização (`video_engineer_agent`):**
- Gerencia a exclusividade de recursos de hardware para evitar estouros de VRAM (*CUDA Out of Memory*) entre o Ollama e a GPU.
- Converte a ficha técnica em um prompt visual e gera um frame inicial com Stable Diffusion (`IMAGE_MODEL_ID`).
- Usa esse frame como entrada do modelo image-to-video `stabilityai/stable-video-diffusion-img2vid-xt`, em vez de um frame sintético quase preto.
- Instrui o SVD a manter a câmera fixa e usa `motion_bucket_id=1` e `noise_aug_strength=0` para reduzir movimento global.
- Estabiliza deslocamentos residuais da câmera por fluxo óptico e concatena os frames em ordem normal e reversa (ping-pong), sem duplicar os quadros dos extremos.
- Exporta para `.mp4` e rejeita saídas quase pretas ou que não possam ser decodificadas.

3. **Validador de Seamless Loop (`validator_agent`):**
- Verifica a paridade e a suavidade da junção entre o início e o fim do vídeo.
- Lê o MP4 gerado e informa resolução, frames, brilho, contraste e diferença real entre o primeiro e o último frame; não declara sucesso com métricas simuladas.
- No loop ping-pong, a reprodução reversa retorna ao frame inicial e permite que a animação recomece sem corte de câmera.

4. **Editor Técnico e Documentador (`editor_agent`):**
- Recebe os metadados consolidados da execução, o prompt original, os parâmetros visuais, as métricas de validação e o caminho do arquivo de vídeo.
- Valida e compila a documentação estruturada em Markdown, salvando o relatório final vinculado ao arquivo de vídeo.

---

## 🔄 Fluxos de Geração Disponíveis

O projeto suporta diferentes fluxos de geração através do padrão Strategy:

- **Fluxo Original (Pipeline Completo com Validação de Loop)**: Pipeline completo incluindo a etapa de validação métrica da transição.
- **Fluxo Rápido (Geração Direta)**: Versão simplificada que ignora a validação intermediária para renderização mais rápida.

Ao executar o projeto, você poderá selecionar qual fluxo deseja utilizar.

---

## 📂 Estrutura do Projeto

```text
job-sync-seamless-loop-langgraph/
├── config/
│   └── prompts.yaml      # Configuração unificada de papéis, goals, backstories e templates
├── data/                 # Diretório opcional para dados complementares
├── output/               # Diretório onde os arquivos de vídeo (.mp4) e relatórios (.md) são salvos com timestamp e slug
├── src/
│   ├── __init__.py
│   ├── state.py          # Contrato de dados compartilhado (TypedDict) que trafega no grafo
│   ├── tools.py          # Ferramenta de síntese de vídeo (SVD) e gestão de VRAM da GPU
│   ├── nodes.py          # Funções de cada etapa executando via ChatOllama
│   ├── graph.py          # Montagem, interligação das arestas e compilação do StateGraph
│   └── flow_strategy.py  # Implementação do padrão Strategy para diferentes fluxos de geração
├── .env                  # Variáveis de ambiente (URL do Ollama, modelo de vídeo e diretório)
├── README.md             # Documentação do projeto
├── .gitignore            # Contrato de arquivos a serem ignorados pelo Git
├── main.py               # Ponto de entrada leve (executa o fluxo e persiste os arquivos)
└── requirements.txt      # Dependências do projeto
```

---

## ⚙️ Pré-requisitos

- **Python 3.10+** (Ambiente WSL / Ubuntu recomendado)
- **OpenCV / FFmpeg** instalados no sistema operacional para manipulação de vídeo (`sudo apt install ffmpeg libgl1-mesa-glx`)
- **Ollama** rodando localmente (ex: via Docker) com o modelo configurado (`qwen3-coder:30b` na porta `11434`).
- **Placa de Vídeo NVIDIA** com suporte CUDA para inferência dos modelos locais de linguagem e difusão de vídeo.

---

## 🚀 Instalação e Configuração

Abra o seu terminal no WSL/Ubuntu e siga os passos abaixo:

1. **Instalar dependências de sistema (FFmpeg e dependências gráficas)**
   ```bash
   sudo apt update && sudo apt install -y ffmpeg libgl1-mesa-glx
   ```

2. **Navegue até o diretório do projeto**
   ```bash
   cd ~/devtools/repos/job-sync-seamless-loop-langgraph
   ```


3. **Criar e ativar o ambiente virtual Python**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Instalar as dependências do projeto**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ▶️ Como Executar o Projeto

Certifique-se de que o seu container do Ollama está rodando e acessível na porta `11434`.

Com o ambiente virtual ativado (`source venv/bin/activate`), basta executar o script principal:

```bash
python main.py
```

O pipeline será iniciado, e você poderá escolher entre os fluxos disponíveis:
- 1. Original (Pipeline Completo com Validação de Loop)
- 2. Rápido (Geração Direta)

Na primeira execução, o Diffusers baixa o modelo text-to-image configurado em `IMAGE_MODEL_ID` (padrão: `stable-diffusion-v1-5/stable-diffusion-v1-5`), além do modelo de vídeo. É possível selecionar outros IDs de modelos compatíveis por `IMAGE_MODEL_ID` e `VIDEO_MODEL_ID` no ambiente. Os arquivos gerados serão salvos automaticamente no diretório `output/` utilizando um prefixo unificado com timestamp e slug de estilo (ex: `20261118_183000_seamless-loop.mp4` e `20261118_183000_seamless-loop.md`).