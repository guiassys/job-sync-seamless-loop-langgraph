# 🤖 Job Sync Seamless Loop LangGraph (LangGraph + Ollama Local)

Pipeline local de geração de vídeos curtos a partir de prompts em linguagem natural. O fluxo usa **Ollama** para interpretar o pedido e analisar uma imagem de referência, e **Diffusers** para criar a imagem-base e animá-la com Stable Video Diffusion. A geração pode usar CUDA quando disponível.

---

## 🏗️ Como o Projeto Funciona (Arquitetura)

O projeto utiliza um pipeline sequencial controlado por um Grafo de Estados (*StateGraph*), composto por quatro nós especializados cujos papéis e diretrizes são desacoplados no arquivo de configuração `config/prompts.yaml`:

1. **Diretor de Arte (`seamless_loop_director_agent`):** analisa o pedido do usuário e a imagem `input/lofi-picture-style.png` com um modelo multimodal do Ollama. A ficha técnica descreve estilo, paleta, iluminação, textura, composição e atmosfera da referência, aplicando essa linguagem visual sem substituir o assunto solicitado. Também propõe um slug para os arquivos de saída.

2. **Engenheiro de Vídeo (`video_engineer_agent`):** transforma a ficha em um prompt visual em inglês, gera a imagem-base com Stable Diffusion (`IMAGE_MODEL_ID`) e a anima com Stable Video Diffusion (`VIDEO_MODEL_ID`). A configuração de baixa movimentação e a estabilização por fluxo óptico reduzem o movimento global de câmera, mas não garantem que todo movimento de câmera gerado pelo modelo seja removido. Os frames estabilizados são exportados na ordem normal e depois reversa (ping-pong), sem repetir os extremos. Saídas não decodificáveis ou quase pretas são rejeitadas.

3. **Validador (`validator_agent`):** no fluxo original, lê o MP4 e informa número de frames, resolução, FPS, brilho, contraste e diferença entre o primeiro e o último frame. No fluxo rápido, a exportação ainda verifica que o arquivo é decodificável e contém conteúdo visual.

4. **Editor (`editor_agent`):** produz um relatório Markdown associado ao vídeo gerado, com o prompt, a ficha visual, as métricas disponíveis e o caminho do MP4.

---

## 🔄 Fluxos de Geração Disponíveis

O projeto suporta diferentes fluxos de geração através do padrão Strategy:

- **Original:** planejamento visual, geração, métricas detalhadas do MP4 e relatório.
- **Rápido:** planejamento visual, geração e relatório; omite as métricas detalhadas do nó validador.

Ao executar o projeto, você poderá selecionar qual fluxo deseja utilizar.

---

## 📂 Estrutura do Projeto

```text
job-sync-seamless-loop-langgraph/
├── config/
│   └── prompts.yaml      # Configuração unificada de papéis, goals, backstories e templates
├── input/
│   └── lofi-picture-style.png # Imagem de referência visual do agente diretor
├── output/               # Vídeos (.mp4) e relatórios (.md) com timestamp e slug
├── src/
│   ├── __init__.py
│   ├── state.py          # Estado compartilhado do fluxo
│   ├── reference_image.py # Codificação da imagem para entrada multimodal do Ollama
│   ├── tools.py          # Geração, estabilização, ping-pong, exportação e inspeção do vídeo
│   ├── nodes.py          # Nós do pipeline e chamadas ao Ollama
│   ├── graph.py          # Grafo LangGraph
│   └── flow_strategy.py  # Fluxos Original e Rápido
├── tests/                # Testes de imagem de referência e processamento de vídeo
├── .env                  # Configuração local (não versionar credenciais)
├── README.md             # Documentação do projeto
├── .gitignore            # Contrato de arquivos a serem ignorados pelo Git
├── main.py               # Ponto de entrada leve (executa o fluxo e persiste os arquivos)
└── requirements.txt      # Dependências do projeto
```

---

## ⚙️ Pré-requisitos

- **Python 3.10+** (Ambiente WSL / Ubuntu recomendado)
- **OpenCV / FFmpeg** instalados no sistema operacional para manipulação de vídeo (`sudo apt install ffmpeg libgl1-mesa-glx`)
- **Ollama** instalado e disponível em `http://localhost:11434` (ou no endereço configurado em `OLLAMA_BASE_URL`).
- Modelo textual no Ollama (padrão do projeto: `qwen3-coder:30b`) e modelo multimodal capaz de receber imagens (padrão: `qwen2.5vl:7b`).
- **GPU NVIDIA com CUDA** recomendada para geração prática com os modelos de difusão. A execução em CPU pode ser muito lenta e consumir bastante memória.

---

## 🚀 Instalação e Configuração

Abra o seu terminal no WSL/Ubuntu e siga os passos abaixo:

1. **Instalar dependências de sistema**
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
Baixe os modelos do Ollama se ainda não estiverem disponíveis:

```bash
ollama pull qwen2.5vl:7b
ollama pull qwen3-coder:30b
```

Com o ambiente virtual ativado (`source venv/bin/activate`), execute:

```bash
python main.py
```

O pipeline será iniciado, e você poderá escolher entre os fluxos disponíveis:
- 1. Original (Pipeline Completo com Validação de Loop)
- 2. Rápido (Geração Direta)

Na primeira execução, o Diffusers baixa os modelos de imagem e vídeo, que ocupam espaço em disco e podem demandar bastante VRAM. A imagem de referência precisa existir antes de executar o pipeline.

### Configuração por variáveis de ambiente

| Variável | Padrão | Uso |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endereço do servidor Ollama |
| `OLLAMA_MODEL` | `qwen3-coder:30b` | Modelo textual usado pelo engenheiro/editor |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Modelo multimodal usado pelo diretor |
| `REFERENCE_IMAGE_PATH` | `input/lofi-picture-style.png` | Imagem de referência visual |
| `IMAGE_MODEL_ID` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Modelo text-to-image |
| `VIDEO_MODEL_ID` | `stabilityai/stable-video-diffusion-img2vid-xt` | Modelo image-to-video |
| `OUTPUT_DIR` | `output/` | Diretório dos relatórios Markdown |

O caminho relativo configurado em `REFERENCE_IMAGE_PATH` é resolvido a partir da raiz do projeto. O vídeo é salvo no diretório `output/` do projeto. Os nomes de vídeo e relatório compartilham timestamp e slug, por exemplo `20261118_183000_seamless-loop.mp4` e `20261118_183000_seamless-loop.md`.

O MP4 ping-pong contém o clipe estabilizado seguido dos mesmos frames em ordem reversa. Isso cria uma ida e volta contínua, mas não equivale a uma animação cíclica unidirecional; eventos irreversíveis da cena também serão reproduzidos ao contrário.

Para executar os testes de processamento:

```bash
python -m unittest discover -s tests -v
```