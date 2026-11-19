# Architecture Decision Records (ADRs): Gerador de Vídeos em Seamless Loop

Este documento estabelece as decisões arquiteturais para a versão baseada em **LangGraph** do pipeline de geração automatizada de vídeos curtos em *Seamless Loop* (loop infinito e fluido) para o YouTube a partir de um prompt em linguagem natural.

---

## ADR 001: Arquitetura Baseada em LangGraph e Multi-Agentes

- **Status:** Aprovado
- **Data:** 18 de Novembro de 2026

### Contexto e Problema
A criação de um vídeo em *seamless loop* para o YouTube exige um nível de refinamento superior ao de uma simples imagem estática ou prompt direto. O sistema precisa orquestrar etapas complexas: expandir o prompt criativo garantindo consistência visual de bordas, gerar os quadros-chave ou o vídeo base via modelos generativos de vídeo, validar tecnicamente a fluidez do loop (garantindo que o primeiro e o último frame se conectem perfeitamente) e formatar o conteúdo para o padrão do YouTube. Abordagens lineares simples falham ao tentar corrigir falhas de descontinuidade visual sem um fluxo de decisão retroativo.

### Decisão
Adotar uma arquitetura de orquestração baseada em **LangGraph**, definindo um grafo de estados direcionado onde agentes especializados colaboram, validam e refinam o conteúdo iterativamente antes da renderização final do arquivo de vídeo.

---

## ADR 002: Fluxo de Processamento em Múltiplas Etapas (Pipeline LangGraph)

- **Status:** Aprovado
- **Data:** 18 de Novembro de 2026

### Contexto e Problema
O sistema precisa traduzir um texto abstrato do usuário em um vídeo fluido, otimizado para reprodução contínua e formatado especificamente para a plataforma do YouTube (resolução padrão horizontal, taxa de quadros e duração adequadas).

### Decisão
O grafo de estados executará um pipeline estruturado composto pelas seguintes fases principais:
1. **Nó de Planejamento e Engenharia de Prompt:** O LLM analisa o prompt do usuário (ex.: estilo *Seamless Loop*) e expande os comandos para focar em texturas contínuas, iluminação uniforme e ausência de elementos direcionais bruscos nas bordas laterais.
2. **Nó de Geração de Vídeo / Quadros:** Aciona o modelo de geração de vídeo correspondente para produzir a mídia base com foco na transição cíclica (seamless).
3. **Nó de Validação de Loop e Pós-Processamento:** Um agente procedural ou rotina computacional verifica a paridade e a suavidade da junção entre o início e o fim do vídeo, realizando ajustes de interpolação se necessário.
4. **Nó de Empacotamento para o YouTube:** Realiza a codificação final no formato e resolução adequados para vídeos padrão do YouTube (ex: 16:9, codec H.264/MP4).

---

## ADR 003: Geração Automatizada de Saídas Vinculadas (`output/`)

- **Status:** Aprovado
- **Data:** 18 de Novembro de 2026

### Contexto e Problema
Os arquivos gerados (o vídeo em loop e o relatório técnico da execução) precisam ser salvos de forma organizada e rastreável na máquina do usuário para fácil publicação e auditoria do prompt original.

### Decisão
A cada execução bem-sucedida do fluxo do LangGraph, o sistema criará automaticamente o diretório `output/` (caso não exista) e gerará um par de arquivos vinculados contendo um prefixo com timestamp unificado e um slug descritivo extraído do prompt (ex: `YYYYMMDD_HHMMSS_seamless-loop-florestal.mp4` e `YYYYMMDD_HHMMSS_seamless-loop-florestal.md`). O arquivo `.md` documentará o grafo executado, os parâmetros de prompt refinados, as métricas de validação do loop e o caminho para o arquivo de vídeo gerado.