# Avaliação de Arquiteturas de Pipeline de Dados para Classificação de Sentimentos em Diferentes Estratégias de Processamento

## Introdução
Este projeto investiga, em base experimental controlada, como diferentes arquiteturas de pipeline de dados afetam a classificação de sentimentos em tarefas de texto. O foco não é apenas a acurácia de modelos, mas a relação entre estratégia de processamento, custo computacional, reprodutibilidade e qualidade do experimento.

Em termos de pesquisa, o problema central é comparar duas formas de organizar o fluxo de dados para uma mesma tarefa de aprendizado de máquina:

- **pipeline batch**, no qual o conjunto de dados é processado em lote e a etapa de treino ocorre de forma consolidada;
- **pipeline microbatch**, no qual o fluxo é simulado por blocos sequenciais de dados, preservando uma noção temporal de chegada dos registros.

O benchmark proposto busca responder se a mudança de arquitetura altera o comportamento do sistema sob condições controladas e quais trade-offs surgem entre qualidade preditiva, tempo de processamento e reprodutibilidade.

## Objetivo
O objetivo geral é construir e avaliar um benchmark experimental controlado comparando duas arquiteturas para classificação de sentimentos:

1. **pipeline batch**;
2. **pipeline microbatch (streaming simulado)**.

O experimento pretende isolar o efeito da estratégia de processamento, mantendo constantes os demais elementos do pipeline sempre que possível, de modo que a comparação seja metodologicamente defensável em um contexto de iniciação científica.

## Hipóteses
As hipóteses iniciais são formuladas para orientar a análise experimental e não devem ser tratadas como conclusões antecipadas.

**H1.** Mantidas constantes as etapas de pré-processamento, vetorização, modelo e conjunto de dados, o pipeline batch tende a apresentar melhor eficiência global de processamento do que o pipeline microbatch, sobretudo em tempo total de execução.

**H2.** O pipeline microbatch tende a apresentar menor latência por bloco de dados, mas com maior sobrecarga operacional acumulada quando comparado ao processamento batch.

**H3 (opcional).** Se o corpus apresentar variação temporal relevante, a estratégia microbatch pode exibir comportamento mais sensível a mudanças de distribuição ao longo do tempo, o que pode afetar a estabilidade de métricas como F1-score.

## Benchmark Experimental
Este projeto deve ser interpretado como um benchmark experimental controlado, e não como mera implementação de software.

### Variável independente
- **Estratégia de processamento do pipeline**: batch vs microbatch.

### Variáveis dependentes
- **F1-score**;
- **precisão**;
- **recall**;
- **latência**;
- **throughput**;
- **tempo de processamento**.

### Variáveis controladas
Para que a comparação tenha rigor, recomenda-se controlar ao máximo os seguintes elementos:

- corpus utilizado;
- idioma do texto;
- regra de limpeza e normalização;
- estratégia de divisão treino/validação/teste;
- modelo de classificação;
- hiperparâmetros;
- versão das bibliotecas;
- seeds aleatórias;
- hardware e ambiente de execução;
- formato de serialização e armazenamento de artefatos.

### Protocolo experimental
1. Selecionar um corpus de sentimentos com rótulos bem definidos e, preferencialmente, informação temporal.
2. Aplicar uma etapa única e compartilhada de ingestão, limpeza e normalização textual.
3. Definir uma divisão temporal ou estratificada, dependendo da natureza do corpus.
4. Executar o baseline batch com os mesmos modelos e parâmetros definidos para o benchmark.
5. Executar o pipeline microbatch com blocos sequenciais reprodutíveis, simulando chegada de dados ao longo do tempo.
6. Registrar métricas de modelo e métricas operacionais para cada execução.
7. Repetir os experimentos com seeds fixas e configuração versionada.
8. Consolidar os resultados em tabelas comparativas e discussão crítica.

### Métricas
#### Métricas de modelo
- **F1-score**: principal métrica agregada para comparação entre classes;
- **precisão**: útil para observar falsos positivos;
- **recall**: útil para observar falsos negativos.

#### Métricas de pipeline
- **latência**: tempo entre entrada do lote e resposta do pipeline;
- **throughput**: volume processado por unidade de tempo, quando aplicável;
- **tempo de processamento**: custo total para concluir uma execução;
- **uso de memória**: opcional, se houver instrumentação confiável.

### Observação crítica sobre o benchmark
Um risco metodológico importante é comparar batch e microbatch sem separar claramente o que é efeito da arquitetura e o que é efeito do tamanho do lote, da divisão temporal ou da reexecução incremental do modelo. Para evitar esse problema, o protocolo precisa explicitar se a comparação será feita sob treino único, re-treino por janela ou avaliação incremental. Sem essa definição, a interpretação dos resultados fica ambígua.

## Arquitetura do Projeto
Estrutura proposta do repositório:

```text
src/
data/
experiments/
metrics/
docs/
notebooks/
configs/
```

### Papel de cada diretório
- `src/`: código-fonte principal do pipeline, incluindo ingestão, pré-processamento, treino, avaliação e microbatch;
- `data/`: dados brutos, intermediários e artefatos derivados;
- `experiments/`: scripts, manifests e resultados versionados dos experimentos;
- `metrics/`: tabelas consolidadas, logs e saídas comparativas;
- `docs/`: metodologia, decisões arquiteturais, figuras e texto científico;
- `notebooks/`: exploração, análise e geração de evidências reproduzíveis;
- `configs/`: parâmetros, seeds, perfis de execução e definições do benchmark.

### Princípio arquitetural
A lógica do experimento deve residir em módulos de código reutilizáveis. Notebooks devem servir para exploração e comunicação científica, não como local principal da implementação. Isso reduz variabilidade, melhora reprodutibilidade e facilita auditoria do pipeline.

## Stack Tecnológica
O projeto adota uma stack enxuta e adequada ao escopo de PIBIC:

- **Python**: linguagem principal do projeto;
- **Pandas**: manipulação de dados tabulares;
- **Scikit-learn**: modelos, vetorização e avaliação;
- **PySpark**: opcional, para representar processamento local em estilo streaming quando houver ganho metodológico;
- **Docker**: empacotamento do ambiente experimental;
- **Docker Compose**: orquestração mínima de serviços, quando útil.

## Baseline Inicial
A primeira implementação do projeto fixa o seguinte baseline:

- **dataset inicial**: IMDb reviews;
- **particionamento**: split estratificado treino/teste;
- **pipeline**: TF-IDF + Logistic Regression;
- **objetivo imediato**: estabelecer uma linha de base reprodutível para comparar com o futuro microbatch.

Essa escolha é deliberada: o IMDb fornece um problema binário clássico, reduz ambiguidade de rótulos e permite concentrar a análise na arquitetura do pipeline, não na complexidade do corpus.

## Anotações Operacionais
Esta base foi desenhada para a Fase 1 do PIBIC, com foco em benchmark controlado e reprodutível.

- O corpus inicial é o IMDb, obtido automaticamente na primeira execução.
- O notebook de laboratório está em [notebooks/01_phase1_baseline_batch.ipynb](notebooks/01_phase1_baseline_batch.ipynb).
- Os resultados consolidados do runner são gravados em `metrics/`.
- Os artefatos de modelo e manifesto de execução são gravados em `experiments/runs/`.
- A execução reprodutível usa `uv` da Astral, não um ambiente Python manual solto.
- PyTorch, Hadoop e DataBricks ficam fora da Fase 1 por enquanto.
- PostgreSQL não é obrigatório nesta etapa; se entrar depois, deve ser por justificativa clara de persistência/consulta experimental.

## Como Rodar
### Pré-requisitos
- `uv` instalado.
- Docker e Docker Compose instalados.

### 1. Sincronizar o ambiente
```bash
uv sync
```

Esse comando cria e/ou atualiza a virtualenv do projeto com as dependências do `pyproject.toml`.

### 2. Executar os testes mínimos
```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Isso valida a normalização, o baseline batch e o runner experimental.

### 3. Rodar o baseline batch individualmente
```bash
uv run run-batch-baseline
```

Opcionalmente, você pode ajustar parâmetros:

```bash
uv run run-batch-baseline --dataset imdb --test-size 0.2 --random-state 42 --output-dir metrics --artifacts-dir experiments/runs
```

### 4. Rodar o Sprint 1 completo do benchmark
```bash
uv run run-sprint1
```

Esse comando executa múltiplas seeds e os dois modelos da fase inicial: `logreg` e `linear_svm`.

### 5. Abrir o notebook de laboratório
Abra [notebooks/01_phase1_baseline_batch.ipynb](notebooks/01_phase1_baseline_batch.ipynb) no VS Code ou no Jupyter. O notebook importa os módulos do pacote e chama o baseline batch sem concentrar a lógica do experimento nele.

### 6. Executar via Docker
Build da imagem:

```bash
docker compose build
```

Executar o baseline no container:

```bash
docker compose run --rm batch-baseline
```

Se quiser substituir o comando padrão, você pode invocar diretamente o runner:

```bash
docker compose run --rm batch-baseline run-sprint1
```

### 7. Onde encontrar os artefatos
- Métricas agregadas: `metrics/batch_baseline_metrics.csv` e `metrics/benchmark_summary.csv`.
- Logs de execução: `metrics/batch_run_log.jsonl`.
- Manifestos por execução: `experiments/runs/*.json`.
- Modelos serializados: `experiments/runs/*.joblib`.

## Observações de Uso
- Na primeira execução, o IMDb pode ser baixado pela biblioteca `datasets`.
- Se você quiser comparar resultados entre execuções, mantenha as seeds fixas.
- O baseline batch foi desenhado para ser a referência científica do projeto; o microbatch virá depois como comparação arquitetural.

## Reprodutibilidade
O uso de Docker não é apenas uma conveniência operacional; ele faz parte da metodologia experimental.

### Por que Docker é importante aqui
- fixa a versão das dependências;
- reduz variações entre máquinas;
- facilita a replicação do benchmark por terceiros;
- documenta o ambiente como parte da evidência científica;
- permite reconstruir o contexto de execução dos experimentos.

### Diretriz de uso
O ambiente deve permanecer mínimo. A meta é preservar reprodutibilidade sem adicionar complexidade desnecessária. Dockerfile e, quando necessário, `docker-compose.yml` devem ser usados para padronizar a execução do pipeline, dos notebooks e dos artefatos experimentais.

### Critério de suficiência
Se o experimento puder ser executado de forma íntegra dentro de um container, com dados e dependências versionados, a reprodutibilidade está adequadamente endereçada para o escopo do projeto.

## Ameaças à Validade
Como em todo benchmark experimental, este projeto está sujeito a ameaças metodológicas que precisam ser explicitadas.

### Validade interna
Risco: diferenças observadas entre batch e microbatch podem ser causadas por fatores externos, como tamanho de lote, seed, ordem dos dados ou variação de hiperparâmetros.

Mitigação:
- manter etapas comuns entre os pipelines;
- fixar seeds e parâmetros;
- registrar versões de dependências;
- usar o mesmo corpus e a mesma estratégia de pré-processamento.

### Validade externa
Risco: os resultados podem não generalizar para outros corpora, outros idiomas ou fluxos de produção reais.

Mitigação:
- declarar claramente o domínio do corpus;
- evitar extrapolações para produção industrial;
- contextualizar o benchmark como estudo comparativo em ambiente controlado.

### Validade de construção
Risco: métricas de latência, throughput e tempo de processamento podem não capturar toda a complexidade arquitetural do pipeline.

Mitigação:
- definir formalmente como cada métrica será medida;
- distinguir métricas de modelo e de pipeline;
- documentar o ponto exato em que a medição ocorre;
- explicitar limitações de instrumentação.

## Roadmap do Projeto
O desenvolvimento será organizado em fases sequenciais.

### 1. Benchmark protocol
Definição do corpus, hipóteses, variáveis, métricas, seeds e protocolo experimental.

### 2. Baseline batch
Implementação do pipeline batch como referência técnica e científica.

### 3. Microbatch pipeline
Implementação da simulação de streaming por blocos sequenciais de dados.

### 4. Comparative evaluation
Execução dos experimentos, comparação das métricas e análise crítica dos trade-offs.

### 5. Relatório/artigo
Consolidação dos resultados em texto científico, figuras, tabelas e discussão metodológica.

## Estrutura futura de experimentos
Para sustentar reprodutibilidade e evolução do estudo, recomenda-se versionar cada execução experimental.

### Organização sugerida
- cada experimento deve ter um identificador único;
- parâmetros devem ser salvos em arquivo de configuração;
- resultados devem ser exportados em formato tabular;
- artefatos de modelos devem ficar associados ao hash da configuração;
- gráficos e tabelas devem ser gerados a partir dos resultados versionados, nunca manualmente.

### Boas práticas de versionamento
- registrar data, seed, commit e imagem Docker utilizada;
- separar resultados brutos de resultados agregados;
- evitar sobrescrever execuções anteriores;
- manter rastreabilidade entre código, dados e métricas.

## Fragilidades metodológicas esperadas
Este projeto tem potencial científico, mas ainda depende de decisões que precisam ser fechadas com rigor para evitar conclusões frágeis.

- Se o corpus não tiver timestamp confiável, a simulação de streaming ficará menos realista e mais dependente de uma ordenação artificial.
- Se o microbatch for comparado com batch sem controle do tamanho do bloco, a interpretação dos resultados pode ficar enviesada.
- Se a avaliação usar apenas um corpus, a generalização será limitada; isso não invalida o PIBIC, mas precisa ser assumido explicitamente.
- Se os modelos forem avaliados sem análise de custo computacional, o benchmark ficará incompleto do ponto de vista arquitetural.

## Próximos passos
1. Definir o corpus e a estratégia de divisão temporal.
2. Fechar o protocolo experimental com variáveis e hipóteses.
3. Implementar a base compartilhada de pré-processamento.
4. Criar o Dockerfile e o ambiente reprodutível mínimo.
5. Estruturar o baseline batch antes de avançar para o microbatch.
