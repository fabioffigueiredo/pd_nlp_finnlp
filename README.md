<div align="center">
  <h1>
    <img src="images/logo_infnet.png" alt="Instituto Infnet" width="80" title="Instituto Infnet" align="absmiddle"/>
    Projeto de Disciplina: FinNLP
  </h1>
  <h3>Pipeline de NLP para Atribuição de Performance e Inteligência de Mercado</h3>
</div>

<div align="center">

  **Pós-Graduação em Machine Learning, Deep Learning e Inteligência Artificial**<br>
  **Disciplina:** Sistemas Cognitivos e Linguagem Natural<br>
  **Professor:** Fernando Guimarães Ferreira<br>
  **Aluno:** Fabio Ferreira Figueiredo <a href="https://github.com/fabioffigueiredo"><img src="https://img.shields.io/badge/GitHub-repo-black?logo=github" alt="GitHub"></a>

  <p>
    <img src="https://img.shields.io/badge/python-v._3.12-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/jupyter-notebook-orange?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
    <img src="https://img.shields.io/badge/spaCy-3.7-09a3d5?style=flat-square&logo=spacy&logoColor=white" alt="spaCy">
    <img src="https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/MLflow-2.12-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
  </p>
</div>

> **Projeto acadêmico.** O cliente ("Gestão do Fundo") é **fictício e genérico** —
> nenhuma instituição financeira real é referenciada. Os dados são públicos
> (`financial_phrasebank`, via Hugging Face) e/ou sintéticos; quaisquer trechos
> de fontes corporativas foram **anonimizados** antes do processamento.

---

## Visão Geral do Projeto

O **FinNLP** é um pipeline _end-to-end_ de Processamento de Linguagem Natural que
transforma notícias e relatórios financeiros brutos em inteligência acionável para
a **Diretoria de Estratégia de um fundo de investimentos**. O sistema classifica o
sentimento de mercado (Risco/Oportunidade), descobre temas latentes, implementa
busca semântica e constrói um grafo de conhecimento entre entidades do mercado,
com versionamento histórico via **SCD Tipo 2**.

A solução parte do texto bruto e percorre toda a progressão analítica:
pré-processamento linguístico → representação vetorial → modelagem e classificação
→ extração de entidades e grafo → comunicação executiva — sempre priorizando a
reprodutibilidade e a interpretabilidade das decisões técnicas.

---

## O Corpus

Estratégia **bilíngue** combinando duas fontes complementares:

- **[financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank)** (inglês):
  4.845 sentenças financeiras rotuladas por analistas de mercado
  (*negative / neutral / positive*) — benchmark de referência para sentimento financeiro.
- **Web scraping PT-BR** (Agência Brasil): demonstra a coleta autônoma com
  `requests` + `BeautifulSoup`.

A combinação garante o mínimo de **1.000 documentos**, **2+ classes** comparáveis e
presença de **entidades nomeadas** do domínio financeiro.

---

## O que o Notebook Faz

O arquivo [`notebooks/FinNLP_Pipeline.ipynb`](notebooks/FinNLP_Pipeline.ipynb) cobre
todo o ciclo do pipeline, organizado pelas **5 rubricas** da disciplina:

1. **Pré-processamento (Rubrica 1)** — NLTK + spaCy, tokenização, stopwords
   financeiras customizadas, **comparação lematização vs stemming**, POS tagging,
   nuvem de palavras e histograma de comprimento.
2. **Representação vetorial e busca (Rubrica 2)** — TF-IDF (uni + bigramas),
   Word2Vec, **motor de busca por similaridade de cosseno** (3 consultas de
   performance attribution) e visualização t-SNE.
3. **Modelagem e tópicos (Rubrica 3)** — divisão estratificada, **Naive Bayes vs
   SVM** (F1-Score, Precision, Recall, matriz de confusão), **LDA + pyLDAvis** e
   rastreamento de experimentos com **MLflow**.
4. **NER, grafo e engenharia de dados (Rubrica 4)** — extração de organizações
   com spaCy, padrões via **RegEx**, normalização por **distância de Levenshtein**,
   grafo de conhecimento **NetworkX (≥ 20 nós)** com centralidade de grau e
   **SCD Tipo 2** (SQLAlchemy/SQLite) para versionar o histórico de sentimento.
5. **Comunicação (Rubrica 5)** — síntese executiva em linguagem não técnica,
   reprodutibilidade (`random_state=42`) e discussão de limitações.

### Artefatos de Entrega

| Artefato | Caminho |
|---|---|
| **Notebook executado** (Run All) | `notebooks/FinNLP_Pipeline.ipynb` |
| **Relatório técnico em PDF** | `reports/fabio_figueiredo_sistemas-cognitivos-linguagem-natural_pln.pdf` |
| **Visualizações** | `reports/images/*.png` |
| **Grafo interativo** | `reports/images/grafo_interativo_r4.html` |
| **Grafo exportado (Gephi)** | `data/processed/grafo_conhecimento.gexf` |
| **Banco SCD2 (SQLite)** | `data/db/finnlp.sqlite` |

---

## Estrutura do Repositório

```
pd-nlp-finnlp/
├── notebooks/
│   └── FinNLP_Pipeline.ipynb     # Notebook principal (executado end-to-end)
├── src/                          # Módulos do pipeline (importados pelo notebook)
│   ├── coleta_preprocessamento.py   # Fases 0 e 1
│   ├── modelagem_vetorizacao.py     # Fases 2 e 3
│   ├── ner_grafo.py                 # Fase 4 (NER + RegEx + Levenshtein + grafo)
│   └── scd2_manager.py              # Fase 4 (engenharia de dados — SCD Tipo 2)
├── reports/
│   ├── fabio_figueiredo_..._pln.pdf  # Relatório técnico (entrega)
│   └── images/                       # Visualizações + grafo interativo
├── data/
│   ├── processed/grafo_conhecimento.gexf
│   └── db/finnlp.sqlite
├── requirements.txt
└── README.md
```

---

## Como Executar o Projeto Localmente

> Ambiente recomendado: **Python 3.12** (CPU-only, sem GPU).

### 1. Ambiente virtual

```bash
# Com uv (recomendado)
uv venv .venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Ou com venv padrão
python3.12 -m venv .venv && source .venv/bin/activate
```

### 2. Dependências

```bash
uv pip install -r requirements.txt
#  (sem uv: pip install -r requirements.txt)
```

### 3. Modelos de linguagem do spaCy

```bash
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
```

### 4. Executar o notebook

```bash
jupyter notebook notebooks/FinNLP_Pipeline.ipynb
#  → Kernel > Restart & Run All
```

### 5. (Opcional) Abrir o MLflow

```bash
mlflow ui                          # http://localhost:5000
```

> A primeira execução baixa o `financial_phrasebank` (~50 MB) via Hugging Face.
> Requer conexão com a internet.

---

## Padrões de Engenharia

- **Complexidade ciclomática < 10** por função (SOLID / SRP).
- **DRY** — lógica centralizada em `src/`, reutilizada pelo notebook.
- **Reprodutibilidade**: `random_state=42` em todas as etapas estocásticas.
- **Narrativa em 1ª pessoa** nas docstrings e células Markdown.

---
<div align="center">
  <small>Desenvolvido para fins acadêmicos.<br>Maio / 2026</small>
</div>
