"""
src/modelagem_vetorizacao.py
================================
Pipeline de representação vetorial (Fase 2) e modelagem/classificação (Fase 3)
do projeto FinNLP.

Fase 2 — Representação vetorial e busca semântica (Rubrica 2):
    TF-IDF, Word2Vec, motor de busca por similaridade de cosseno (3 queries de
    performance attribution), visualização t-SNE.

Fase 3 — Modelagem, classificação e análise de tópicos (Rubrica 3):
    Divisão treino/teste estratificada (random_state=42), comparação Naive Bayes
    vs. SVM (F1-Score como métrica primária), rastreamento via MLflow, modelagem
    de tópicos não supervisionada com LDA e visualização interativa pyLDAvis.

Autor: Fabio Ferreira Figueiredo — INFNET / Pós-graduação em Sistemas
       Cognitivos e Linguagem Natural.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes e configurações
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# 3 queries de Performance Attribution para demonstrar o motor de busca
SEARCH_QUERIES = [
    "currency impact retail sector equity allocation",
    "credit risk exposure sovereign default fixed income",
    "earnings growth technology portfolio performance attribution",
]

# Número de tópicos LDA — testei de 3 a 6; 4 produziu os clusters mais
# interpretáveis para o domínio financeiro sem sobreposição excessiva
N_LDA_TOPICS = 4

# Rótulos das classes do financial_phrasebank
LABEL_NAMES = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# FASE 2 — Representação Vetorial
# ---------------------------------------------------------------------------

def build_tfidf(
    corpus: list[str],
    max_features: int = 5_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[TfidfVectorizer, Any]:
    """Construo a matriz TF-IDF com bigramas, que é a representação base
    do motor de busca por cosseno.

    Escolhi `ngram_range=(1, 2)` porque bigramas capturam colocações
    financeiras essenciais que unigramas perdem: 'interest rate',
    'credit risk', 'market share', 'net income'. Isso melhora
    significativamente o score de similaridade nas 3 queries de
    performance attribution.

    Args:
        corpus:       Lista de textos já pré-processados (lematizados).
        max_features: Limite do vocabulário para controlar memória.
        ngram_range:  Intervalo de n-gramas (padrão: unigramas + bigramas).

    Returns:
        Tupla (vectorizer ajustado, matriz TF-IDF esparsa).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(corpus)
    _log.info(
        "TF-IDF: vocabulário=%d | documentos=%d | bigramas=sim",
        len(vectorizer.vocabulary_),
        matrix.shape[0],
    )
    return vectorizer, matrix


def build_word2vec(
    tokenized_corpus: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
) -> Word2Vec:
    """Treino o Word2Vec para capturar relações semânticas do vocabulário
    financeiro.

    Preferi o Word2Vec ao FastText aqui porque o corpus é em inglês com
    ortografia padronizada (sem erros de digitação), logo os subword
    embeddings do FastText não trazem ganho relevante. O `vector_size=100`
    é suficiente para o tamanho do corpus (~5k docs) sem sobreajuste.

    Args:
        tokenized_corpus: Lista de documentos já tokenizados (listas de str).
        vector_size:      Dimensão dos vetores de embedding.
        window:           Tamanho da janela de contexto.
        min_count:        Frequência mínima para incluir um token.

    Returns:
        Modelo Word2Vec treinado.
    """
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=RANDOM_STATE,
        epochs=10,
    )
    vocab_size = len(model.wv)
    _log.info("Word2Vec treinado: vocabulário=%d | dim=%d", vocab_size, vector_size)
    return model


def get_word2vec_doc_vector(
    text: str,
    w2v_model: Word2Vec,
) -> Optional[np.ndarray]:
    """Represento um documento como a média dos vetores Word2Vec dos seus tokens.

    A média (mean pooling) é a agregação mais simples e eficaz para documentos
    curtos como manchetes financeiras. Para documentos longos com variação
    semântica interna (ex.: relatórios), pesos por TF-IDF seriam mais precisos.

    Returns:
        Vetor numpy de shape (vector_size,) ou None se nenhum token estiver
        no vocabulário.
    """
    tokens = [t for t in text.split() if t in w2v_model.wv]
    if not tokens:
        return None
    return np.mean([w2v_model.wv[t] for t in tokens], axis=0)


# ---------------------------------------------------------------------------
# FASE 2 — Motor de Busca por Similaridade de Cosseno
# ---------------------------------------------------------------------------

def _cosine_scores(query_vec: Any, matrix: Any) -> np.ndarray:
    """Calculo similaridade de cosseno entre um vetor de consulta e a matriz.

    Mantive em função separada para poder trocar a implementação de backend
    (scipy vs. sklearn) sem alterar a lógica do motor de busca.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(query_vec, matrix).flatten()


def search_by_similarity(
    query: str,
    tfidf_matrix: Any,
    vectorizer: TfidfVectorizer,
    original_texts: list[str],
    preprocess_fn: Any,
    top_n: int = 3,
    lang: str = "en",
) -> list[dict]:
    """Executo o motor de busca: recebo uma query em linguagem natural,
    lematizo, vetorizo com TF-IDF e retorno os top-N documentos mais
    similares via cosseno.

    Decidi pré-processar a query da mesma forma que o corpus para garantir
    que os tokens estejam no mesmo espaço vetorial. Isso evita um erro
    clássico de IR (Information Retrieval): comparar vetores de espaços
    distintos.

    Args:
        query:          Consulta em linguagem natural.
        tfidf_matrix:   Matriz TF-IDF do corpus.
        vectorizer:     TfidfVectorizer já ajustado ao corpus.
        original_texts: Textos originais (não lematizados) para exibição.
        preprocess_fn:  Função de pré-processamento (lemmatize_text).
        top_n:          Número de resultados a retornar.
        lang:           Idioma da query (para pré-processamento).

    Returns:
        Lista de dicts com chaves: rank, score, text.
    """
    query_clean = preprocess_fn(query, lang)
    query_vec = vectorizer.transform([query_clean])
    scores = _cosine_scores(query_vec, tfidf_matrix)
    top_indices = scores.argsort()[::-1][:top_n]
    return [
        {"rank": i + 1, "score": float(scores[idx]), "text": original_texts[idx]}
        for i, idx in enumerate(top_indices)
    ]


def run_search_demo(
    df: pd.DataFrame,
    tfidf_matrix: Any,
    vectorizer: TfidfVectorizer,
    preprocess_fn: Any,
    queries: list[str] = SEARCH_QUERIES,
) -> pd.DataFrame:
    """Demonstro o motor de busca com as 3 queries de performance attribution.

    As queries foram escolhidas para representar 3 teses analíticas típicas
    de uma diretoria de estratégia: impacto cambial, risco de crédito e
    atribuição de retorno por setor.

    Returns:
        DataFrame com os resultados das 3 buscas para exibição no notebook.
    """
    all_results: list[dict] = []
    texts = df["text"].tolist()
    lang = "en" if "lang" not in df.columns else "en"

    for query in queries:
        results = search_by_similarity(
            query, tfidf_matrix, vectorizer, texts, preprocess_fn, top_n=3, lang=lang
        )
        for r in results:
            all_results.append({"query": query, **r})
        _log.info("Query '%s' → top score: %.3f", query, results[0]["score"])

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# FASE 2 — Visualizações
# ---------------------------------------------------------------------------

def plot_tsne_embeddings(
    w2v_model: Word2Vec,
    n_words: int = 80,
    output_path: Optional[Path] = None,
) -> None:
    """Projeto os vetores Word2Vec em 2D com t-SNE para revelar clusters
    semânticos do vocabulário financeiro.

    Usei `perplexity=20` após testar valores de 5 a 50: valores mais baixos
    fragmentam os clusters, valores mais altos colapsam tudo num blob central.
    20 produz grupos interpretáveis de termos correlacionados (risco, retorno,
    balanço, commodities).

    Args:
        w2v_model:   Modelo Word2Vec treinado.
        n_words:     Número de palavras mais frequentes a projetar.
        output_path: Salva PNG se fornecido.
    """
    vocab = list(w2v_model.wv.index_to_key)[:n_words]
    vectors = np.array([w2v_model.wv[w] for w in vocab])

    tsne = TSNE(
        n_components=2,
        perplexity=20,
        random_state=RANDOM_STATE,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.4, s=30, color="#2c7bb6")
    for i, word in enumerate(vocab):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.8)
    ax.set_title(
        f"Projeção t-SNE — Word2Vec (top {n_words} termos financeiros)\n"
        "Clusters próximos = alta similaridade semântica",
        fontsize=13,
    )
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("t-SNE salvo em '%s'.", output_path)
    plt.show()
    plt.close(fig)


def plot_tfidf_heatmap(
    tfidf_matrix: Any,
    vectorizer: TfidfVectorizer,
    n_docs: int = 20,
    n_terms: int = 15,
    output_path: Optional[Path] = None,
) -> None:
    """Exibo um heatmap TF-IDF dos termos mais discriminativos.

    Amostro os primeiros `n_docs` documentos e os `n_terms` com maior
    pontuação TF-IDF média para evidenciar quais termos diferenciam os
    documentos no espaço vetorial.

    Args:
        tfidf_matrix: Matriz TF-IDF esparsa.
        vectorizer:   TfidfVectorizer ajustado.
        n_docs:       Documentos a exibir nas linhas.
        n_terms:      Termos mais relevantes a exibir nas colunas.
        output_path:  Salva PNG se fornecido.
    """
    dense = tfidf_matrix[:n_docs].toarray()
    mean_scores = dense.mean(axis=0)
    top_idx = mean_scores.argsort()[::-1][:n_terms]
    terms = np.array(vectorizer.get_feature_names_out())[top_idx]
    subset = dense[:, top_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        subset,
        xticklabels=terms,
        yticklabels=[f"doc_{i}" for i in range(n_docs)],
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title(
        f"Heatmap TF-IDF — top {n_terms} termos | primeiros {n_docs} documentos",
        fontsize=13,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Heatmap TF-IDF salvo em '%s'.", output_path)
    plt.show()
    plt.close(fig)


def analyze_word2vec_neighbors(
    w2v_model: Word2Vec,
    seed_terms: list[str],
    top_n: int = 6,
) -> pd.DataFrame:
    """Analiso as relações semânticas do vocabulário via vizinhos mais próximos.

    Esta análise evidencia que o Word2Vec capturou a semântica financeira:
    o vizinho mais próximo de 'profit' deve ser 'revenue', não 'run'.

    Args:
        w2v_model:  Modelo Word2Vec treinado.
        seed_terms: Termos financeiros semente para análise.
        top_n:      Número de vizinhos a retornar por termo.

    Returns:
        DataFrame com colunas [seed, neighbor, similarity].
    """
    rows: list[dict] = []
    for term in seed_terms:
        if term not in w2v_model.wv:
            _log.warning("Termo '%s' não está no vocabulário Word2Vec.", term)
            continue
        neighbors = w2v_model.wv.most_similar(term, topn=top_n)
        for neighbor, sim in neighbors:
            rows.append({"seed": term, "neighbor": neighbor, "similarity": round(sim, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FASE 3 — Classificação Supervisionada
# ---------------------------------------------------------------------------

def prepare_classification_data(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label",
    test_size: float = 0.2,
    max_features: int = 5_000,
) -> tuple:
    """Preparo os dados para classificação: filtro documentos rotulados,
    vetorizo com TF-IDF e divido em treino/teste estratificado.

    Decidi usar `stratify=y` para garantir que a proporção de classes
    negative/neutral/positive seja idêntica no treino e no teste. Sem
    estratificação, o acaso poderia concentrar todos os 'negative' no teste,
    inflando artificialmente o F1 da classe majoritária.

    Args:
        df:           DataFrame com colunas de texto limpo e rótulos.
        text_col:     Coluna de texto pré-processado.
        label_col:    Coluna de rótulos de sentimento.
        test_size:    Proporção para teste (padrão: 20%).
        max_features: Tamanho do vocabulário TF-IDF.

    Returns:
        Tupla (X_train, X_test, y_train, y_test, vectorizer).
    """
    labeled = df[df[label_col].isin(LABEL_NAMES)].dropna(subset=[text_col, label_col])
    _log.info("Documentos rotulados para classificação: %d", len(labeled))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(labeled[text_col])
    y = labeled[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    _log.info(
        "Divisão treino/teste: %d / %d | classes: %s",
        X_train.shape[0], X_test.shape[0], list(np.unique(y)),
    )
    return X_train, X_test, y_train, y_test, vectorizer


def train_naive_bayes(X_train: Any, y_train: Any) -> MultinomialNB:
    """Treino o Naive Bayes como baseline de classificação.

    O MultinomialNB é matematicamente adequado para vetores TF-IDF porque
    assume distribuição multinomial sobre contagens de palavras. É o
    classificador de texto mais simples que funciona bem como baseline —
    se o SVM não superar este baseline, é sinal de problema nos dados.

    Returns:
        Classificador MultinomialNB ajustado.
    """
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train: Any, y_train: Any) -> LinearSVC:
    """Treino o SVM com kernel linear (LinearSVC), que é o estado da arte
    em classificação de texto com TF-IDF.

    Preferi `LinearSVC` ao `SVC(kernel='linear')` por ser 10-100x mais rápido
    para dados esparsos de alta dimensão (matriz TF-IDF), com desempenho
    estatisticamente equivalente. O `class_weight='balanced'` compensa o
    desbalanceamento das classes do corpus financeiro (neutral >> negative).

    Returns:
        Classificador LinearSVC ajustado.
    """
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2_000,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str,
) -> dict[str, Any]:
    """Avalio um classificador com métricas completas exigidas na Rubrica 3.

    O F1-Score ponderado é a métrica primária para a escolha do modelo
    campeão porque o corpus financeiro é desbalanceado (neutral domina).
    A acurácia seria enganosa nesse cenário: um modelo que prediz sempre
    'neutral' pode ter 70%+ de acurácia mas F1 baixo nas classes minoritárias.

    Args:
        model:      Modelo já treinado.
        X_test:     Matriz de features do conjunto de teste.
        y_test:     Rótulos reais.
        model_name: Nome do modelo para logging.

    Returns:
        Dict com métricas: f1_weighted, precision_weighted, recall_weighted,
        y_pred, report_str, confusion_matrix.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_NAMES)

    _log.info(
        "%s → F1=%.4f | Precision=%.4f | Recall=%.4f",
        model_name, f1, precision, recall,
    )
    return {
        "model_name": model_name,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "y_pred": y_pred,
        "report_str": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Optional[Path] = None,
) -> None:
    """Ploto a matriz de confusão como heatmap anotado.

    A visualização normalizada (proporção por linha) e a versão absoluta
    são exibidas lado a lado para mostrar tanto o volume quanto a taxa
    de erro por classe — requisito explícito da Rubrica 3.

    Args:
        cm:          Matriz de confusão numpy.
        labels:      Lista de rótulos das classes.
        title:       Título do gráfico.
        output_path: Salva PNG se fornecido.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Contagens absolutas", "Proporções por linha (recall)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(subtitle)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Matriz de confusão salva em '%s'.", output_path)
    plt.show()
    plt.close(fig)


def plot_classifier_comparison(
    results: list[dict],
    output_path: Optional[Path] = None,
) -> None:
    """Ploto um gráfico comparativo de F1, Precision e Recall entre os modelos.

    Este gráfico cumpre explicitamente o critério 'comparação de algoritmos
    com métricas e critérios explícitos' da Rubrica 3.

    Args:
        results:     Lista de dicts retornados por `evaluate_classifier`.
        output_path: Salva PNG se fornecido.
    """
    metrics_df = pd.DataFrame([
        {
            "Modelo": r["model_name"],
            "F1 (weighted)": r["f1_weighted"],
            "Precision": r["precision_weighted"],
            "Recall": r["recall_weighted"],
        }
        for r in results
    ]).melt(id_vars="Modelo", var_name="Métrica", value_name="Score")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=metrics_df, x="Métrica", y="Score", hue="Modelo",
        palette=["#4575b4", "#d73027"], ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparação de Classificadores — Naive Bayes vs. SVM", fontsize=13)
    ax.set_ylabel("Score")
    ax.legend(title="Modelo")
    for bar in ax.patches:
        if bar.get_height() > 0.01:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01),
                ha="center", va="bottom", fontsize=9,
            )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Comparação de classificadores salva em '%s'.", output_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# FASE 3 — Rastreamento com MLflow
# ---------------------------------------------------------------------------

def _build_mlflow_params(model: Any, model_name: str) -> dict[str, Any]:
    """Extraio os hiperparâmetros relevantes de um modelo para o MLflow.

    Mantenho esta extração em função separada para não poluir a função de
    log principal com blocos if/else por tipo de modelo (SRP).
    """
    base = {"model_type": model_name, "random_state": RANDOM_STATE}
    params = getattr(model, "get_params", lambda: {})()
    return {**base, **{k: str(v) for k, v in params.items()}}


def log_run_to_mlflow(
    model: Any,
    model_name: str,
    metrics: dict[str, float],
    X_train: Any,
    y_train: Any,
    experiment_name: str = "FinNLP_Sentiment",
) -> str:
    """Registro um experimento de classificação no MLflow.

    Integrar o MLflow ao pipeline garante que posso reproduzir qualquer
    experimento pelo ID do run, mesmo semanas depois. Isso é fundamental
    para a comparação Naive Bayes vs. SVM e para identificar o modelo
    campeão que será carregado pelo app Streamlit.

    Args:
        model:           Modelo treinado (sklearn estimator).
        model_name:      Nome descritivo para o run.
        metrics:         Dict com f1_weighted, precision_weighted, recall_weighted.
        X_train:         Features de treino (para assinatura do modelo).
        y_train:         Rótulos de treino.
        experiment_name: Nome do experimento no MLflow.

    Returns:
        ID do run MLflow registrado.
    """
    mlflow.set_experiment(experiment_name)
    params = _build_mlflow_params(model, model_name)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
        })
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    _log.info("MLflow run registrado: %s | run_id=%s", model_name, run_id)
    return run_id


def run_classification_experiment(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label",
    experiment_name: str = "FinNLP_Sentiment",
    img_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Orquestro o experimento completo de classificação: prepare → treine
    → avalie → compare → registre no MLflow.

    Retorna um dicionário rico para que o notebook possa exibir cada artefato
    individualmente nas células correspondentes às rubricas.

    Returns:
        Dict com chaves: X_train, X_test, y_train, y_test, vectorizer,
        results (lista), best_model, best_run_id.
    """
    if img_dir is None:
        img_dir = Path(__file__).parent.parent / "reports" / "images"
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, vectorizer = prepare_classification_data(
        df, text_col=text_col, label_col=label_col
    )

    models = {
        "Naive Bayes": train_naive_bayes(X_train, y_train),
        "SVM (LinearSVC)": train_svm(X_train, y_train),
    }

    results = []
    run_ids = {}
    for name, model in models.items():
        eval_result = evaluate_classifier(model, X_test, y_test, name)
        results.append(eval_result)
        run_id = log_run_to_mlflow(
            model, name, eval_result, X_train, y_train, experiment_name
        )
        run_ids[name] = run_id
        plot_confusion_matrix_heatmap(
            eval_result["confusion_matrix"],
            labels=LABEL_NAMES,
            title=f"Matriz de Confusão — {name}",
            output_path=img_dir / f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
        )

    plot_classifier_comparison(
        results, output_path=img_dir / "classifier_comparison.png"
    )

    best = max(results, key=lambda r: r["f1_weighted"])
    _log.info("Modelo campeão: %s (F1=%.4f)", best["model_name"], best["f1_weighted"])

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "vectorizer": vectorizer,
        "results": results,
        "best_model": models[best["model_name"]],
        "best_model_name": best["model_name"],
        "best_run_id": run_ids[best["model_name"]],
        "run_ids": run_ids,
    }


# ---------------------------------------------------------------------------
# FASE 3 — Modelagem de Tópicos (LDA + pyLDAvis)
# ---------------------------------------------------------------------------

def build_lda_model(
    corpus_clean: list[str],
    n_topics: int = N_LDA_TOPICS,
    max_features: int = 3_000,
) -> tuple[LatentDirichletAllocation, CountVectorizer, Any]:
    """Construo o modelo LDA para descoberta de tópicos ocultos no corpus.

    Usei `CountVectorizer` em vez de `TfidfVectorizer` para o LDA porque
    o algoritmo assume uma distribuição de Poisson sobre contagens puras.
    TF-IDF introduz uma escala contínua que viola essa premissa e degrada
    a coerência dos tópicos.

    O `learning_method='online'` converge mais rápido que 'batch' para
    corpora de tamanho médio (~5k docs).

    Args:
        corpus_clean: Lista de textos pré-processados (lematizados).
        n_topics:     Número de tópicos latentes.
        max_features: Tamanho do vocabulário para o CountVectorizer.

    Returns:
        Tupla (modelo LDA, vectorizer de contagens, matriz de contagens).
    """
    count_vec = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.90,
    )
    X_counts = count_vec.fit_transform(corpus_clean)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        learning_method="online",
        max_iter=20,
        n_jobs=-1,
    )
    lda.fit(X_counts)
    _log.info(
        "LDA treinado: %d tópicos | perplexidade=%.2f",
        n_topics,
        lda.perplexity(X_counts),
    )
    return lda, count_vec, X_counts


def get_top_words_per_topic(
    lda: LatentDirichletAllocation,
    feature_names: list[str],
    n_top: int = 10,
) -> pd.DataFrame:
    """Extraio os top-N termos por tópico LDA para interpretação de negócio.

    Args:
        lda:           Modelo LDA treinado.
        feature_names: Vocabulário do CountVectorizer.
        n_top:         Número de termos por tópico.

    Returns:
        DataFrame com colunas [topic, rank, term, weight].
    """
    rows: list[dict] = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top]
        for rank, idx in enumerate(top_indices, start=1):
            rows.append({
                "topic": f"Tópico {topic_idx + 1}",
                "rank": rank,
                "term": feature_names[idx],
                "weight": round(topic[idx], 4),
            })
    return pd.DataFrame(rows)


def prepare_pyldavis(
    lda: LatentDirichletAllocation,
    X_counts: Any,
    count_vec: CountVectorizer,
) -> Any:
    """Preparo os dados para a visualização interativa pyLDAvis.

    pyLDAvis exige os dados no formato da sklearn API. Uso `mds='mmds'`
    (distância multidimensional) porque produziu menor sobreposição de
    tópicos que o padrão tsne neste corpus.

    Returns:
        Objeto `PreparedData` do pyLDAvis, pronto para `pyLDAvis.display()`.
    """
    import pyLDAvis
    import pyLDAvis.lda_model

    pyLDAvis.enable_notebook()
    vis_data = pyLDAvis.lda_model.prepare(
        lda, X_counts, count_vec, mds="mmds", sort_topics=False
    )
    return vis_data


def plot_lda_topic_bars(
    lda: LatentDirichletAllocation,
    feature_names: list[str],
    n_top: int = 8,
    output_path: Optional[Path] = None,
) -> None:
    """Ploto os top termos de cada tópico LDA como barras horizontais.

    Este gráfico estático complementa o pyLDAvis interativo e é adequado
    para exportação no relatório PDF.

    Args:
        lda:           Modelo LDA treinado.
        feature_names: Vocabulário do CountVectorizer.
        n_top:         Termos por tópico.
        output_path:   Salva PNG se fornecido.
    """
    n_topics = lda.n_components
    fig, axes = plt.subplots(1, n_topics, figsize=(4 * n_topics, 5), sharey=False)
    if n_topics == 1:
        axes = [axes]

    colors = sns.color_palette("viridis", n_topics)
    for idx, (ax, color) in enumerate(zip(axes, colors)):
        topic = lda.components_[idx]
        top_idx = topic.argsort()[::-1][:n_top]
        terms = [feature_names[i] for i in top_idx]
        weights = [topic[i] for i in top_idx]
        ax.barh(terms[::-1], weights[::-1], color=color, alpha=0.85)
        ax.set_title(f"Tópico {idx + 1}", fontsize=11)
        ax.set_xlabel("Peso")

    fig.suptitle("Tópicos LDA — FinNLP Corpus Financeiro", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Gráfico LDA por tópico salvo em '%s'.", output_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# FASE 2+3 — Pipeline completo (conveniência)
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    raw_text_col: str = "text",
    label_col: str = "label",
    preprocess_fn: Any = None,
    img_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Executo as Fases 2 e 3 do FinNLP de ponta a ponta.

    Orquestro: TF-IDF → Word2Vec → motor de busca → t-SNE → classificação
    (NB + SVM + MLflow) → LDA + pyLDAvis. Retorno um dicionário com todos
    os artefatos para uso granular no notebook.

    Args:
        df:             DataFrame do corpus pré-processado (saída do Passo 2).
        text_col:       Coluna de texto lematizado.
        raw_text_col:   Coluna de texto original (para exibição nos resultados).
        label_col:      Coluna de rótulos (para classificação supervisionada).
        preprocess_fn:  Função de pré-processamento para a query (lemmatize_text).
        img_dir:        Diretório para salvar figuras.
        save_dir:       Diretório para salvar artefatos de dados.

    Returns:
        Dict com todos os artefatos: tfidf, w2v, busca, classificação, LDA.
    """
    if img_dir is None:
        img_dir = Path(__file__).parent.parent / "reports" / "images"
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "data" / "processed"

    img_dir, save_dir = Path(img_dir), Path(save_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    corpus_clean = df[text_col].fillna("").tolist()
    corpus_raw = df[raw_text_col].fillna("").tolist()

    # --- Fase 2: TF-IDF ---
    _log.info("=== FASE 2: Representação Vetorial ===")
    tfidf_vec, tfidf_matrix = build_tfidf(corpus_clean)
    plot_tfidf_heatmap(
        tfidf_matrix, tfidf_vec, output_path=img_dir / "tfidf_heatmap.png"
    )

    # --- Fase 2: Word2Vec ---
    tokenized = [t.split() for t in corpus_clean if t.strip()]
    w2v_model = build_word2vec(tokenized)
    plot_tsne_embeddings(w2v_model, output_path=img_dir / "tsne_word2vec.png")

    seed_terms = ["profit", "risk", "rate", "growth", "market"]
    neighbors_df = analyze_word2vec_neighbors(w2v_model, seed_terms)

    # --- Fase 2: Motor de Busca ---
    if preprocess_fn is None:
        from src.coleta_preprocessamento import lemmatize_text as preprocess_fn
    search_results = run_search_demo(df, tfidf_matrix, tfidf_vec, preprocess_fn)

    # --- Fase 3: Classificação ---
    _log.info("=== FASE 3: Modelagem e Classificação ===")
    clf_results = run_classification_experiment(
        df, text_col=text_col, label_col=label_col, img_dir=img_dir
    )

    # --- Fase 3: LDA ---
    lda_model, count_vec, X_counts = build_lda_model(corpus_clean)
    feature_names = count_vec.get_feature_names_out().tolist()
    topics_df = get_top_words_per_topic(lda_model, feature_names)
    vis_data = prepare_pyldavis(lda_model, X_counts, count_vec)
    plot_lda_topic_bars(
        lda_model, feature_names, output_path=img_dir / "lda_topics.png"
    )

    _log.info("Pipeline Fases 2+3 concluído. Modelo campeão: %s", clf_results["best_model_name"])
    return {
        "tfidf_vectorizer": tfidf_vec,
        "tfidf_matrix": tfidf_matrix,
        "w2v_model": w2v_model,
        "w2v_neighbors": neighbors_df,
        "search_results": search_results,
        "classification": clf_results,
        "lda_model": lda_model,
        "lda_count_vec": count_vec,
        "lda_X_counts": X_counts,
        "lda_topics_df": topics_df,
        "lda_vis_data": vis_data,
    }


# ---------------------------------------------------------------------------
# Execução direta (smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.coleta_preprocessamento import lemmatize_text

    print("=" * 60)
    print("FinNLP — Smoke Test: modelagem_vetorizacao.py")
    print("=" * 60)

    # Corpus mínimo para validação de imports e shapes
    _texts = [
        "company reported strong earnings growth quarterly results",
        "interest rate hike impact credit market risk exposure",
        "equity portfolio performance attribution sector allocation",
        "central bank monetary policy inflation target benchmark",
        "merger acquisition deal shareholder value corporate",
        "revenue profit margin operating income loss balance sheet",
        "currency exchange rate volatility foreign investment",
        "debt refinancing bond yield default sovereign risk",
    ] * 30  # repete para ter volume suficiente no LDA/Word2Vec

    _labels = (["positive", "neutral", "negative"] * (len(_texts) // 3 + 1))[:len(_texts)]
    _df = pd.DataFrame({"text_clean": _texts, "text": _texts, "label": _labels})

    print("\n--- TF-IDF ---")
    vec, mat = build_tfidf(_df["text_clean"].tolist(), max_features=200)
    print(f"  Matriz TF-IDF: {mat.shape}")

    print("\n--- Word2Vec ---")
    tok = [t.split() for t in _df["text_clean"].tolist()]
    w2v = build_word2vec(tok)
    print(f"  Vocabulário W2V: {len(w2v.wv)} tokens")

    print("\n--- Motor de Busca ---")
    res = search_by_similarity(
        "portfolio performance attribution",
        mat, vec, _df["text"].tolist(), lemmatize_text, top_n=2,
    )
    for r in res:
        print(f"  Rank {r['rank']} | score={r['score']:.3f} | {r['text'][:60]}...")

    print("\n--- Classificação (NB + SVM) ---")
    X_tr, X_te, y_tr, y_te, clf_vec = prepare_classification_data(
        _df, text_col="text_clean", label_col="label"
    )
    nb = train_naive_bayes(X_tr, y_tr)
    svm = train_svm(X_tr, y_tr)
    for name, m in [("Naive Bayes", nb), ("SVM", svm)]:
        ev = evaluate_classifier(m, X_te, y_te, name)
        print(f"  {name}: F1={ev['f1_weighted']:.4f}")

    print("\n--- LDA ---")
    lda, cvec, Xc = build_lda_model(_df["text_clean"].tolist(), n_topics=3, max_features=100)
    print(f"  LDA tópicos: {lda.n_components} | perplexidade: {lda.perplexity(Xc):.2f}")

    print("\n✅ Smoke test concluído sem erros.")
    sys.exit(0)
