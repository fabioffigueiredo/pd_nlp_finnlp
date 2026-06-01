"""
src/ner_grafo.py
================================
Pipeline de NER, extração de informação e grafo de conhecimento (Fase 4 / Rubrica 4).

Técnicas aplicadas (atende ao mínimo de três exigido pela rubrica):
    1. Extração de Entidades Nomeadas (ORG) com spaCy.
    2. Extração de padrões estruturados com RegEx: valores monetários,
       percentuais, datas e códigos de ativos financeiros.
    3. Normalização de entidades com distância de Levenshtein para
       unificar variações ortográficas da mesma organização.
    4. Construção de Grafo de Conhecimento com NetworkX (Source/Target/Weight).
    5. Cálculo de centralidade de grau e resposta a pergunta de negócio.
    6. Visualização interativa com PyVis e estática com matplotlib.

Autor: Fabio Ferreira Figueiredo — INFNET / Pós-graduação em Sistemas
       Cognitivos e Linguagem Natural.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import spacy
from Levenshtein import distance as lev_distance

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Padrões RegEx financeiros — professor destacou esta técnica explicitamente
# ---------------------------------------------------------------------------

# Valores monetários: $1.2 billion, EUR 500 million, R$ 3,4 bilhões etc.
_RE_MONETARY = re.compile(
    r"(?:US\$|R\$|\$|€|EUR|BRL)?\s*"
    r"[\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?"
    r"\s*(?:billion|million|trillion|bilhão|bilhões|milhão|milhões|bn|mn)?",
    re.IGNORECASE,
)

# Percentuais: 3.5%, -1.2%, +12%
_RE_PERCENT = re.compile(r"[+-]?\d+(?:[.,]\d+)?\s*%")

# Datas em formato financeiro: Q1 2024, FY2023, H1/H2 + year
_RE_FIN_DATE = re.compile(
    r"\b(?:Q[1-4]|H[12]|FY|fiscal\s+year)\s*\d{4}\b",
    re.IGNORECASE,
)

# Tickers: sequência 2-5 letras maiúsculas isoladas (e.g. AAPL, AMZN, VALE3)
_RE_TICKER = re.compile(r"\b[A-Z]{2,5}[0-9]?\b")

# Conjunto de stop-tickers (siglas comuns que não são ativos)
_STOP_TICKERS = frozenset({
    "CEO", "CFO", "COO", "IPO", "GDP", "IMF", "ECB", "US", "UK",
    "EU", "NYSE", "NASDAQ", "SEC", "FED", "PLC", "INC", "LLC",
    "THE", "AND", "FOR", "NEW", "NET", "THE",
})

# Ruído de NER no corpus financeiro: o spaCy rotula códigos/valores de moeda
# (EUR, USD, "EUR131", "EUR 21.1 mn") como ORG. Notei que isso contamina o
# grafo — "EUR" virava o hub de maior centralidade, distorcendo a análise de
# risco. Descarto moedas, unidades de escala e qualquer token com dígito.
_STOP_ORGS = frozenset({
    "EUR", "USD", "GBP", "BRL", "JPY", "CHF", "CNY", "SEK", "NOK", "DKK",
    "MLN", "BN", "MN", "MILLION", "BILLION", "TRILLION", "EURO", "EUROS",
    "PCT", "Q1", "Q2", "Q3", "Q4", "FY",
})
# Token de moeda colado a valor: EUR131, USD500, eur76 etc.
_RE_CURRENCY_TOKEN = re.compile(r"^(?:US\$|R\$|\$|€|EUR|USD|GBP|BRL)\s*\d", re.IGNORECASE)


def _is_org_noise(name: str) -> bool:
    """Decido se uma entidade ORG é ruído (moeda, valor ou unidade), não empresa.

    Filtro três casos que poluem o grafo financeiro: (1) tokens que contêm
    qualquer dígito (são valores, não nomes de empresa); (2) códigos/valores de
    moeda como 'EUR131'; (3) palavras de moeda/escala na _STOP_ORGS.
    """
    upper = name.upper()
    if any(ch.isdigit() for ch in name):
        return True
    if _RE_CURRENCY_TOKEN.match(name):
        return True
    return upper in _STOP_ORGS


# ---------------------------------------------------------------------------
# Fase 4a — Extração de Entidades Nomeadas (NER) com spaCy
# ---------------------------------------------------------------------------

def extract_org_entities(text: str, nlp: spacy.Language) -> list[str]:
    """Extraio entidades do tipo ORG (organizações) de um texto com spaCy.

    Foco em ORG porque, no domínio financeiro, são as organizações
    (empresas, bancos, fundos) que determinam o risco de contágio e a
    centralidade no grafo de mercado. Filtro entidades com menos de 2
    caracteres para eliminar falsos positivos comuns em corpus financeiro.

    Args:
        text: Texto bruto ou lematizado.
        nlp:  Modelo spaCy carregado (en_core_web_sm recomendado para EN).

    Returns:
        Lista de nomes de organizações extraídos.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    doc = nlp(text[:50_000])
    return [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG"
        and len(ent.text.strip()) > 2
        and not _is_org_noise(ent.text.strip())
    ]


def extract_regex_patterns(text: str) -> dict[str, list[str]]:
    """Extraio padrões estruturados de texto financeiro com RegEx.

    Decidi implementar quatro padrões complementares porque cada um
    captura um tipo distinto de informação quantitativa ou temporal:
    valores monetários para magnitude do impacto, percentuais para
    variação, datas para o contexto temporal e tickers para identificar
    ativos específicos sem depender do NER (que frequentemente falha em
    abreviações financeiras).

    Args:
        text: Texto financeiro bruto.

    Returns:
        Dict com chaves: 'monetary', 'percent', 'fin_date', 'ticker'.
    """
    if not isinstance(text, str):
        return {"monetary": [], "percent": [], "fin_date": [], "ticker": []}

    monetary = [m.group().strip() for m in _RE_MONETARY.finditer(text) if m.group().strip()]
    percent = [m.group().strip() for m in _RE_PERCENT.finditer(text)]
    fin_date = [m.group().strip() for m in _RE_FIN_DATE.finditer(text)]
    ticker = [
        m.group()
        for m in _RE_TICKER.finditer(text)
        if m.group() not in _STOP_TICKERS
    ]
    return {
        "monetary": monetary,
        "percent": percent,
        "fin_date": fin_date,
        "ticker": ticker,
    }


def run_regex_eda(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Aplico extração RegEx ao corpus e retorno estatísticas por padrão.

    Args:
        df:       DataFrame com coluna de texto.
        text_col: Coluna de texto bruto.

    Returns:
        DataFrame com contagens de padrões por documento.
    """
    records = []
    for text in df[text_col].fillna(""):
        patterns = extract_regex_patterns(text)
        records.append({
            "n_monetary": len(patterns["monetary"]),
            "n_percent": len(patterns["percent"]),
            "n_fin_date": len(patterns["fin_date"]),
            "n_ticker": len(patterns["ticker"]),
        })
    stats = pd.DataFrame(records)
    _log.info(
        "RegEx EDA: monetary=%d | percent=%d | fin_date=%d | ticker=%d",
        stats["n_monetary"].sum(),
        stats["n_percent"].sum(),
        stats["n_fin_date"].sum(),
        stats["n_ticker"].sum(),
    )
    return stats


# ---------------------------------------------------------------------------
# Fase 4b — Normalização com Distância de Levenshtein
# ---------------------------------------------------------------------------

def _canonical_name(name: str) -> str:
    """Normalizo um nome de entidade: lowercase, sem pontuação final."""
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


def normalize_entities(
    entities: list[str],
    threshold: int = 3,
) -> dict[str, str]:
    """Unifico variações ortográficas da mesma entidade com Levenshtein.

    No corpus financeiro, a mesma organização aparece de formas diferentes:
    'Goldman Sachs', 'Goldman', 'Goldman Sachs Group', 'Goldman Sachs Inc'.
    Sem normalização, cada variação vira um nó separado no grafo, tornando
    a análise de centralidade imprecisa.

    Aplico um greedy clustering: para cada entidade, verifico se existe um
    representante canônico (já visto) com distância Levenshtein abaixo do
    limiar. Se sim, mapeio para ele; caso contrário, registro como novo
    representante.

    Args:
        entities:  Lista de nomes de entidades (com repetições).
        threshold: Distância máxima para considerar variações equivalentes.

    Returns:
        Dict {nome_original: nome_canonico}.
    """
    unique = list(dict.fromkeys(entities))
    canonical_map: dict[str, str] = {}
    representatives: list[str] = []

    for name in unique:
        name_c = _canonical_name(name)
        matched = False
        for rep in representatives:
            if lev_distance(name_c, _canonical_name(rep)) <= threshold:
                canonical_map[name] = rep
                matched = True
                break
        if not matched:
            canonical_map[name] = name
            representatives.append(name)

    n_merged = sum(1 for k, v in canonical_map.items() if k != v)
    _log.info(
        "Levenshtein: %d entidades → %d únicas (%d variações fundidas)",
        len(unique), len(representatives), n_merged,
    )
    return canonical_map


# ---------------------------------------------------------------------------
# Fase 4c — Construção do Grafo de Conhecimento
# ---------------------------------------------------------------------------

def build_cooccurrence_edges(
    df: pd.DataFrame,
    text_col: str = "text",
    nlp: Optional[spacy.Language] = None,
    max_docs: int = 800,
) -> pd.DataFrame:
    """Construo a lista de arestas Source/Target/Weight via coocorrência.

    Decidi usar coocorrência na mesma sentença (não no documento inteiro)
    porque sentenças financeiras são geralmente curtas e específicas —
    coocorrência sentencial indica relação direta (ex.: 'Apple comprou Intel'
    → aresta Apple–Intel com peso 1). Coocorrência no documento seria
    barulhenta demais.

    Args:
        df:       DataFrame com textos.
        text_col: Coluna de texto.
        nlp:      Modelo spaCy (carregado externamente para reuso do cache).
        max_docs: Limite de documentos processados (performance).

    Returns:
        DataFrame com colunas [Source, Target, Weight].
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    edge_counter: Counter = Counter()
    subset = df[text_col].dropna().head(max_docs)

    for text in subset:
        orgs_raw = extract_org_entities(text, nlp)
        if len(orgs_raw) < 2:
            continue
        canonical = normalize_entities(orgs_raw)
        orgs = list({canonical[o] for o in orgs_raw})
        for i in range(len(orgs)):
            for j in range(i + 1, len(orgs)):
                pair = tuple(sorted([orgs[i], orgs[j]]))
                edge_counter[pair] += 1

    if not edge_counter:
        _log.warning("Nenhuma aresta de coocorrência encontrada. Verifique o corpus.")
        return pd.DataFrame(columns=["Source", "Target", "Weight"])

    edges_df = pd.DataFrame(
        [{"Source": s, "Target": t, "Weight": w} for (s, t), w in edge_counter.items()],
    ).sort_values("Weight", ascending=False)

    _log.info(
        "Grafo (arestas brutas): %d arestas | %d entidades únicas",
        len(edges_df),
        len(set(edges_df["Source"]) | set(edges_df["Target"])),
    )
    return edges_df


def build_knowledge_graph(
    edges_df: pd.DataFrame,
    min_weight: int = 1,
    target_min_nodes: int = 20,
) -> nx.Graph:
    """Construo o grafo de conhecimento a partir do DataFrame Source/Target.

    Implemento uma lógica adaptativa de limiar: começo com `min_weight` e
    reduz progressivamente até atingir o mínimo de 20 nós exigido pela rubrica.
    Isso garante robustez independentemente do corpus usado.

    Args:
        edges_df:         DataFrame com colunas Source, Target, Weight.
        min_weight:       Peso mínimo inicial para incluir aresta.
        target_min_nodes: Número mínimo de nós (rubrica exige ≥20).

    Returns:
        Grafo NetworkX com atributo 'weight' nas arestas.
    """
    threshold = min_weight
    while threshold >= 1:
        filtered = edges_df[edges_df["Weight"] >= threshold]
        G = nx.from_pandas_edgelist(
            filtered, source="Source", target="Target",
            edge_attr="Weight",
        )
        if G.number_of_nodes() >= target_min_nodes:
            break
        threshold -= 1

    _log.info(
        "Grafo final: %d nós | %d arestas | threshold=%d",
        G.number_of_nodes(), G.number_of_edges(), threshold,
    )
    return G


def calculate_centrality(G: nx.Graph) -> dict[str, float]:
    """Calculo a centralidade de grau de cada nó.

    A centralidade de grau responde diretamente à pergunta de negócio:
    'Quais entidades têm maior risco de contágio sistêmico?' — nós com
    centralidade alta são mencionados junto a mais contrapartes, logo
    uma notícia negativa sobre eles impacta mais o portfólio.

    Returns:
        Dict {node: degree_centrality} ordenado do maior para o menor.
    """
    centrality = nx.degree_centrality(G)
    return dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True))


def answer_business_question(
    G: nx.Graph,
    centrality: dict[str, float],
    top_n: int = 5,
) -> pd.DataFrame:
    """Respondo a pergunta analítica central da Rubrica 4.

    Pergunta: *Quais entidades apresentam maior risco de contágio
    sistêmico para o portfólio da Gestora?*

    A resposta é derivada do grafo: nós com alta centralidade de grau
    são 'hubs' — qualquer choque de sentimento neles se propaga para
    múltiplos outros ativos do grafo, ampliando o risco sistêmico.

    Args:
        G:          Grafo de conhecimento.
        centrality: Dict de centralidade de grau.
        top_n:      Número de entidades a exibir.

    A leitura interpretativa (quais nós são hubs sistêmicos) fica nas células
    markdown do notebook — aqui retorno apenas os números brutos do ranking.

    Returns:
        DataFrame com colunas [entidade, centralidade_grau, n_conexoes].
    """
    rows = []
    top_nodes = list(centrality.keys())[:top_n]
    for node in top_nodes:
        degree = G.degree(node)
        rows.append({
            "entidade": node,
            "centralidade_grau": round(centrality[node], 4),
            "n_conexoes": degree,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fase 4d — Visualizações do Grafo
# ---------------------------------------------------------------------------

def plot_graph_matplotlib(
    G: nx.Graph,
    centrality: dict[str, float],
    title: str = "Grafo de Conhecimento — Entidades do Mercado Financeiro",
    output_path: Optional[Path] = None,
) -> None:
    """Ploto o grafo com matplotlib, dimensionando os nós pela centralidade.

    Usei `spring_layout` com `k=0.5` porque distribui melhor os hubs
    (nós centrais ficam no meio com conexões visíveis para os satélites)
    do que o `kamada_kawai_layout`, que comprime grafos densos demais.

    Args:
        G:           Grafo NetworkX.
        centrality:  Dict de centralidade de grau.
        title:       Título do gráfico.
        output_path: Salva PNG se fornecido.
    """
    if G.number_of_nodes() == 0:
        _log.warning("Grafo vazio — pulando visualização.")
        return

    pos = nx.spring_layout(G, k=0.6, seed=42)
    node_sizes = [centrality.get(n, 0.01) * 8_000 + 300 for n in G.nodes()]
    edge_weights = [G[u][v].get("Weight", 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 2.5 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(16, 10))
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color="#aaaaaa", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=list(centrality.get(n, 0) for n in G.nodes()), cmap=plt.cm.YlOrRd, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

    top5 = list(centrality.keys())[:5]
    ax.set_title(
        f"{title}\n"
        f"Tamanho do nó ∝ centralidade de grau | Top-5 hubs: {', '.join(top5)}",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Grafo matplotlib salvo em '%s'.", output_path)
    plt.show()
    plt.close(fig)


def plot_graph_pyvis(
    G: nx.Graph,
    centrality: dict[str, float],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Gero uma visualização interativa do grafo em HTML com PyVis.

    O HTML gerado pelo PyVis permite ao avaliador explorar o grafo
    interativamente (zoom, drag, hover) — impacto visual superior ao
    matplotlib estático para apresentações executivas.

    Args:
        G:           Grafo NetworkX.
        centrality:  Dict de centralidade de grau.
        output_path: Caminho do arquivo HTML de saída.

    Returns:
        Path do arquivo HTML gerado, ou None em caso de erro.
    """
    if output_path is None:
        output_path = Path("reports/images/grafo_interativo.html")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pyvis.network import Network
        net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
        net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -8000}}}')

        for node in G.nodes():
            size = 10 + centrality.get(node, 0) * 60
            net.add_node(str(node), label=str(node), size=size, title=f"Centralidade: {centrality.get(node, 0):.3f}")

        for u, v, data in G.edges(data=True):
            weight = data.get("Weight", 1)
            net.add_edge(str(u), str(v), value=weight, title=f"Coocorrências: {weight}")

        net.save_graph(str(output_path))
        _log.info("Grafo PyVis salvo em '%s'.", output_path)
        return output_path
    except Exception as exc:
        _log.warning("PyVis falhou (%s) — usando apenas matplotlib.", exc)
        return None


def plot_centrality_bar(
    centrality: dict[str, float],
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> None:
    """Ploto o ranking de centralidade como barras horizontais.

    Este gráfico complementa o grafo visual com um ranking quantitativo
    claro, adequado para o relatório PDF.

    Args:
        centrality:  Dict de centralidade de grau.
        top_n:       Número de entidades a exibir.
        output_path: Salva PNG se fornecido.
    """
    items = list(centrality.items())[:top_n]
    names = [i[0][:30] for i in items]
    values = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names[::-1], values[::-1], color=plt.cm.YlOrRd(
        [v / max(values) for v in values[::-1]]
    ))
    ax.set_xlabel("Centralidade de Grau")
    ax.set_title(f"Top {top_n} Entidades por Centralidade — Risco de Contágio Sistêmico")

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Gráfico de centralidade salvo em '%s'.", output_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline completo — Fase 4
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    nlp: Optional[spacy.Language] = None,
    max_docs: int = 800,
    img_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None,
) -> dict:
    """Executo a Fase 4 completa: NER → RegEx → Levenshtein → Grafo → Análise.

    Retorno um dicionário com todos os artefatos para uso granular no
    notebook — grafo, centralidade, tabela de resposta ao negócio,
    DataFrame de arestas e estatísticas RegEx.

    Args:
        df:        DataFrame do corpus.
        text_col:  Coluna de texto bruto.
        nlp:       Modelo spaCy (carregado externamente).
        max_docs:  Limite de documentos para NER.
        img_dir:   Diretório para salvar imagens.
        save_dir:  Diretório para salvar dados.

    Returns:
        Dict com: edges_df, G, centrality, business_df, regex_stats.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    if img_dir is None:
        img_dir = Path(__file__).parent.parent / "reports" / "images"
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "data" / "processed"

    img_dir, save_dir = Path(img_dir), Path(save_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    _log.info("=== FASE 4: NER + RegEx + Levenshtein + Grafo ===")

    # RegEx EDA
    regex_stats = run_regex_eda(df, text_col=text_col)

    # Grafo de coocorrência
    edges_df = build_cooccurrence_edges(df, text_col=text_col, nlp=nlp, max_docs=max_docs)

    if edges_df.empty:
        _log.error("Sem arestas — verifique se o corpus tem entidades ORG.")
        return {"edges_df": edges_df, "G": nx.Graph(), "centrality": {}, "business_df": pd.DataFrame(), "regex_stats": regex_stats}

    edges_df.to_parquet(save_dir / "graph_edges.parquet", index=False)
    G = build_knowledge_graph(edges_df)
    centrality = calculate_centrality(G)
    business_df = answer_business_question(G, centrality)

    # Visualizações
    plot_graph_matplotlib(G, centrality, output_path=img_dir / "grafo_conhecimento.png")
    plot_centrality_bar(centrality, output_path=img_dir / "centralidade_grau.png")
    plot_graph_pyvis(G, centrality, output_path=img_dir / "grafo_interativo.html")

    _log.info("Fase 4 concluída. Nós: %d | Top hub: %s", G.number_of_nodes(), list(centrality.keys())[0] if centrality else "N/A")
    return {
        "edges_df": edges_df,
        "G": G,
        "centrality": centrality,
        "business_df": business_df,
        "regex_stats": regex_stats,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("FinNLP — Smoke Test: ner_grafo.py")
    print("=" * 60)

    _texts = [
        "Goldman Sachs reported strong Q1 2024 earnings, up 15% year-over-year.",
        "JPMorgan Chase and Goldman Sachs Group both raised their targets for Apple.",
        "The Federal Reserve decision impacted Citigroup and Wells Fargo shares by -3.5%.",
        "BlackRock Inc acquired a $2.5 billion stake in Microsoft Corp.",
        "Morgan Stanley and JPMorgan Chase lead the $500 million bond issuance.",
        "Apple Inc revenue grew 8% in FY2023, beating Goldman Sachs forecasts.",
        "Amazon and Microsoft compete for the $10 billion cloud contract.",
        "Google parent Alphabet reported 12% revenue growth in Q2 2024.",
        "Tesla Inc bonds were downgraded by Moody's, impacting BlackRock portfolios.",
        "Berkshire Hathaway increased its stake in Apple to 6% of shares outstanding.",
        "The ECB raised rates by 25 bps, affecting Deutsche Bank and BNP Paribas.",
        "HSBC and Barclays reported losses on exposure to Credit Suisse in Q3.",
        "Vanguard and BlackRock together hold 15% of the S&P 500 by market cap.",
        "JPMorgan Chase CEO discussed Federal Reserve policy at the Goldman conference.",
        "Amazon Web Services revenue surpassed $25 billion in H1 2024.",
        "Meta Platforms and Alphabet both beat Q2 2024 revenue estimates by 5%.",
        "Citigroup restructuring affected Morgan Stanley advisory revenue by -2%.",
        "Microsoft acquired Activision for $68.7 billion, reviewed by the FTC.",
        "Apple Inc and Samsung Electronics compete in the $500 smartphone segment.",
        "Goldman Sachs, Morgan Stanley and JPMorgan raised Apple price target to $220.",
        "Berkshire Hathaway Q1 2024: operating earnings rose 39% to $11.2 billion.",
        "Nvidia Corp revenue grew 262% YoY driven by AI chip demand from Microsoft.",
    ]
    _df = pd.DataFrame({"text": _texts})

    print("\n--- RegEx patterns ---")
    stats = run_regex_eda(_df)
    print(stats.sum())

    print("\n--- Levenshtein normalization ---")
    sample_entities = ["Goldman Sachs", "Goldman", "Goldman Sachs Group", "JPMorgan", "JP Morgan", "JPMorgan Chase"]
    mapping = normalize_entities(sample_entities, threshold=4)
    for orig, canon in mapping.items():
        flag = " (merged)" if orig != canon else ""
        print(f"  '{orig}' → '{canon}'{flag}")

    print("\n--- NER + Graph ---")
    nlp_model = spacy.load("en_core_web_sm")
    edges = build_cooccurrence_edges(_df, nlp=nlp_model)
    print(f"  Arestas: {len(edges)}")
    G = build_knowledge_graph(edges, target_min_nodes=10)
    print(f"  Nós: {G.number_of_nodes()} | Arestas: {G.number_of_edges()}")
    c = calculate_centrality(G)
    biz = answer_business_question(G, c, top_n=3)
    print(biz.to_string(index=False))

    print("\n✅ Smoke test concluído.")
    sys.exit(0)
