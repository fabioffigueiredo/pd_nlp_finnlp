"""
src/scd2_manager.py
================================
Engenharia de Dados com SCD Tipo 2 (Slowly Changing Dimension Type 2).

Implementa a tabela `Dim_Ativo_Status` em SQLite via SQLAlchemy para
rastrear o histórico temporal de sentimento e centralidade das entidades
extraídas pelo pipeline de NER e classificação.

Lógica de versionamento (SCD2 Upsert):
    1. Verifica o registro ativo atual (status_ativo=1) para a entidade.
    2. Se o sentimento mudou → UPDATE no registro antigo (fecha data_fim,
       status_ativo=0) + INSERT de novo registro com data_inicio=hoje.
    3. Se o sentimento não mudou → apenas atualiza o score de centralidade.

Esta abordagem permite análises de "time travel": a Gestora pode visualizar
como o sentimento de uma entidade evoluiu ao longo do tempo, viabilizando
o backtesting de teses de investimento.

Autor: Fabio Ferreira Figueiredo — INFNET / Pós-graduação em Sistemas
       Cognitivos e Linguagem Natural.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modelo ORM
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


class DimAtivoStatus(_Base):
    """Tabela dimensional de status histórico das entidades do mercado.

    Cada linha representa uma versão do sentimento de uma entidade.
    A versão ativa tem status_ativo=1 e data_fim=NULL.
    """

    __tablename__ = "Dim_Ativo_Status"

    id_versao = Column(Integer, primary_key=True, autoincrement=True)
    nome_entidade = Column(String(200), nullable=False, index=True)
    sentimento = Column(String(20), nullable=False)          # positive/neutral/negative
    topico_lda = Column(Integer, nullable=True)              # tópico LDA (0..N-1)
    score_centralidade = Column(Float, nullable=True)        # degree centrality [0..1]
    data_inicio = Column(Date, nullable=False)
    data_fim = Column(Date, nullable=True)                   # NULL = registro atual
    status_ativo = Column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return (
            f"<DimAtivoStatus(entidade='{self.nome_entidade}', "
            f"sentimento='{self.sentimento}', ativo={self.status_ativo})>"
        )


# ---------------------------------------------------------------------------
# Engine e criação de schema
# ---------------------------------------------------------------------------

def get_engine(db_path: Optional[Path] = None):
    """Retorno um engine SQLAlchemy para o banco SQLite do projeto.

    Preferi SQLite por ser serverless e adequado para projetos acadêmicos
    locais — sem necessidade de configurar um servidor de banco de dados.
    Para um ambiente de produção, bastaria trocar a connection string para
    PostgreSQL ou outro SGBD suportado pelo SQLAlchemy.

    Args:
        db_path: Caminho do arquivo .sqlite. Se None, usa o padrão do projeto.

    Returns:
        Engine SQLAlchemy configurado.
    """
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "db" / "finnlp.sqlite"
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _log.info("Engine SQLite criado: %s", db_path)
    return engine


def create_schema(engine) -> None:
    """Crio o schema do banco (idempotente — não sobrescreve dados existentes).

    Args:
        engine: Engine SQLAlchemy.
    """
    _Base.metadata.create_all(engine)
    _log.info("Schema Dim_Ativo_Status criado (ou já existe).")


# ---------------------------------------------------------------------------
# Operações SCD2
# ---------------------------------------------------------------------------

def get_current_record(
    session: Session,
    nome_entidade: str,
) -> Optional[DimAtivoStatus]:
    """Busco o registro ativo atual de uma entidade.

    Args:
        session:       Sessão SQLAlchemy.
        nome_entidade: Nome da entidade a consultar.

    Returns:
        Registro ativo ou None se a entidade não tiver histórico.
    """
    return (
        session.query(DimAtivoStatus)
        .filter(
            DimAtivoStatus.nome_entidade == nome_entidade,
            DimAtivoStatus.status_ativo.is_(True),
        )
        .first()
    )


def _close_record(session: Session, record: DimAtivoStatus, today: date) -> None:
    """Fecho um registro SCD2 (UPDATE): seto data_fim e status_ativo=False.

    Mantenho esta operação separada para respeitar o SRP — a decisão de
    'fechar ou não' fica na função de upsert, e o 'como fechar' fica aqui.
    """
    record.data_fim = today
    record.status_ativo = False


def _insert_record(
    session: Session,
    nome_entidade: str,
    sentimento: str,
    topico_lda: Optional[int],
    score_centralidade: Optional[float],
    today: date,
) -> DimAtivoStatus:
    """Insiro um novo registro ativo para uma entidade."""
    novo = DimAtivoStatus(
        nome_entidade=nome_entidade,
        sentimento=sentimento,
        topico_lda=topico_lda,
        score_centralidade=score_centralidade,
        data_inicio=today,
        data_fim=None,
        status_ativo=True,
    )
    session.add(novo)
    return novo


def upsert_entity_status(
    engine,
    nome_entidade: str,
    novo_sentimento: str,
    topico_lda: Optional[int] = None,
    score_centralidade: Optional[float] = None,
    reference_date: Optional[date] = None,
) -> str:
    """Aplico a regra SCD Tipo 2 para uma entidade.

    Fluxo:
        - Se não há registro ativo → INSERT (primeira ocorrência).
        - Se há registro ativo com o mesmo sentimento → UPDATE só do score.
        - Se sentimento mudou → CLOSE registro antigo + INSERT novo.

    Args:
        engine:            Engine SQLAlchemy.
        nome_entidade:     Nome canônico da entidade.
        novo_sentimento:   Sentimento atual do pipeline ('positive'/'neutral'/'negative').
        topico_lda:        Tópico LDA associado (opcional).
        score_centralidade: Centralidade de grau atual (opcional).
        reference_date:    Data de referência (padrão: hoje).

    Returns:
        String descrevendo a operação realizada: 'inserted', 'updated', 'unchanged'.
    """
    today = reference_date or date.today()

    with Session(engine) as session:
        current = get_current_record(session, nome_entidade)

        if current is None:
            _insert_record(session, nome_entidade, novo_sentimento,
                           topico_lda, score_centralidade, today)
            session.commit()
            _log.debug("SCD2 INSERT: %s (%s)", nome_entidade, novo_sentimento)
            return "inserted"

        if current.sentimento != novo_sentimento:
            _close_record(session, current, today)
            _insert_record(session, nome_entidade, novo_sentimento,
                           topico_lda, score_centralidade, today)
            session.commit()
            _log.debug(
                "SCD2 CHANGE: %s | %s → %s",
                nome_entidade, current.sentimento, novo_sentimento,
            )
            return "updated"

        # Mesmo sentimento — atualiza apenas score sem versionar
        current.score_centralidade = score_centralidade
        current.topico_lda = topico_lda
        session.commit()
        return "unchanged"


def run_scd2_batch(
    engine,
    entities_df: pd.DataFrame,
    entity_col: str = "entidade",
    sentiment_col: str = "sentimento",
    topic_col: Optional[str] = "topico_lda",
    centrality_col: Optional[str] = "centralidade",
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """Processo um lote de entidades com a regra SCD2.

    Ideal para chamar após cada execução do pipeline de NER+classificação,
    registrando o estado atual de todas as entidades extraídas.

    Args:
        engine:         Engine SQLAlchemy.
        entities_df:    DataFrame com entidades e seus atributos.
        entity_col:     Coluna com nome da entidade.
        sentiment_col:  Coluna com sentimento predito.
        topic_col:      Coluna com tópico LDA (opcional).
        centrality_col: Coluna com centralidade de grau (opcional).
        reference_date: Data de referência para os registros.

    Returns:
        DataFrame de resumo com colunas [entidade, operacao].
    """
    create_schema(engine)
    results: list[dict] = []

    for _, row in entities_df.iterrows():
        nome = str(row[entity_col])
        sentimento = str(row[sentiment_col])
        topico = int(row[topic_col]) if topic_col and topic_col in row else None
        centralidade = float(row[centrality_col]) if centrality_col and centrality_col in row else None

        operacao = upsert_entity_status(
            engine, nome, sentimento,
            topico_lda=topico,
            score_centralidade=centralidade,
            reference_date=reference_date,
        )
        results.append({"entidade": nome, "operacao": operacao})

    summary = pd.DataFrame(results)
    counts = summary["operacao"].value_counts()
    _log.info(
        "SCD2 batch: %d inseridos | %d atualizados | %d sem mudança",
        counts.get("inserted", 0),
        counts.get("updated", 0),
        counts.get("unchanged", 0),
    )
    return summary


# ---------------------------------------------------------------------------
# Consultas analíticas
# ---------------------------------------------------------------------------

def query_current_status(engine) -> pd.DataFrame:
    """Retorno a visão atual de todas as entidades (status_ativo=1).

    Esta é a consulta mais usada pela equipe de estratégia:
    'Qual o sentimento atual de cada ativo monitorado?'

    Returns:
        DataFrame com o estado atual de todas as entidades.
    """
    with Session(engine) as session:
        records = (
            session.query(DimAtivoStatus)
            .filter(DimAtivoStatus.status_ativo.is_(True))
            .order_by(DimAtivoStatus.score_centralidade.desc())
            .all()
        )
    if not records:
        return pd.DataFrame()

    return pd.DataFrame([{
        "nome_entidade": r.nome_entidade,
        "sentimento": r.sentimento,
        "topico_lda": r.topico_lda,
        "score_centralidade": r.score_centralidade,
        "data_inicio": r.data_inicio,
    } for r in records])


def query_entity_history(engine, nome_entidade: str) -> pd.DataFrame:
    """Retorno o histórico completo de versões de uma entidade.

    Permite a análise de 'time travel': ver como o sentimento de uma
    entidade evoluiu ao longo do tempo — essencial para backtesting.

    Args:
        engine:        Engine SQLAlchemy.
        nome_entidade: Nome da entidade a consultar.

    Returns:
        DataFrame com todas as versões históricas, ordenadas por data.
    """
    with Session(engine) as session:
        records = (
            session.query(DimAtivoStatus)
            .filter(DimAtivoStatus.nome_entidade == nome_entidade)
            .order_by(DimAtivoStatus.data_inicio)
            .all()
        )
    if not records:
        return pd.DataFrame()

    return pd.DataFrame([{
        "id_versao": r.id_versao,
        "sentimento": r.sentimento,
        "topico_lda": r.topico_lda,
        "score_centralidade": r.score_centralidade,
        "data_inicio": r.data_inicio,
        "data_fim": r.data_fim,
        "status_ativo": r.status_ativo,
    } for r in records])


def plot_sentiment_timeline(
    engine,
    nome_entidade: str,
    output_path: Optional[Path] = None,
) -> None:
    """Ploto a linha do tempo de sentimento de uma entidade.

    Visualiza o valor do SCD2: mostra quando e como o sentimento mudou,
    evidenciando o rastreamento histórico para a equipe de estratégia.

    Args:
        engine:        Engine SQLAlchemy.
        nome_entidade: Nome da entidade a visualizar.
        output_path:   Salva PNG se fornecido.
    """
    history = query_entity_history(engine, nome_entidade)
    if history.empty:
        _log.warning("Nenhum histórico para '%s'.", nome_entidade)
        return

    color_map = {"positive": "#4575b4", "neutral": "#fee090", "negative": "#d73027"}
    fig, ax = plt.subplots(figsize=(10, 3))

    for _, row in history.iterrows():
        start = pd.to_datetime(row["data_inicio"])
        end = pd.to_datetime(row["data_fim"]) if row["data_fim"] else pd.Timestamp.today()
        color = color_map.get(row["sentimento"], "#aaaaaa")
        ax.barh(0, (end - start).days, left=start, height=0.4, color=color, alpha=0.8,
                label=row["sentimento"])

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for s, c in color_map.items()]
    ax.legend(handles, list(color_map.keys()), loc="upper right")
    ax.set_title(f"Histórico SCD2 — {nome_entidade}")
    ax.set_yticks([])
    ax.set_xlabel("Data")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        _log.info("Timeline SCD2 salva em '%s'.", output_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("FinNLP — Smoke Test: scd2_manager.py")
    print("=" * 60)

    _engine = get_engine(Path("/tmp/finnlp_test.sqlite"))
    create_schema(_engine)

    _entities = pd.DataFrame([
        {"entidade": "Apple Inc", "sentimento": "positive", "topico_lda": 2, "centralidade": 0.45},
        {"entidade": "Goldman Sachs", "sentimento": "neutral", "topico_lda": 0, "centralidade": 0.38},
        {"entidade": "Tesla Inc", "sentimento": "negative", "topico_lda": 1, "centralidade": 0.22},
    ])

    print("\n--- 1ª execução (INSERT) ---")
    summary = run_scd2_batch(_engine, _entities, reference_date=date(2026, 5, 1))
    print(summary)

    print("\n--- 2ª execução (Apple muda para negative) ---")
    _entities_v2 = _entities.copy()
    _entities_v2.loc[0, "sentimento"] = "negative"
    summary2 = run_scd2_batch(_engine, _entities_v2, reference_date=date(2026, 5, 30))
    print(summary2)

    print("\n--- Status atual ---")
    print(query_current_status(_engine).to_string(index=False))

    print("\n--- Histórico Apple Inc ---")
    print(query_entity_history(_engine, "Apple Inc").to_string(index=False))

    import os
    os.remove("/tmp/finnlp_test.sqlite")
    print("\n✅ Smoke test SCD2 concluído.")
    sys.exit(0)
