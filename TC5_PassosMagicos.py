"""
app.py — Dashboard Passos Mágicos
Datathon FIAP — Fase 5 Tech Challenge

Execute com:
    streamlit run app.py

Requisitos:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Dashboard",
    page_icon="✨",
    layout="wide",
)

# ─────────────────────────────────────────────
# Constantes de paleta
# ─────────────────────────────────────────────
CORES = {
    "primaria":   "#1E4D8C",
    "secundaria": "#F4831F",
    "verde":      "#27AE60",
    "vermelho":   "#E74C3C",
    "cinza":      "#7F8C8D",
    "amarelo":    "#F1C40F",
}
AZUL    = CORES["primaria"]
LARANJA = CORES["secundaria"]
VERDE   = CORES["verde"]
VERMELHO= CORES["vermelho"]
CINZA   = CORES["cinza"]
ROXO    = "#8E44AD"

PEDRAS_ORDEM = ["Quartzo", "Ágata", "Ametista", "Topázio"]
PEDRAS_CORES = ["#7F8C8D", "#1ABC9C", "#8E44AD", "#F1C40F"]

CORES_IAN = {
    "Em Fase":            "#27AE60",
    "Defasagem Moderada": "#F1C40F",
    "Defasagem Severa":   "#E74C3C",
}
ORDEM_IAN = ["Defasagem Severa", "Defasagem Moderada", "Em Fase"]

plt.rcParams.update({
    "figure.dpi":      110,
    "axes.titlesize":  12,
    "axes.labelsize":  10,
    "font.family":     "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

indicadores = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]

# ─────────────────────────────────────────────
# Carregamento e limpeza dos dados (cacheado)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando e limpando os dados...")
def carregar_dados(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    import io
    df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=";")

    df = df_raw.copy()

    # Converter numéricos de 2020
    cols_num_2020 = [
        "IDADE_ALUNO_2020", "ANOS_PM_2020", "INDE_2020",
        "IAA_2020", "IEG_2020", "IPS_2020", "IDA_2020",
        "IPP_2020", "IPV_2020", "IAN_2020",
    ]
    for col in cols_num_2020:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Limpar '#NULO!'
    df["PEDRA_2021"] = df["PEDRA_2021"].replace("#NULO!", np.nan)

    # Booleanos
    cols_bool = [
        "PONTO_VIRADA_2020", "PONTO_VIRADA_2021", "PONTO_VIRADA_2022",
        "BOLSISTA_2022", "INDICADO_BOLSA_2022",
    ]
    mapa_bool = {"Sim": 1, "sim": 1, "SIM": 1, "Não": 0, "não": 0, "NÃO": 0, "Nao": 0}
    for col in cols_bool:
        if col in df.columns:
            df[col] = df[col].map(mapa_bool)

    # Fase/turma 2020
    df["FASE_2020"]  = df["FASE_TURMA_2020"].str.extract(r"^(\d+)").astype(float)
    df["TURMA_2020"] = df["FASE_TURMA_2020"].str.extract(r"^\d+([A-Za-z]+)")

    # Pedra como categoria
    pedra_cat = pd.CategoricalDtype(categories=PEDRAS_ORDEM, ordered=True)
    for col in ["PEDRA_2020", "PEDRA_2021", "PEDRA_2022"]:
        if col in df.columns:
            df[col] = df[col].astype(pedra_cat)

    # Conversão numérica de todos os indicadores
    for ind in indicadores:
        for ano in ["2020", "2021", "2022"]:
            col = f"{ind}_{ano}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering
    df["DELTA_INDE_2021_2020"] = df["INDE_2021"] - df["INDE_2020"]
    df["DELTA_INDE_2022_2021"] = df["INDE_2022"] - df["INDE_2021"]
    df["DELTA_INDE_2022_2020"] = df["INDE_2022"] - df["INDE_2020"]

    pedra_num = {"Quartzo": 1, "Ágata": 2, "Ametista": 3, "Topázio": 4}
    for ano in ["2020", "2021", "2022"]:
        df[f"PEDRA_NUM_{ano}"] = df[f"PEDRA_{ano}"].map(pedra_num)

    df["MELHORA_PEDRA_2021"] = (df["PEDRA_NUM_2021"] > df["PEDRA_NUM_2020"]).astype(float)
    df["MELHORA_PEDRA_2022"] = (df["PEDRA_NUM_2022"] > df["PEDRA_NUM_2021"]).astype(float)

    df["EM_RISCO_2022"] = (
        (df["INDE_2022"] < 6.187) | (df["IAN_2022"] < 6.187)
    ).astype(float)

    # Fase/ano para Q2
    df["FASE_2021"] = pd.to_numeric(df.get("FASE_2021"), errors="coerce")
    df["FASE_2022"] = pd.to_numeric(df.get("FASE_2022"), errors="coerce")

    # IAN categoria
    mapa_ian = {10.0: "Em Fase", 5.0: "Defasagem Moderada", 2.5: "Defasagem Severa"}
    for ano in ["2020", "2021", "2022"]:
        df[f"IAN_CAT_{ano}"] = df[f"IAN_{ano}"].map(mapa_ian)

    # DataFrame long
    indicadores_base = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE"]
    frames = []
    for ano in ["2020", "2021", "2022"]:
        cols_ano = {f"{ind}_{ano}": ind for ind in indicadores_base if f"{ind}_{ano}" in df.columns}
        cols_ano[f"PEDRA_{ano}"]         = "PEDRA"
        cols_ano[f"PONTO_VIRADA_{ano}"]  = "PONTO_VIRADA"
        temp = df[["NOME"] + list(cols_ano.keys())].copy()
        temp = temp.rename(columns=cols_ano)
        temp["ANO"] = int(ano)
        frames.append(temp)
    df_long = pd.concat(frames, ignore_index=True)
    df_long = df_long.dropna(subset=["INDE"])

    return df, df_long


# ─────────────────────────────────────────────
# Modelo preditivo (cacheado por df)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Treinando modelo preditivo...")
def treinar_modelo(df_hash, _df):
    for ind in indicadores:
        _df[f"DELTA_{ind}"] = _df[f"{ind}_2021"] - _df[f"{ind}_2020"]

    FEAT_BASE  = [f"{ind}_{ano}" for ind in indicadores for ano in ["2020", "2021"]]
    FEAT_DELTA = [f"DELTA_{ind}" for ind in indicadores]
    FEAT_EXTRA = ["PONTO_VIRADA_2020", "PONTO_VIRADA_2021"]
    FEAT_COLS  = FEAT_BASE + FEAT_DELTA + FEAT_EXTRA

    df_model = _df[_df["EM_RISCO_2022"].notna()].copy()
    X_raw = df_model[FEAT_COLS].values
    y     = df_model["EM_RISCO_2022"].values

    imputer = SimpleImputer(strategy="median")
    X       = imputer.fit_transform(X_raw)

    modelo = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42
    )
    modelo.fit(X, y)

    # Probabilidades para todos
    mask_consulta    = _df[FEAT_BASE].notna().any(axis=1)
    df_consulta      = _df.loc[mask_consulta, ["NOME"] + FEAT_COLS].copy()
    X_consulta       = imputer.transform(df_consulta[FEAT_COLS].values)
    df_consulta["PROB_RISCO"]    = modelo.predict_proba(X_consulta)[:, 1]
    df_consulta["CLASSIFICACAO"] = df_consulta["PROB_RISCO"].apply(
        lambda p: "ALTO RISCO"     if p >= 0.70 else
                  "RISCO MODERADO" if p >= 0.40 else
                  "BAIXO RISCO"
    )

    return modelo, imputer, FEAT_COLS, df_consulta


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:{CORES['primaria']};padding:20px 30px;border-radius:10px;margin-bottom:20px">
        <h1 style="color:white;margin:0">✨ Passos Mágicos — Dashboard Analítico</h1>
        <p style="color:#cde;margin:4px 0 0">Datathon FIAP · Fase 5 Tech Challenge · 2020–2022</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Upload do CSV
# ─────────────────────────────────────────────
st.sidebar.header("📂 Dados")
uploaded = st.sidebar.file_uploader(
    "Faça upload do arquivo CSV",
    type=["csv"],
    help="Arquivo: PEDE_PASSOS_DATASET_FIAP.csv (separador ;)",
)

if uploaded is None:
    st.info(
        "👆 Faça o upload do arquivo **PEDE_PASSOS_DATASET_FIAP.csv** na barra lateral para começar.",
        icon="📎",
    )
    st.stop()

df, df_long = carregar_dados(uploaded.read())

# KPIs no sidebar
st.sidebar.markdown("---")
st.sidebar.metric("Alunos únicos",  df["NOME"].nunique())
st.sidebar.metric("Colunas (wide)", df.shape[1])
st.sidebar.metric("Registros long", len(df_long))

# ─────────────────────────────────────────────
# Abas principais
# ─────────────────────────────────────────────
abas = st.tabs([
    "📊 EDA",
    "Q1 — Defasagem (IAN)",
    "Q2 — Desempenho (IDA)",
    "Q3 — Engajamento (IEG)",
    "Q4 — Autoavaliação (IAA)",
    "Q5 — Bem-estar (IPS)",
    "Q6 — Avaliação Pedagógica (IPP)",
    "Q7 — Ponto de Virada (IPV)",
    "Q8 — Nota Global (INDE)",
    "Q9 — Modelo Preditivo",
    "Q10 — Efetividade",
    "Q11 — Ponto de Virada & INDE",
])

# ══════════════════════════════════════════════
# ABA 0 — EDA
# ══════════════════════════════════════════════
with abas[0]:
    st.subheader("Exploração de Dados")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Alunos (2020)", int(df["INDE_2020"].notna().sum()))
    with c2:
        st.metric("Alunos (2021)", int(df["INDE_2021"].notna().sum()))
    with c3:
        st.metric("Alunos (2022)", int(df["INDE_2022"].notna().sum()))

    # Distribuição do INDE por ano
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, ano in zip(axes, [2020, 2021, 2022]):
        dados = df_long[df_long["ANO"] == ano]["INDE"].dropna()
        ax.hist(dados, bins=30, color=AZUL, edgecolor="white", alpha=0.85)
        ax.axvline(dados.mean(), color=LARANJA, linestyle="--", linewidth=1.5,
                   label=f"Média: {dados.mean():.2f}")
        ax.set_title(f"INDE {ano}  (n={len(dados)})")
        ax.set_xlabel("INDE")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frequência")
    plt.suptitle("Distribuição do INDE por ano", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    # Pedras por ano
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        pedra_ano = df_long.groupby(["ANO", "PEDRA"], observed=True).size().unstack(fill_value=0)
        pedra_ano = pedra_ano.reindex(columns=PEDRAS_ORDEM, fill_value=0)
        pedra_pct = pedra_ano.div(pedra_ano.sum(axis=1), axis=0) * 100
        pedra_pct.plot(kind="bar", stacked=True, color=PEDRAS_CORES, ax=ax,
                       edgecolor="white", width=0.5)
        ax.set_title("Distribuição de Pedras por ano (%)")
        ax.set_xlabel("Ano"); ax.set_ylabel("%")
        ax.set_xticklabels(pedra_pct.index, rotation=0)
        ax.legend(title="Pedra", bbox_to_anchor=(1.01, 1), loc="upper left")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f%%", label_type="center",
                         fontsize=8, color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Boxplot indicadores 2022
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        inds_2022 = ["IAN_2022","IDA_2022","IEG_2022","IAA_2022","IPS_2022","IPP_2022","IPV_2022"]
        dados_box = df[[c for c in inds_2022 if c in df.columns]].dropna(how="all")
        dados_box.columns = [c.replace("_2022", "") for c in dados_box.columns]
        bp = ax.boxplot(
            [dados_box[c].dropna() for c in dados_box.columns],
            labels=dados_box.columns,
            patch_artist=True,
            medianprops=dict(color=LARANJA, linewidth=2),
        )
        cores_box = plt.cm.Blues(np.linspace(0.4, 0.8, len(dados_box.columns)))
        for patch, cor in zip(bp["boxes"], cores_box):
            patch.set_facecolor(cor)
        ax.set_title("Distribuição dos indicadores — 2022")
        ax.set_ylabel("Nota (0–10)"); ax.set_ylim(0, 10.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Correlação 2022
    st.markdown("#### Matriz de Correlação — 2022")
    fig, ax = plt.subplots(figsize=(8, 6))
    cols_corr = ["IAN_2022","IDA_2022","IEG_2022","IAA_2022","IPS_2022","IPP_2022","IPV_2022","INDE_2022"]
    corr = df[[c for c in cols_corr if c in df.columns]].corr()
    corr.columns = [c.replace("_2022","") for c in corr.columns]
    corr.index   = [c.replace("_2022","") for c in corr.index]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlação entre indicadores — 2022", pad=12)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Ponto de virada por ano
    st.markdown("#### Ponto de Virada por ano")
    pv_anos = {}
    for ano in ["2020","2021","2022"]:
        col = f"PONTO_VIRADA_{ano}"
        if col in df.columns:
            pv_anos[ano] = df[col].value_counts(normalize=True).get(1, 0) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pv_anos.keys(), pv_anos.values(), color=VERDE, edgecolor="white", width=0.4)
    ax.set_title("% de alunos que atingiram o Ponto de Virada")
    ax.set_ylabel("%"); ax.set_ylim(0, 100)
    for i, (ano, val) in enumerate(pv_anos.items()):
        ax.text(i, val + 1.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 1 — Q1 Defasagem IAN
# ══════════════════════════════════════════════
with abas[1]:
    st.subheader("Q1 — Perfil de Defasagem dos Alunos (IAN)")
    st.caption("IAN 10 = Em Fase · IAN 5 = Defasagem Moderada · IAN 2,5 = Defasagem Severa")

    anos = ["2020","2021","2022"]
    frames_ian = []
    for ano in anos:
        contagem = df[f"IAN_CAT_{ano}"].value_counts().rename(ano)
        frames_ian.append(contagem)

    tabela = pd.DataFrame(frames_ian).fillna(0).astype(int)
    tabela = tabela.reindex(columns=ORDEM_IAN, fill_value=0)
    tabela_pct = tabela.div(tabela.sum(axis=1), axis=0) * 100

    # Barras empilhadas
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(anos))
    for categoria in ORDEM_IAN:
        valores = tabela[categoria].values
        barras  = ax.bar(anos, valores, bottom=bottom,
                         color=CORES_IAN[categoria], label=categoria,
                         edgecolor="white", linewidth=0.8, width=0.45)
        for i, (barra, n) in enumerate(zip(barras, valores)):
            pct = tabela_pct[categoria].iloc[i]
            if pct >= 4:
                ax.text(barra.get_x() + barra.get_width() / 2,
                        bottom[i] + n / 2,
                        f"{n}\n({pct:.0f}%)",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += valores

    for i, total in enumerate(tabela.sum(axis=1)):
        ax.text(i, total + 8, f"n={total}", ha="center",
                fontsize=9, color=CINZA, fontweight="bold")

    ax.set_title("Perfil de Defasagem dos Alunos (IAN) por Ano\nPassos Mágicos — 2020, 2021 e 2022",
                 fontsize=13, fontweight="bold", pad=16, color=AZUL)
    ax.set_xlabel("Ano"); ax.set_ylabel("Número de Alunos")
    ax.set_ylim(0, tabela.sum(axis=1).max() * 1.12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Nível de Defasagem", bbox_to_anchor=(1.01, 1),
              loc="upper left", frameon=False, fontsize=10)
    fig.text(0.5, -0.02,
             "IAN 10 = Em Fase  |  IAN 5 = Defasagem Moderada  |  IAN 2,5 = Defasagem Severa",
             ha="center", fontsize=8, color=CINZA, style="italic")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Pizza por ano
    LABELS_IAN = {"Em Fase": "Em fase (10)", "Defasagem Moderada": "Moderado (5)",
                  "Defasagem Severa": "Severo (2,5)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, ano in zip(axes, anos):
        dados   = tabela.loc[ano].reindex(ORDEM_IAN)
        total   = dados.sum()
        cores   = [CORES_IAN[c] for c in ORDEM_IAN]
        explode = [0.04 if v == dados.max() else 0 for v in dados]
        labels_ext = [LABELS_IAN[c] for c in ORDEM_IAN]
        wedges, texts, autotexts = ax.pie(
            dados, labels=labels_ext, colors=cores,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
            startangle=90, explode=explode,
            pctdistance=0.68, labeldistance=1.18,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for text in texts:
            text.set_fontsize(9); text.set_color("#333333")
        for autotext in autotexts:
            autotext.set_fontsize(10); autotext.set_fontweight("bold"); autotext.set_color("white")
        ax.set_title(f"IAN {ano}  ({total} alunos avaliados)",
                     fontsize=11, fontweight="bold", color=AZUL, pad=14)
    fig.suptitle("Q1 — Perfil de defasagem (IAN) por ano",
                 fontsize=14, fontweight="bold", y=1.01, color="#1a1a1a")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 2 — Q2 IDA
# ══════════════════════════════════════════════
with abas[2]:
    st.subheader("Q2 — Desempenho Acadêmico (IDA): evolução temporal e por fase")

    CORES_TEND = {"Melhora": VERDE, "Estagnado": CORES["amarelo"], "Queda": VERMELHO}
    ANOS = ["2020","2021","2022"]

    for ano in ANOS:
        df[f"IDA_{ano}"] = pd.to_numeric(df.get(f"IDA_{ano}"), errors="coerce")

    medias_ano = {ano: df[f"IDA_{ano}"].mean() for ano in ANOS}
    ns_ano     = {ano: df[f"IDA_{ano}"].notna().sum() for ano in ANOS}

    THRESH = 0.5
    def classificar_delta(delta):
        if delta > THRESH:   return "Melhora"
        elif delta < -THRESH: return "Queda"
        else:                return "Estagnado"

    deltas    = {"2020->2021": medias_ano["2021"] - medias_ano["2020"],
                 "2021->2022": medias_ano["2022"] - medias_ano["2021"]}
    tendencias = {k: classificar_delta(v) for k, v in deltas.items()}

    # Heatmap fase × ano
    frames_h = []
    for ano in ANOS:
        fase_col = f"FASE_{ano}"
        ida_col  = f"IDA_{ano}"
        if fase_col in df.columns:
            temp = df[[fase_col, ida_col]].dropna()
            grp  = temp.groupby(fase_col)[ida_col].mean().rename(ano)
            frames_h.append(grp)

    heatmap_df = pd.DataFrame(frames_h).T
    heatmap_df.index = heatmap_df.index.astype(int)
    heatmap_df = heatmap_df.sort_index()
    heatmap_df.index = [f"Fase {i}" for i in heatmap_df.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("Q2 — Desempenho Acadêmico (IDA): evolução temporal e por fase",
                 fontsize=13, fontweight="bold", y=1.02, color="#1a1a1a")

    x_vals = [0, 1, 2]
    y_vals = [medias_ano[a] for a in ANOS]

    ax1.fill_between(x_vals, y_vals, alpha=0.08, color=AZUL)
    ax1.plot(x_vals, y_vals, color=AZUL, linewidth=2.5,
             marker="o", markersize=9, zorder=5)
    for x, y, ano in zip(x_vals, y_vals, ANOS):
        ax1.text(x, y + 0.18, f"{y:.2f}", ha="center", fontsize=10,
                 fontweight="bold", color=AZUL)
        ax1.text(x, y - 0.32, f"n={ns_ano[ano]}", ha="center",
                 fontsize=8, color=CINZA)

    pares = [(0, 1, "2020->2021"), (1, 2, "2021->2022")]
    for x0, x1, chave in pares:
        xm   = (x0 + x1) / 2
        ym   = (y_vals[x0] + y_vals[x1]) / 2
        tend = tendencias[chave]
        delt = deltas[chave]
        cor  = CORES_TEND[tend]
        seta = "↑" if tend == "Melhora" else ("↓" if tend == "Queda" else "→")
        ax1.annotate(f"{seta} {tend}\n({delt:+.2f})", xy=(xm, ym),
                     fontsize=8.5, ha="center", va="center",
                     color="white", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=cor,
                               edgecolor="none", alpha=0.9))

    media_geral = float(np.mean(y_vals))
    ax1.axhline(media_geral, color=CINZA, linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.text(2.08, media_geral + 0.05, f"Média geral\n{media_geral:.2f}",
             fontsize=7.5, color=CINZA)
    ax1.set_title("IDA Médio por Ano", fontsize=11, fontweight="bold",
                  color=AZUL, pad=10)
    ax1.set_xticks(x_vals); ax1.set_xticklabels(ANOS, fontsize=10)
    ax1.set_ylabel("IDA Médio (0–10)", fontsize=10)
    ax1.set_ylim(min(y_vals) - 1.2, max(y_vals) + 0.9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    leg_patches = [mpatches.Patch(color=v, label=k) for k, v in CORES_TEND.items()]
    ax1.legend(handles=leg_patches, title="Tendência",
               loc="lower right", frameon=False, fontsize=8)

    if not heatmap_df.empty:
        sns.heatmap(heatmap_df, ax=ax2, cmap=sns.color_palette("RdYlGn", as_cmap=True),
                    vmin=0, vmax=10, annot=True, fmt=".2f",
                    annot_kws={"size": 9, "weight": "bold"},
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "IDA Médio", "shrink": 0.8})
        ax2.set_title("IDA Médio por Fase e Ano", fontsize=11, fontweight="bold",
                      color=AZUL, pad=10)
        ax2.set_xlabel("Ano"); ax2.set_ylabel("Fase")
        ax2.tick_params(axis="x", rotation=0); ax2.tick_params(axis="y", rotation=0)

    fig.text(0.5, -0.04,
             f"Threshold: Melhora = delta > +{THRESH}  |  Estagnado = entre ±{THRESH}  |  Queda = delta < -{THRESH}",
             ha="center", fontsize=8, color=CINZA, style="italic")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 3 — Q3 IEG
# ══════════════════════════════════════════════
with abas[2]:
    pass  # já preenchida acima

with abas[3]:
    st.subheader("Q3 — Engajamento vs Desempenho e Ponto de Virada (IEG)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (ycol, ylabel, cor) in zip(axes, [
        ("IDA_2022", "IDA (Desempenho)", AZUL),
        ("IPV_2022", "IPV (Ponto de Virada)", ROXO),
    ]):
        d = df[["IEG_2022", ycol]].dropna()
        r = d.corr().iloc[0, 1]
        sns.regplot(data=d, x="IEG_2022", y=ycol,
                    scatter_kws=dict(alpha=0.3, s=25, color=cor),
                    line_kws=dict(color="black", linewidth=2), ax=ax)
        ax.set_title(f"IEG × {ylabel}\nr = {r:.3f}", fontweight="bold")
        ax.set_xlabel("IEG (Engajamento)"); ax.set_ylabel(ylabel)
        ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))
    plt.suptitle("Q3 — Engajamento vs desempenho e ponto de virada (2022)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Comparativo PV
    pv = df.groupby("PONTO_VIRADA_2022")[["IEG_2022","IDA_2022","IPV_2022"]].mean().round(2)
    if len(pv) >= 2:
        pv.index = ["Não atingiu", "Atingiu PV"]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(3); w = 0.32
        b1 = ax.bar(x - w/2, pv.loc["Não atingiu"], w, label="Não atingiu PV",
                    color=CINZA, edgecolor="white")
        b2 = ax.bar(x + w/2, pv.loc["Atingiu PV"],  w, label="Atingiu PV",
                    color=VERDE, edgecolor="white")
        ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(["IEG","IDA","IPV"], fontsize=12)
        ax.set_ylabel("Nota média"); ax.set_ylim(0, 11)
        ax.set_title("Q3 — Perfil médio: atingiu vs não atingiu o Ponto de Virada (2022)",
                     fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 4 — Q4 IAA
# ══════════════════════════════════════════════
with abas[4]:
    st.subheader("Q4 — Coerência da Autoavaliação (IAA)")

    d = df[["IAA_2022","IDA_2022","IEG_2022"]].dropna().copy()
    d["diff_ida"] = d["IAA_2022"] - d["IDA_2022"]
    d["diff_ieg"] = d["IAA_2022"] - d["IEG_2022"]
    r_ida = d[["IAA_2022","IDA_2022"]].corr().iloc[0, 1]
    r_ieg = d[["IAA_2022","IEG_2022"]].corr().iloc[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (col_y, label_y, diff_col, r) in zip(axes, [
        ("IDA_2022", "IDA (Desempenho real)", "diff_ida", r_ida),
        ("IEG_2022", "IEG (Engajamento)",     "diff_ieg", r_ieg),
    ]):
        sc = ax.scatter(d["IAA_2022"], d[col_y], c=d[diff_col],
                        cmap="RdYlGn_r", alpha=0.5, s=30, vmin=-8, vmax=8)
        ax.plot([0,10],[0,10], "--", color="gray", linewidth=1.5, label="Coerência perfeita")
        plt.colorbar(sc, ax=ax, label=f"IAA−{label_y.split(' ')[0]} (+ = superestima)")
        ax.set_xlabel("IAA (Autoavaliação)"); ax.set_ylabel(label_y)
        ax.set_title(f"IAA vs {label_y.split(' ')[0]} — 2022\nr = {r:.3f}", fontweight="bold")
        pct_s   = (d[diff_col] > 0).mean() * 100
        pct_sub = (d[diff_col] < 0).mean() * 100
        ax.text(0.05, 0.93, f"Superestima: {pct_s:.1f}%\nSubestima: {pct_sub:.1f}%",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))
        ax.legend(fontsize=8)

    plt.suptitle("Q4 — Coerência da autoavaliação (IAA) com desempenho real (IDA) e engajamento (IEG)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 5 — Q5 IPS
# ══════════════════════════════════════════════
with abas[5]:
    st.subheader("Q5 — Bem-estar Psicossocial (IPS) como preditor de risco")

    painel = df[["NOME","IPS_2021","IDA_2021","IDA_2022","IEG_2021","IEG_2022"]].dropna()
    painel["queda_ida"] = painel["IDA_2022"] < painel["IDA_2021"]
    painel["queda_ieg"] = painel["IEG_2022"] < painel["IEG_2021"]
    painel["ips_grupo"] = pd.cut(painel["IPS_2021"],
        bins=[0, 5, 7, 10],
        labels=["IPS Baixo\n(0 a 5)", "IPS Médio\n(5 a 7)", "IPS Alto\n(7 a 10)"])

    CORES_IPS = ["#C0392B", "#E67E22", "#1A5276"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (col_queda, rotulo, titulo) in zip(axes, [
        ("queda_ida", "IDA", "Queda no Desempenho Acadêmico (IDA)"),
        ("queda_ieg", "IEG", "Queda no Engajamento (IEG)"),
    ]):
        taxa = painel.groupby("ips_grupo", observed=True)[col_queda].mean() * 100
        ns   = painel.groupby("ips_grupo", observed=True)[col_queda].count()
        ax.set_facecolor("#FAFAFA")
        bars = ax.bar(taxa.index, taxa.values, color=CORES_IPS,
                      edgecolor="white", width=0.5, linewidth=1.5)
        for bar, val, n in zip(bars, taxa.values, ns.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                    f"{val:.1f}%\n({int(n)} alunos)",
                    ha="center", fontsize=9, fontweight="bold", color="#2C2C2C")
        ax.set_ylabel("% de alunos com queda no ano seguinte", fontsize=11)
        ax.set_ylim(0, 85)
        ax.set_title(titulo, fontweight="bold", fontsize=12, pad=12)
        ax.set_xlabel("Nível de bem-estar psicossocial em 2021", fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#EEEEEE"); ax.set_axisbelow(True)
        media_geral = taxa.mean()
        ax.axhline(media_geral, color="#888888", linestyle="--",
                   linewidth=1.2, label=f"Média geral: {media_geral:.1f}%")
        ax.legend(fontsize=9, framealpha=0.6)

    plt.suptitle(
        "Alunos com bem-estar psicossocial baixo em 2021 apresentam\n"
        "maior risco de queda de desempenho e engajamento em 2022",
        fontsize=13, fontweight="bold", color="#1A1A1A", y=1.02)
    fig.text(0.5, -0.02,
             "IPS = média das avaliações dos psicólogos (comportamental, emocional, social) · escala 0 a 10",
             ha="center", fontsize=9, color="#888888", style="italic")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 6 — Q6 IPP
# ══════════════════════════════════════════════
with abas[6]:
    st.subheader("Q6 — Avaliação Psicopedagógica (IPP) vs Defasagem (IAN)")

    mapa_ian  = {2.5: "Severa\n(IAN = 2,5)", 5.0: "Moderada\n(IAN = 5)", 10.0: "Em fase\n(IAN = 10)"}
    ordem_ian = ["Severa\n(IAN = 2,5)", "Moderada\n(IAN = 5)", "Em fase\n(IAN = 10)"]
    d = df[["IAN_2022","IPP_2022"]].dropna().copy()
    d["Defasagem"] = d["IAN_2022"].map(mapa_ian)
    d = d.dropna(subset=["Defasagem"])

    medias = d.groupby("Defasagem", observed=True)["IPP_2022"].mean().reindex(ordem_ian)
    ns     = d.groupby("Defasagem", observed=True)["IPP_2022"].count().reindex(ordem_ian)
    corr_v = d[["IAN_2022","IPP_2022"]].corr().iloc[0, 1]

    CORES_Q6 = ["#C0392B", "#E67E22", "#1A5276"]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    bars = ax.bar(ordem_ian, medias.values, color=CORES_Q6, edgecolor="white", width=0.5)
    for bar, val, n in zip(bars, medias.values, ns.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                f"{val:.2f}\n({int(n)} alunos)", ha="center",
                fontsize=10, fontweight="bold", color="#2C2C2C")
    ax.plot(range(len(ordem_ian)), medias.values, "o-", color="#2C2C2C",
            linewidth=2.2, markersize=8, zorder=5, label=f"Tendência (r = {corr_v:.3f})")
    media_geral = d["IPP_2022"].mean()
    ax.axhline(media_geral, color="#888888", linestyle="--",
               linewidth=1.2, label=f"IPP médio geral: {media_geral:.2f}")
    ax.set_ylabel("IPP médio"); ax.set_xlabel("Nível de defasagem (IAN)")
    ax.set_title("Q6 — As avaliações psicopedagógicas (IPP) confirmam a defasagem (IAN)?",
                 fontweight="bold", fontsize=11, pad=12)
    ax.set_ylim(0, 10)
    ax.set_xticklabels([g.replace("\n", " ") for g in ordem_ian], fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#EEEEEE"); ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.6)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.metric("Correlação IPP × IAN", f"{corr_v:.3f}")


# ══════════════════════════════════════════════
# ABA 7 — Q7 IPV
# ══════════════════════════════════════════════
with abas[7]:
    st.subheader("Q7 — Indicadores que mais influenciam o Ponto de Virada (IPV)")

    cols_2022 = ["IAN_2022","IDA_2022","IEG_2022","IAA_2022","IPS_2022","IPP_2022"]
    corrs_ipv = df[cols_2022 + ["IPV_2022"]].corr()["IPV_2022"].drop("IPV_2022").sort_values()
    nomes_ipv = [c.replace("_2022","") for c in corrs_ipv.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    cores_bar = ["#C0392B" if v > 0 else "#1A5276" for v in corrs_ipv.values]
    bars = ax.barh(nomes_ipv, corrs_ipv.values, color=cores_bar, edgecolor="white", height=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10, fontweight="bold", color="#2C2C2C")
    ax.axvline(0, color="#2C2C2C", linewidth=1)
    ax.set_xlabel("Correlação de Pearson com IPV", fontsize=11, color="#555555")
    ax.set_title("Q7 — Quais indicadores mais influenciam o Ponto de Virada? (2022)",
                 fontweight="bold", fontsize=12, color="#1A1A1A")
    ax.set_xlim(-0.1, 0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#EEEEEE"); ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 8 — Q8 INDE
# ══════════════════════════════════════════════
with abas[8]:
    st.subheader("Q8 — Indicadores com maior influência na Nota Global (INDE)")

    cols_ind = ["IAN_2022","IDA_2022","IEG_2022","IAA_2022","IPS_2022","IPP_2022","IPV_2022"]
    corrs_inde = df[cols_ind + ["INDE_2022"]].corr()["INDE_2022"].drop("INDE_2022").sort_values(ascending=True)
    pesos = {"IAN": 0.1, "IDA": 0.2, "IEG": 0.2, "IAA": 0.1,
             "IPS": 0.1, "IPP": 0.1, "IPV": 0.2}
    nomes_inde = [c.replace("_2022","") for c in corrs_inde.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    cores_bar = ["#1A5276" if v >= 0.5 else "#4A7FB5" if v >= 0.3 else "#AEB6BF"
                 for v in corrs_inde.values]
    bars = ax.barh(nomes_inde, corrs_inde.values, color=cores_bar, edgecolor="white", height=0.5)
    for bar, val, nome in zip(bars, corrs_inde.values, nomes_inde):
        peso = pesos.get(nome, 0)
        ax.text(1.02, bar.get_y() + bar.get_height()/2,
                f"Correlação: {val:.3f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#1A1A1A")
        ax.text(1.02, bar.get_y() - 0.05,
                f"Peso na fórmula: {peso}", va="center", ha="left",
                fontsize=9, color="#555555")
    ax.axvline(0.5, color="#1A5276", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(0.3, color="#4A7FB5", linestyle="--", linewidth=1.2, alpha=0.7)
    leg1 = mpatches.Patch(color="#1A5276", label="Correlação ≥ 0,50 — relação forte")
    leg2 = mpatches.Patch(color="#4A7FB5", label="Correlação ≥ 0,30 — relação moderada")
    leg3 = mpatches.Patch(color="#AEB6BF", label="Correlação < 0,30 — relação fraca")
    ax.legend(handles=[leg1, leg2, leg3], fontsize=9, framealpha=0.8,
              loc="upper left", bbox_to_anchor=(0.01, 0.35))
    ax.set_xlabel("Correlação de Pearson com o INDE", fontsize=10, color="#555555")
    ax.set_title("Q8 — Quais indicadores mais elevam a nota global do aluno (INDE)?",
                 fontweight="bold", fontsize=11, color="#1A1A1A")
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#EEEEEE"); ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 9 — Q9 Modelo Preditivo
# ══════════════════════════════════════════════
with abas[9]:
    st.subheader("Q9 — Modelo Preditivo de Risco de Defasagem")
    st.caption("GradientBoostingClassifier treinado com indicadores de 2020 e 2021 para prever risco em 2022.")

    df_hash = df["NOME"].count()
    modelo, imputer, FEAT_COLS, df_consulta = treinar_modelo(df_hash, df.copy())

    # Feature importance
    fi = pd.Series(modelo.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    top15 = fi.head(15)
    cores_b = [LARANJA if i < 3 else AZUL for i in range(len(top15))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top15.index[::-1], top15.values[::-1],
                   color=cores_b[::-1], edgecolor="white", height=0.65)
    for bar, val in zip(bars, top15.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("Indicadores com maior poder preditivo de risco de defasagem")
    ax.set_xlabel("Importância relativa")
    ax.grid(axis="x")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Consulta por aluno
    st.markdown("---")
    st.markdown("#### 🔍 Consulta por aluno")
    nome_input = st.text_input("Digite o nome do aluno:", placeholder="Ex: ALUNO-36")

    if nome_input:
        resultado = df_consulta[
            df_consulta["NOME"].str.strip().str.upper() == nome_input.strip().upper()
        ]
        if resultado.empty:
            sugestoes = df_consulta.loc[
                df_consulta["NOME"].str.upper().str.contains(nome_input.strip().upper(), regex=False),
                "NOME",
            ].tolist()
            st.warning(f'Aluno "{nome_input}" não encontrado.')
            if sugestoes:
                st.info(f"Você quis dizer? {sugestoes[:5]}")
        else:
            row = resultado.iloc[0]
            prob = row["PROB_RISCO"]
            clas = row["CLASSIFICACAO"]
            cor_clas = {"ALTO RISCO": "🔴", "RISCO MODERADO": "🟡", "BAIXO RISCO": "🟢"}
            c1, c2, c3 = st.columns(3)
            c1.metric("Aluno",        row["NOME"])
            c2.metric("Classificação", f"{cor_clas.get(clas, '')} {clas}")
            c3.metric("P(Risco)",      f"{prob:.1%}")

    # Top riscos
    st.markdown("---")
    st.markdown("#### 🚨 Top 20 alunos com maior risco previsto")
    top20 = df_consulta.nlargest(20, "PROB_RISCO")[["NOME","PROB_RISCO","CLASSIFICACAO"]]
    top20["PROB_RISCO"] = top20["PROB_RISCO"].map(lambda x: f"{x:.1%}")
    st.dataframe(top20.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════
# ABA 10 — Q10 Efetividade
# ══════════════════════════════════════════════
with abas[10]:
    st.subheader("Q10 — Efetividade do Programa: Progressão de Pedra e INDE")

    CORES_PEDRA = {
        "Quartzo": "#A8C4E0", "Ágata": "#7F8C8D",
        "Ametista": "#1E4D8C", "Topázio": "#0A1F3C",
    }
    CORES_MOB = {"Avançou": "#27AE60", "Estagnado": "#F1C40F", "Recuou": "#E74C3C"}

    df10 = df.copy()
    for ano in ["2020","2021","2022"]:
        df10[f"INDE_{ano}"] = pd.to_numeric(df10.get(f"INDE_{ano}"), errors="coerce")

    pedra_cat = pd.CategoricalDtype(categories=PEDRAS_ORDEM, ordered=True)
    pedra_num = {"Quartzo": 1, "Ágata": 2, "Ametista": 3, "Topázio": 4}
    for ano in ["2020","2021","2022"]:
        df10[f"PEDRA_{ano}"] = df10[f"PEDRA_{ano}"].astype(str).str.strip().replace("nan", np.nan).astype(pedra_cat)
        df10[f"PEDRA_NUM_{ano}"] = df10[f"PEDRA_{ano}"].map(pedra_num).astype(float)

    def calcular_mobilidade(dff, ano_a, ano_b):
        mask = dff[f"PEDRA_NUM_{ano_a}"].notna() & dff[f"PEDRA_NUM_{ano_b}"].notna()
        sub  = dff[mask].copy()
        sub["DELTA"] = sub[f"PEDRA_NUM_{ano_b}"].astype(float) - sub[f"PEDRA_NUM_{ano_a}"].astype(float)
        sub["MOB"]   = sub["DELTA"].apply(
            lambda d: "Avançou" if d > 0 else ("Recuou" if d < 0 else "Estagnado"))
        contagem = sub["MOB"].value_counts()
        total    = len(sub)
        return {cat: {"n": int(contagem.get(cat, 0)),
                      "pct": contagem.get(cat, 0) / total * 100}
                for cat in ["Avançou", "Estagnado", "Recuou"]}, total

    mob_20_21, n_20_21 = calcular_mobilidade(df10, "2020", "2021")
    mob_21_22, n_21_22 = calcular_mobilidade(df10, "2021", "2022")

    anos_x = ["2020","2021","2022"]
    dist_pedra = {}
    for ano in anos_x:
        vc    = df10[f"PEDRA_{ano}"].value_counts()
        total = vc.sum()
        dist_pedra[ano] = {p: vc.get(p, 0) / total * 100 for p in PEDRAS_ORDEM}

    inde_pedra = {}
    for ano in anos_x:
        grp = df10.groupby(f"PEDRA_{ano}", observed=True)[f"INDE_{ano}"].mean()
        inde_pedra[ano] = grp.reindex(PEDRAS_ORDEM)

    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_mob  = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_inde = fig.add_subplot(gs[1, :])
    fig.suptitle(
        "Q10 — Efetividade do Programa: Progressão de Pedra e Evolução do INDE\nPassos Mágicos — 2020, 2021 e 2022",
        fontsize=13, fontweight="bold", y=1.01, color="#1a1a1a")

    pares_mob  = ["2020→2021", "2021→2022"]
    mobs_mob   = [mob_20_21, mob_21_22]
    ns_mob     = [n_20_21, n_21_22]
    y_pos = np.arange(len(pares_mob))
    left  = np.zeros(len(pares_mob))
    for cat in ["Avançou", "Estagnado", "Recuou"]:
        vals = [mob[cat]["pct"] for mob in mobs_mob]
        bars = ax_mob.barh(y_pos, vals, left=left, color=CORES_MOB[cat],
                           label=cat, edgecolor="white", height=0.45)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 5:
                ax_mob.text(left[i] + val / 2, bar.get_y() + bar.get_height() / 2,
                            f"{val:.0f}%", ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
        left += np.array(vals)
    ax_mob.set_yticks(y_pos)
    ax_mob.set_yticklabels([f"{p}\n(n={n})" for p, n in zip(pares_mob, ns_mob)], fontsize=10)
    ax_mob.set_xlabel("% de alunos"); ax_mob.set_xlim(0, 105)
    ax_mob.set_title("Mobilidade entre Pedras", fontweight="bold", color=AZUL, pad=10)
    ax_mob.legend(frameon=False, fontsize=9, loc="lower right")
    ax_mob.grid(axis="x", linestyle="--", alpha=0.2)

    bottom = np.zeros(3)
    for pedra in PEDRAS_ORDEM:
        vals = [dist_pedra[ano][pedra] for ano in anos_x]
        bars = ax_dist.bar(anos_x, vals, bottom=bottom, color=CORES_PEDRA[pedra],
                           label=pedra, edgecolor="white", width=0.45)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 4:
                ax_dist.text(bar.get_x() + bar.get_width() / 2, bottom[i] + val / 2,
                             f"{val:.0f}%", ha="center", va="center",
                             fontsize=8, fontweight="bold", color="white")
        bottom += np.array(vals)
    ax_dist.set_title("Distribuição de Pedras por Ano", fontweight="bold", color=AZUL, pad=10)
    ax_dist.set_ylabel("% de alunos"); ax_dist.set_ylim(0, 115)
    ax_dist.grid(axis="y", linestyle="--", alpha=0.2)
    ax_dist.legend(title="Pedra", frameon=False, fontsize=9,
                   bbox_to_anchor=(1.01, 1), loc="upper left")

    x_vals_i = [0, 1, 2]
    markers  = ["o", "s", "^", "D"]
    for pedra, marker in zip(PEDRAS_ORDEM, markers):
        y_vals_i = [inde_pedra[ano][pedra] for ano in anos_x]
        y_plot   = [y if not np.isnan(y) else None for y in y_vals_i]
        ax_inde.plot(x_vals_i, y_plot, color=CORES_PEDRA[pedra], linewidth=2.2,
                     marker=marker, markersize=8, label=pedra, zorder=5)
        for x, y in zip(x_vals_i, y_plot):
            if y is not None:
                ax_inde.text(x, y + 0.06, f"{y:.2f}", ha="center", fontsize=8,
                             color=CORES_PEDRA[pedra], fontweight="bold")
    for ymin, ymax, cor, alpha in [(3.0,6.1,"#A8C4E0",0.08),(6.1,7.2,"#7F8C8D",0.08),
                                    (7.2,8.2,"#1E4D8C",0.08),(8.2,9.4,"#0A1F3C",0.08)]:
        ax_inde.axhspan(ymin, ymax, color=cor, alpha=alpha)
    ax_inde.set_title("Evolução do INDE Médio por Pedra (2020–2022)",
                      fontweight="bold", color=AZUL, pad=10)
    ax_inde.set_xticks(x_vals_i); ax_inde.set_xticklabels(anos_x, fontsize=11)
    ax_inde.set_ylabel("INDE Médio"); ax_inde.set_ylim(3.5, 9.8)
    ax_inde.legend(title="Pedra", frameon=False, fontsize=9, loc="upper right")
    ax_inde.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════
# ABA 11 — Q11 Ponto de Virada & INDE
# ══════════════════════════════════════════════
with abas[11]:
    st.subheader("Q11 — Trajetória do INDE por Ponto de Virada")

    COR_COM_PV  = AZUL
    COR_SEM_PV  = CINZA
    COR_MEDIANA = LARANJA
    ANOS_LABEL  = [2020, 2021, 2022]

    mask_3anos = (
        df["INDE_2020"].notna() &
        df["INDE_2021"].notna() &
        df["INDE_2022"].notna()
    )
    df3 = df[mask_3anos].copy()
    for col in ["INDE_2020","INDE_2021","INDE_2022"]:
        df3[col] = pd.to_numeric(df3[col], errors="coerce")
    for col in ["PONTO_VIRADA_2020","PONTO_VIRADA_2021","PONTO_VIRADA_2022"]:
        if df3[col].dtype == object:
            df3[col] = df3[col].map({"Sim":1,"Não":0,"sim":1,"não":0})

    df3["PV_ALGUM"] = (
        (df3["PONTO_VIRADA_2020"] == 1) |
        (df3["PONTO_VIRADA_2021"] == 1) |
        (df3["PONTO_VIRADA_2022"] == 1)
    ).astype(int)
    df3["DELTA_INDE"] = df3["INDE_2022"] - df3["INDE_2020"]

    n_com = df3["PV_ALGUM"].sum()
    n_sem = (df3["PV_ALGUM"] == 0).sum()
    st.info(f"Base analítica: **{len(df3)}** alunos nos 3 anos | **{n_com}** com Ponto de Virada | **{n_sem}** sem Ponto de Virada")

    # Gráfico 1: Trajetórias individuais
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle("Trajetória do INDE — Alunos presentes nos 3 anos (2020–2022)",
                 fontsize=15, fontweight="bold", y=1.02)

    for pv_val, titulo, cor_grupo, ax in [
        (1, "Com Ponto de Virada", COR_COM_PV, axes[0]),
        (0, "Sem Ponto de Virada", COR_SEM_PV, axes[1]),
    ]:
        sub = df3[df3["PV_ALGUM"] == pv_val]
        for _, row in sub.iterrows():
            ax.plot(ANOS_LABEL,
                    [row["INDE_2020"], row["INDE_2021"], row["INDE_2022"]],
                    color=cor_grupo, alpha=0.08, linewidth=0.8, zorder=1)
        medianas = [sub["INDE_2020"].median(), sub["INDE_2021"].median(), sub["INDE_2022"].median()]
        ax.plot(ANOS_LABEL, medianas, color=COR_MEDIANA, linewidth=3.2,
                marker="o", markersize=9, zorder=3, label="Mediana do grupo")
        for ano, med in zip(ANOS_LABEL, medianas):
            ax.annotate(f"{med:.2f}", xy=(ano, med), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontsize=10,
                        fontweight="bold", color=COR_MEDIANA, zorder=4)
        ax.axhline(6.0, color=VERMELHO, linestyle=":", linewidth=1.5, alpha=0.75,
                   label="Limiar Ágata (6.0)")
        ax.set_title(f"{titulo}\n(n = {len(sub)})", fontsize=12, color=cor_grupo, pad=10)
        ax.set_xticks(ANOS_LABEL); ax.set_xlabel("Ano")
        ax.set_ylim(2.0, 10.5); ax.grid(axis="y")
        ax.legend(fontsize=9, loc="lower left")
    axes[0].set_ylabel("INDE")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Gráfico 2: Delta INDE acumulado
    df_long_q11 = df3[["NOME","INDE_2020","INDE_2021","INDE_2022","PV_ALGUM"]].melt(
        id_vars=["NOME","PV_ALGUM"],
        value_vars=["INDE_2020","INDE_2021","INDE_2022"],
        var_name="ANO_COL", value_name="INDE")
    df_long_q11["ANO"]   = df_long_q11["ANO_COL"].str.extract(r"(\d{4})").astype(int)
    df_long_q11["GRUPO"] = df_long_q11["PV_ALGUM"].map({1:"Com Ponto de Virada", 0:"Sem Ponto de Virada"})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_long_q11, x="ANO", y="INDE", hue="GRUPO",
                 palette={"Com Ponto de Virada": COR_COM_PV, "Sem Ponto de Virada": COR_SEM_PV},
                 estimator="median", errorbar=("ci", 95),
                 linewidth=2.8, marker="o", markersize=9, err_style="band", ax=ax)
    ax.axhline(6.0, color=VERMELHO, linestyle=":", linewidth=1.5, alpha=0.8, label="Limiar Ágata (6.0)")
    ax.set_title("Evolução do INDE — Mediana com IC 95% por grupo",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(ANOS_LABEL); ax.set_xlabel("Ano"); ax.set_ylabel("INDE (mediana)")
    ax.set_ylim(4.5, 10.5); ax.legend(title="Grupo", fontsize=10)
    ax.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Gráfico 3: Distribuição Delta INDE
    fig, ax = plt.subplots(figsize=(11, 5))
    for pv_val, label, cor in [(1, "Com Ponto de Virada", COR_COM_PV),
                                (0, "Sem Ponto de Virada", COR_SEM_PV)]:
        dados = df3[df3["PV_ALGUM"] == pv_val]["DELTA_INDE"].dropna()
        sns.histplot(dados, ax=ax, color=cor, alpha=0.30, bins=30,
                     stat="density", label=f"{label} (n={len(dados)})")
        sns.kdeplot(dados, ax=ax, color=cor, linewidth=2.5)
        med   = dados.median()
        y_pos = ax.get_ylim()[1] * (0.88 if pv_val == 1 else 0.72)
        ax.axvline(med, color=cor, linestyle="--", linewidth=1.8, alpha=0.9)
        ax.text(med + 0.05, y_pos, f"Mediana\n{med:+.2f}",
                color=cor, fontsize=9, fontweight="bold", va="top")
    ax.axvline(0, color="black", linewidth=1.2, alpha=0.5, label="Delta = 0 (sem variação)")
    ax.set_title("Distribuição do Delta INDE acumulado (2022 − 2020)\npor grupo de Ponto de Virada",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Delta INDE (2022 − 2020)"); ax.set_ylabel("Densidade")
    ax.legend(fontsize=10); ax.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Datathon Passos Mágicos · Fase 5 Tech Challenge · FIAP · 2020–2022")
