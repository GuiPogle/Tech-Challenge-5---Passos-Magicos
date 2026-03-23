import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# ESTILOS GLOBAIS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #F5F7FB !important; }
    [data-testid="stSidebar"] * { color: #1B2A4A !important; }
    [data-testid="stSidebar"] p { color: #4A5568 !important; }
    [data-testid="stSidebar"] span { color: #1B2A4A !important; }
    [data-testid="stSidebar"] label { color: #1B2A4A !important; }
    [data-testid="stSidebar"] hr { border-color: #B8CEDD !important; }
    .main .block-container { background: #F5F7FB; padding-top: 1.5rem; }
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 18px 16px;
        text-align: center;
        border-top: 3px solid #1A3F7A;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .metric-card h2 { margin: 0; color: #1A3F7A; font-size: 1.9rem; font-weight: 600; }
    .metric-card p  { margin: 4px 0 0 0; color: #6B7A99; font-size: 0.85rem; }
    .risco-alto  { background: #FBEAEA; border-left: 4px solid #B03A2E; border-radius: 8px; padding: 16px; }
    .risco-medio { background: #EAF0FB; border-left: 4px solid #2E6BB0; border-radius: 8px; padding: 16px; }
    .risco-baixo { background: #EAF4EE; border-left: 4px solid #1A6B3A; border-radius: 8px; padding: 16px; }
    .secao-titulo { font-size: 1rem; font-weight: 600; color: #1A3F7A; margin-bottom: 8px; letter-spacing: 0.01em; }
    .upload-box {
        background: #FFFFFF;
        border: 2px dashed #B8CEDD;
        border-radius: 14px;
        padding: 56px 32px;
        text-align: center;
        margin: 60px auto;
        max-width: 520px;
    }
    .upload-box h2 { color: #1A3F7A; margin-bottom: 10px; }
    .upload-box p  { color: #6B7A99; font-size: 0.95rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────
AZUL_ESC  = '#1B2A4A'
AZUL      = '#2C4A7C'
AZUL_MED  = '#4A7FB5'
AZUL_CLA  = '#B8CEDD'
MARSALA   = '#6B2D3E'
VERM      = '#9B3A3A'
VERM_CLA  = '#C0635C'
CINZA_T   = '#3D3D3D'
CINZA     = '#8A9AB0'
CINZA_CLA = '#EEF1F6'
VERMELHO  = VERM

PEDRAS_ORDEM = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
PEDRAS_CORES = {
    'Quartzo':  '#8A9AB0',
    'Ágata':    '#4A7FB5',
    'Ametista': '#9B3A3A',
    'Topázio':  '#6B2D3E',
}

plt.rcParams.update({
    'axes.facecolor':    '#FFFFFF',
    'figure.facecolor':  '#FFFFFF',
    'axes.edgecolor':    '#D0D5DE',
    'axes.labelcolor':   '#3D3D3D',
    'xtick.color':       '#3D3D3D',
    'ytick.color':       '#3D3D3D',
    'text.color':        '#3D3D3D',
    'axes.titlecolor':   '#3D3D3D',
    'legend.labelcolor': '#3D3D3D',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#E5E9EF',
    'grid.linewidth':    0.5,
    'font.family':       'sans-serif',
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'figure.dpi':        130,
})

INDICADORES_ML = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE']
ALL_FEATURES = (
    [f'{ind}_2020' for ind in INDICADORES_ML] +
    [f'{ind}_2021' for ind in INDICADORES_ML]
)

# ──────────────────────────────────────────────
# SIDEBAR — UPLOAD + NAVEGAÇÃO
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Georgia,serif;font-size:1.6rem;font-weight:900;'
        'color:#1B3A6B;letter-spacing:0.03em;font-style:italic;margin-bottom:2px;">'
        'Passos Mágicos</div>'
        '<div style="font-size:0.75rem;color:#4A7FB5;font-weight:600;letter-spacing:0.06em;margin-bottom:16px;">'
        'ANALYTICS · PEDE 2020–2022</div>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#B8CEDD;margin:0 0 12px 0">', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#4A7FB5;font-size:0.75rem;font-weight:700;'
        'letter-spacing:0.1em;margin-bottom:6px;">DADOS</p>',
        unsafe_allow_html=True)

    arquivo_csv = st.file_uploader(
        "Faça upload do CSV",
        type=["csv"],
        help="Arquivo: PEDE_PASSOS_DATASET_FIAP.csv  (separador ponto-e-vírgula)",
        label_visibility="collapsed",
    )

    dados_carregados = arquivo_csv is not None

    if dados_carregados:
        st.success("✅ Arquivo carregado")
    else:
        st.caption("📂 Faça o upload do CSV para liberar o dashboard.")

    st.markdown('<hr style="border-color:#B8CEDD;margin:12px 0">', unsafe_allow_html=True)

    # Menu de navegação só aparece com dados
    if dados_carregados:
        st.markdown(
            '<p style="color:#4A7FB5;font-size:0.75rem;font-weight:700;'
            'letter-spacing:0.1em;margin-bottom:6px;">NAVEGAÇÃO</p>',
            unsafe_allow_html=True)
        pagina = st.radio("", [
            "Visão Geral",
            "Indicadores",
            "Perfil do Aluno",
            "Previsão de Risco",
            "Alunos em Risco",
            "Efetividade"
        ], label_visibility="collapsed")
    else:
        pagina = None

    st.markdown('<hr style="border-color:#B8CEDD;margin:12px 0">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8A9AB0;font-size:0.72rem;">Datathon FIAP · Fase 5<br>'
        'Passos Mágicos · 2020–2022</p>',
        unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TELA DE BOAS-VINDAS (sem dados)
# ──────────────────────────────────────────────
if not dados_carregados:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:3.2rem;margin-bottom:14px;">✨</div>
        <h2>Passos Mágicos — Analytics</h2>
        <p>
            Para visualizar o dashboard, faça o <strong>upload do arquivo CSV</strong>
            na barra lateral.<br><br>
            <strong>Arquivo esperado:</strong> PEDE_PASSOS_DATASET_FIAP.csv<br>
            <strong>Separador:</strong> ponto-e-vírgula ( ; )<br><br>
            <span style="font-size:0.82rem;color:#aaa;">
                Nenhum dado é enviado a servidores externos.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # ← bloqueia todo o restante do script

# ──────────────────────────────────────────────
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ──────────────────────────────────────────────
@st.cache_data
def carregar_dados(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), sep=';')
    indicadores = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE']
    for ind in indicadores:
        for ano in ['2020','2021','2022']:
            col = f'{ind}_{ano}'
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['IDADE_ALUNO_2020','ANOS_PM_2020','FASE_2022','NOTA_PORT_2022','NOTA_MAT_2022']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    mapa_bool = {'Sim':1,'sim':1,'Não':0,'Nao':0,'nao':0}
    for col in ['PONTO_VIRADA_2020','PONTO_VIRADA_2021','PONTO_VIRADA_2022',
                'BOLSISTA_2022','INDICADO_BOLSA_2022']:
        if col in df.columns:
            df[col] = df[col].map(mapa_bool)
    df['PEDRA_2020'] = df['PEDRA_2020'].replace({'D9891/2A': np.nan})
    df['PEDRA_2021'] = df['PEDRA_2021'].replace({'#NULO!': np.nan})
    df['PEDRA_2020_NUM'] = df['PEDRA_2020'].map({'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4})
    df['PEDRA_2021_NUM'] = df['PEDRA_2021'].map({'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4})
    df['DELTA_INDE'] = df['INDE_2021'] - df['INDE_2020']
    df['DELTA_IDA']  = df['IDA_2021']  - df['IDA_2020']
    df['DELTA_IEG']  = df['IEG_2021']  - df['IEG_2020']
    df['DELTA_IAN']  = df['IAN_2021']  - df['IAN_2020']
    df['DELTA_IPV']  = df['IPV_2021']  - df['IPV_2020']
    df['MEDIA_INDE'] = df[['INDE_2020','INDE_2021']].mean(axis=1)
    df['MEDIA_IDA']  = df[['IDA_2020','IDA_2021']].mean(axis=1)
    df['MEDIA_IEG']  = df[['IEG_2020','IEG_2021']].mean(axis=1)
    df['MEDIA_IAN']  = df[['IAN_2020','IAN_2021']].mean(axis=1)
    df['EM_RISCO']   = ((df['INDE_2022'] < 6.187) | (df['IAN_2022'] < 6.187)).astype(float)

    frames = []
    for ano in ['2020','2021','2022']:
        cols = {f'{ind}_{ano}': ind for ind in indicadores if f'{ind}_{ano}' in df.columns}
        cols[f'PEDRA_{ano}'] = 'PEDRA'
        cols[f'PONTO_VIRADA_{ano}'] = 'PONTO_VIRADA'
        temp = df[['NOME'] + list(cols.keys())].copy().rename(columns=cols)
        temp['ANO'] = int(ano)
        frames.append(temp)
    df_long = pd.concat(frames, ignore_index=True).dropna(subset=['INDE'])
    return df, df_long

@st.cache_resource
def treinar_modelo(_df):
    feats_disp = [f for f in ALL_FEATURES if f in _df.columns]
    d = _df[feats_disp + ['EM_RISCO']].dropna(subset=['EM_RISCO'])
    X = d[feats_disp]
    y = d['EM_RISCO']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    modelo.fit(X_train_imp, y_train)
    return modelo, imputer, feats_disp

# Lê os bytes uma única vez e repassa ao cache
file_bytes = arquivo_csv.read()
with st.spinner("Carregando e processando os dados..."):
    df, df_long = carregar_dados(file_bytes)
    modelo, imputer, FEATS_USADAS = treinar_modelo(df)

X_all     = df[FEATS_USADAS].copy()
X_all_imp = imputer.transform(X_all)
df['PROB_RISCO'] = modelo.predict_proba(X_all_imp)[:, 1]

# ══════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ══════════════════════════════════════════════
if pagina == "Visão Geral":
    st.title("Passos Mágicos — Visão Geral do Programa")
    st.markdown("Dados do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) · 2020–2022")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{df['NOME'].nunique():,}</h2>
            <p>Alunos no programa</p></div>""", unsafe_allow_html=True)
    with col2:
        pct_pv = df['PONTO_VIRADA_2022'].mean() * 100
        st.markdown(f"""<div class="metric-card">
            <h2>{pct_pv:.1f}%</h2>
            <p>Atingiram o Ponto de Virada (2022)</p></div>""", unsafe_allow_html=True)
    with col3:
        inde_medio = df['INDE_2022'].mean()
        st.markdown(f"""<div class="metric-card">
            <h2>{inde_medio:.2f}</h2>
            <p>INDE médio em 2022</p></div>""", unsafe_allow_html=True)
    with col4:
        pct_risco = df['EM_RISCO'].mean() * 100
        st.markdown(f"""<div class="metric-card" style="border-left-color:#E74C3C">
            <h2 style="color:#E74C3C">{pct_risco:.1f}%</h2>
            <p>Alunos em risco de defasagem</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="secao-titulo">Evolução do INDE médio (2020–2022)</p>', unsafe_allow_html=True)
        inde_ano = df_long.groupby('ANO')['INDE'].agg(['mean','std']).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.fill_between(inde_ano['ANO'],
                        inde_ano['mean'] - inde_ano['std'],
                        inde_ano['mean'] + inde_ano['std'],
                        alpha=0.15, color=AZUL)
        ax.plot(inde_ano['ANO'], inde_ano['mean'], 'o-', color=AZUL, linewidth=2.5, markersize=9)
        for _, row in inde_ano.iterrows():
            ax.annotate(f"{row['mean']:.2f}", xy=(row['ANO'], row['mean']),
                        xytext=(0, 12), textcoords='offset points',
                        ha='center', fontsize=11, fontweight='bold', color=AZUL)
        ax.set_xticks([2020, 2021, 2022])
        ax.set_ylim(5, 9)
        ax.set_ylabel('INDE médio')
        ax.grid(alpha=0.3)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown('<p class="secao-titulo">Distribuição de Pedras por ano (%)</p>', unsafe_allow_html=True)
        pedra_ano = df_long.groupby(['ANO','PEDRA'], observed=True).size().unstack(fill_value=0)
        pedra_ano = pedra_ano.reindex(columns=PEDRAS_ORDEM, fill_value=0)
        pedra_pct = pedra_ano.div(pedra_ano.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        pedra_pct.plot(kind='bar', stacked=True,
                       color=[PEDRAS_CORES[p] for p in PEDRAS_ORDEM],
                       ax=ax, edgecolor='white', width=0.45)
        ax.set_xticklabels([2020, 2021, 2022], rotation=0)
        ax.set_ylabel('%')
        ax.legend(title='Pedra', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f%%', label_type='center',
                         fontsize=8, color='white', fontweight='bold')
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('---')
    st.markdown('<p class="secao-titulo">Sistema de classificação por Pedras</p>', unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    pedra_info = [
        ('Quartzo',  '#8A9AB0', '2,405 – 5,506', 'Estágio inicial. Requer acompanhamento próximo para superar defasagens.'),
        ('Ágata',    '#4A7FB5', '5,506 – 6,868', 'Desenvolvimento moderado. Já demonstra progresso, ainda precisa de suporte.'),
        ('Ametista', '#9B3A3A', '6,868 – 8,230', 'Bom desempenho. Está dentro do esperado e tende a continuar evoluindo.'),
        ('Topázio',  '#6B2D3E', '8,230 – 9,294', 'Excelência. Candidato prioritário a bolsas e oportunidades externas.'),
    ]
    for col_p, (nome, cor, faixa, desc) in zip([col_p1,col_p2,col_p3,col_p4], pedra_info):
        with col_p:
            st.markdown(
                f'<div style="border-top:4px solid {cor};border-radius:10px;padding:14px;background:{cor}18;">'
                f'<div style="font-weight:700;color:{cor};font-size:1rem;margin-bottom:4px;">{nome}</div>'
                f'<div style="font-size:0.78rem;color:#3D3D3D;font-weight:600;margin-bottom:8px;">INDE: {faixa}</div>'
                f'<div style="font-size:0.78rem;color:#555;line-height:1.4;">{desc}</div>'
                '</div>',
                unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<p class="secao-titulo">% de alunos que atingiram o Ponto de Virada</p>', unsafe_allow_html=True)
    pv_data = {str(ano): df[f'PONTO_VIRADA_{ano}'].mean()*100 for ano in [2020,2021,2022]}
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(pv_data.keys(), pv_data.values(),
                  color=[AZUL_CLA, AZUL_MED, AZUL_ESC], edgecolor='white', width=0.4)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 25)
    ax.set_ylabel('%')
    ax.grid(axis='y', alpha=0.3)
    sns.despine()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════
# PÁGINA 2 — INDICADORES
# ══════════════════════════════════════════════
elif pagina == "Indicadores":
    st.title("Análise por Indicador")

    with st.expander("📖 Dicionário de indicadores — o que é cada sigla?"):
        st.markdown("""
| Sigla | Nome completo | O que mede |
|---|---|---|
| **IAN** | Adequação ao Nível | Se o aluno está no nível certo para sua fase |
| **IDA** | Desempenho Acadêmico | Média das notas nas disciplinas do programa |
| **IEG** | Engajamento | Participação e envolvimento nas atividades |
| **IAA** | Autoavaliação | Como o aluno avalia seu próprio progresso |
| **IPS** | Psicossocial | Bem-estar emocional e contexto social |
| **IPP** | Psicopedagogógico | Avaliação da equipe psicopedagogógica |
| **IPV** | Ponto de Virada | Proximidade ao momento de transformação |
| **INDE** | Índice de Desenvolvimento | Nota geral — média ponderada de todos acima |
""")

    st.markdown("---")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        indicador = st.selectbox("Indicador", ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE'])
    with col_f2:
        ano_sel = st.selectbox("Ano", [2022, 2021, 2020])

    col = f'{indicador}_{ano_sel}'
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<p class="secao-titulo">Distribuição do {indicador} em {ano_sel}</p>', unsafe_allow_html=True)
        dados = df[col].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(dados, bins=25, color=AZUL, alpha=0.4, ax=ax, stat='density')
        sns.kdeplot(dados, color=AZUL, linewidth=2.5, ax=ax)
        ax.axvline(dados.mean(), color=VERMELHO, linestyle='--', linewidth=2,
                   label=f'Média: {dados.mean():.2f}')
        ax.axvline(dados.median(), color=AZUL_MED, linestyle='--', linewidth=2,
                   label=f'Mediana: {dados.median():.2f}')
        ax.set_xlabel(indicador)
        ax.set_ylabel('Densidade')
        ax.legend()
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown(f'<p class="secao-titulo">{indicador} médio por Pedra em {ano_sel}</p>', unsafe_allow_html=True)
        pedra_col = f'PEDRA_{ano_sel}'
        d_bar = df[[pedra_col, col]].dropna()
        ordem = [p for p in PEDRAS_ORDEM if p in d_bar[pedra_col].values]
        medias_p = d_bar.groupby(pedra_col, observed=True)[col].mean().reindex(ordem)
        ns_p     = d_bar.groupby(pedra_col, observed=True)[col].count().reindex(ordem)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars_p = ax.bar(ordem, medias_p.values,
                        color=[PEDRAS_CORES[p] for p in ordem],
                        edgecolor='white', width=0.5)
        for bar_p, val_p, n_p in zip(bars_p, medias_p.values, ns_p.values):
            ax.text(bar_p.get_x()+bar_p.get_width()/2, val_p+0.12,
                    f'{val_p:.2f}\n(n={int(n_p)})',
                    ha='center', fontsize=9, color=CINZA_T, fontweight='bold')
        ax.axhline(d_bar[col].mean(), color=VERM, linestyle='--',
                   linewidth=1.5, label=f'Média geral: {d_bar[col].mean():.2f}')
        ax.set_xlabel('Pedra', color=CINZA_T)
        ax.set_ylabel(f'{indicador} médio', color=CINZA_T)
        ax.set_ylim(0, 11)
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=10)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown(f'<p class="secao-titulo">Matriz de correlação entre indicadores ({ano_sel})</p>', unsafe_allow_html=True)
    cols_corr = [f'{ind}_{ano_sel}' for ind in ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE']
                 if f'{ind}_{ano_sel}' in df.columns]
    corr = df[cols_corr].corr()
    corr.columns = [c.replace(f'_{ano_sel}','') for c in corr.columns]
    corr.index   = [c.replace(f'_{ano_sel}','') for c in corr.index]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
                center=0.3, vmin=0, vmax=1, linewidths=0.5, ax=ax, square=True,
                annot_kws={'size': 10})
    ax.set_title(f'Correlação entre indicadores — {ano_sel}', fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown(f'<p class="secao-titulo">Estatísticas descritivas — {indicador} {ano_sel}</p>', unsafe_allow_html=True)
    stats = df[col].describe().round(2)
    st.dataframe(stats.to_frame().T, use_container_width=True)

# ══════════════════════════════════════════════
# PÁGINA 3 — PERFIL DO ALUNO
# ══════════════════════════════════════════════
elif pagina == "Perfil do Aluno":
    st.title("Perfil Individual do Aluno")
    st.markdown("---")

    alunos = sorted(df['NOME'].dropna().unique().tolist())
    aluno_sel = st.selectbox("Selecione o aluno", alunos)
    st.markdown("---")

    row = df[df['NOME'] == aluno_sel].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pedra_22 = row.get('PEDRA_2022', 'N/D')
        cor_pedra = PEDRAS_CORES.get(str(pedra_22), CINZA) if pd.notna(pedra_22) else CINZA
        st.markdown(f"""<div class="metric-card" style="border-left-color:{cor_pedra}">
            <h2 style="color:{cor_pedra}">{pedra_22 if pd.notna(pedra_22) else 'N/D'}</h2>
            <p>Pedra 2022</p></div>""", unsafe_allow_html=True)
    with col2:
        inde_22 = row.get('INDE_2022', np.nan)
        st.markdown(f"""<div class="metric-card">
            <h2>{f'{inde_22:.2f}' if pd.notna(inde_22) else 'N/D'}</h2>
            <p>INDE 2022</p></div>""", unsafe_allow_html=True)
    with col3:
        pv_22 = row.get('PONTO_VIRADA_2022', np.nan)
        pv_txt = 'Sim ✅' if pv_22 == 1 else ('Não ❌' if pv_22 == 0 else 'N/D')
        st.markdown(f"""<div class="metric-card">
            <h2 style="font-size:1.4rem">{pv_txt}</h2>
            <p>Ponto de Virada 2022</p></div>""", unsafe_allow_html=True)
    with col4:
        prob = row.get('PROB_RISCO', np.nan)
        cor_risco = VERMELHO if prob >= 0.70 else (AZUL_MED if prob >= 0.40 else '#1A6B3A')
        st.markdown(f"""<div class="metric-card" style="border-left-color:{cor_risco}">
            <h2 style="color:{cor_risco}">{f'{prob*100:.1f}%' if pd.notna(prob) else 'N/D'}</h2>
            <p>Probabilidade de risco</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="secao-titulo">Radar de indicadores (2022) vs média geral</p>', unsafe_allow_html=True)
        cats = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV']
        vals_aluno = [row.get(f'{c}_2022', np.nan) for c in cats]
        vals_media = [df[f'{c}_2022'].mean() for c in cats]

        if any(pd.notna(v) for v in vals_aluno):
            N = len(cats)
            angulos = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
            va = [v if pd.notna(v) else 0 for v in vals_aluno] + [vals_aluno[0] if pd.notna(vals_aluno[0]) else 0]
            vm = vals_media + [vals_media[0]]
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax.plot(angulos, va, 'o-', linewidth=2, color=AZUL, label=aluno_sel)
            ax.fill(angulos, va, alpha=0.2, color=AZUL)
            ax.plot(angulos, vm, 's--', linewidth=1.5, color=CINZA, label='Média geral')
            ax.fill(angulos, vm, alpha=0.08, color=CINZA)
            ax.set_xticks(angulos[:-1])
            ax.set_xticklabels(cats, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 10)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Dados insuficientes para o radar.")

    with col_b:
        st.markdown('<p class="secao-titulo">Evolução do INDE ao longo dos anos</p>', unsafe_allow_html=True)
        inde_aluno = {2020: row.get('INDE_2020', np.nan),
                      2021: row.get('INDE_2021', np.nan),
                      2022: row.get('INDE_2022', np.nan)}
        inde_medio_geral = {2020: df['INDE_2020'].mean(),
                            2021: df['INDE_2021'].mean(),
                            2022: df['INDE_2022'].mean()}
        anos_disp = [a for a, v in inde_aluno.items() if pd.notna(v)]
        if anos_disp:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(list(inde_medio_geral.keys()), list(inde_medio_geral.values()),
                    's--', color=CINZA, linewidth=1.5, markersize=6, label='Média geral', alpha=0.7)
            ax.plot(anos_disp, [inde_aluno[a] for a in anos_disp],
                    'o-', color=AZUL, linewidth=2.5, markersize=9, label=aluno_sel)
            for a in anos_disp:
                ax.annotate(f'{inde_aluno[a]:.2f}', xy=(a, inde_aluno[a]),
                            xytext=(0, 12), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold', color=AZUL)
            ax.set_xticks([2020, 2021, 2022])
            ax.set_ylim(0, 10)
            ax.set_ylabel('INDE')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            sns.despine()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Sem dados de INDE disponíveis.")

    st.markdown("---")
    st.markdown('<p class="secao-titulo">Indicadores detalhados por ano</p>', unsafe_allow_html=True)
    inds = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV','INDE']
    tabela = {}
    for ind in inds:
        tabela[ind] = {
            2020: round(row.get(f'{ind}_2020', np.nan), 2),
            2021: round(row.get(f'{ind}_2021', np.nan), 2),
            2022: round(row.get(f'{ind}_2022', np.nan), 2),
        }
    df_tabela = pd.DataFrame(tabela).T
    df_tabela.columns = ['2020','2021','2022']
    st.dataframe(df_tabela, use_container_width=True)

# ══════════════════════════════════════════════
# PÁGINA 4 — PREVISÃO DE RISCO
# ══════════════════════════════════════════════
elif pagina == "Previsão de Risco":
    st.title("Previsão de Risco de Defasagem")
    st.markdown("Insira os indicadores do aluno para calcular a probabilidade de risco de defasagem no próximo ano.")

    with st.expander("📖 O que significa cada sigla? Clique para ver o dicionário de indicadores"):
        st.markdown("""
| Sigla | Nome completo | O que mede |
|---|---|---|
| **IAN** | Indicador de Adequação ao Nível | Se o aluno está no nível certo para sua fase |
| **IDA** | Indicador de Desempenho Acadêmico | Média das notas de aprendizagem do aluno nas disciplinas |
| **IEG** | Indicador de Engajamento | Grau de participação e envolvimento nas atividades do programa |
| **IAA** | Indicador de Autoavaliação | Como o aluno percebe seu próprio desempenho e evolução |
| **IPS** | Indicador Psicossocial | Bem-estar emocional e condições psicossociais do aluno |
| **IPP** | Indicador Psicopedagógico | Avaliação feita pela equipe psicopedagógica sobre o aluno |
| **IPV** | Indicador de Ponto de Virada | Proximidade do aluno ao seu momento de transformação |
| **INDE** | Índice de Desenvolvimento Educacional | Nota geral do aluno — média ponderada de todos os indicadores acima |
        """)

    st.markdown("---")
    st.info("**IAN** só assume 3 valores conforme o dicionário oficial: "
            "**10** (em fase) · **5** (defasagem moderada) · **2,5** (defasagem severa). "
            "Os demais indicadores são contínuos de 0 a 10.")

    st.markdown("### Indicadores de 2021")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ian_21 = st.selectbox("IAN — Adequação ao Nível (2021)", options=[10.0, 5.0, 2.5],
            format_func=lambda x: {10.0:"10 — Em fase",5.0:"5 — Defasagem moderada",2.5:"2,5 — Defasagem severa"}[x], index=0)
        ida_21 = st.slider("IDA — Desempenho Acadêmico (2021)", 0.0, 10.0, 6.0, 0.1)
    with col2:
        ieg_21 = st.slider("IEG — Engajamento (2021)", 0.0, 10.0, 7.0, 0.1)
        iaa_21 = st.slider("IAA — Autoavaliação (2021)", 0.0, 10.0, 8.0, 0.1)
    with col3:
        ips_21 = st.slider("IPS — Psicossocial (2021)", 0.0, 10.0, 7.0, 0.1)
        ipp_21 = st.slider("IPP — Psicopedagógico (2021)", 0.0, 10.0, 7.5, 0.1)
    with col4:
        ipv_21 = st.slider("IPV — Ponto de Virada (2021)", 0.0, 10.0, 7.5, 0.1)
        inde_21 = st.slider("INDE — Índice Geral (2021)", 0.0, 10.0, 7.0, 0.1)

    st.markdown("### Indicadores de 2020")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        ian_20 = st.selectbox("IAN — Adequação ao Nível (2020)", options=[10.0, 5.0, 2.5],
            format_func=lambda x: {10.0:"10 — Em fase",5.0:"5 — Defasagem moderada",2.5:"2,5 — Defasagem severa"}[x], index=0)
        ida_20 = st.slider("IDA — Desempenho Acadêmico (2020)", 0.0, 10.0, 6.5, 0.1)
    with col6:
        ieg_20 = st.slider("IEG — Engajamento (2020)", 0.0, 10.0, 7.5, 0.1)
        iaa_20 = st.slider("IAA — Autoavaliação (2020)", 0.0, 10.0, 8.5, 0.1)
    with col7:
        ips_20 = st.slider("IPS — Psicossocial (2020)", 0.0, 10.0, 7.0, 0.1)
        ipp_20 = st.slider("IPP — Psicopedagógico (2020)", 0.0, 10.0, 7.0, 0.1)
    with col8:
        ipv_20 = st.slider("IPV — Ponto de Virada (2020)", 0.0, 10.0, 7.5, 0.1)
        inde_20 = st.slider("INDE — Índice Geral (2020)", 0.0, 10.0, 7.3, 0.1)

    st.markdown("---")
    if st.button("Calcular Probabilidade de Risco", type="primary", use_container_width=True):
        entrada = np.array([[
            ian_20, ida_20, ieg_20, iaa_20, ips_20, ipp_20, ipv_20, inde_20,
            ian_21, ida_21, ieg_21, iaa_21, ips_21, ipp_21, ipv_21, inde_21,
        ]])
        entrada_imp = imputer.transform(entrada)
        proba = modelo.predict_proba(entrada_imp)[0][1]

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            if proba >= 0.70:
                nivel = "ALTO"; classe = "risco-alto"; emoji = "🔴"
            elif proba >= 0.40:
                nivel = "MODERADO"; classe = "risco-medio"; emoji = "🟡"
            else:
                nivel = "BAIXO"; classe = "risco-baixo"; emoji = "🟢"
            st.markdown(f"""
            <div class="{classe}" style="text-align:center; padding: 24px;">
                <h1>{emoji} {proba*100:.1f}%</h1>
                <h3>Risco {nivel} de Defasagem</h3>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### O que esse resultado significa?")
        if proba >= 0.4:
            st.error("**Risco ALTO — Ação imediata recomendada**\n\n"
                     "- **Avaliação psicopedagógica** prioritária\n"
                     "- **Acompanhamento individualizado** com reforço nas disciplinas\n"
                     "- **Conversa com a família** sobre fatores externos que impactam o IPS\n"
                     "- **Revisão da fase atual** — verificar adequação ao nível (IAN)\n"
                     "- **Reavaliação em 30 dias**")
        elif proba >= 0.2:
            st.warning("**Risco MODERADO — Atenção e monitoramento**\n\n"
                       "- **Monitoramento próximo** dos indicadores IEG e IDA\n"
                       "- **Diálogo com o aluno** sobre sua autoavaliação\n"
                       "- **Suporte adicional** nas áreas abaixo da média da fase\n"
                       "- **Reavaliação** no próximo ciclo do PEDE")
        else:
            st.success("**Risco BAIXO — Perfil favorável**\n\n"
                       "- Manter o acompanhamento regular\n"
                       "- Considerar indicação para bolsa se INDE ≥ 8,0\n"
                       "- Continuar monitorando no próximo PEDE")

        st.markdown("---")
        st.markdown("### Quais indicadores mais pesaram nesta previsão?")
        imp_df = pd.DataFrame({
            'feature': FEATS_USADAS,
            'importance': modelo.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        imp_df['nome'] = imp_df['feature'].apply(
            lambda x: x.replace('_2020',' (2020)').replace('_2021',' (2021)'))
        fig, ax = plt.subplots(figsize=(8, 5))
        cores_imp = [VERMELHO if 'MEDIA' in f or 'DELTA' in f else
                     AZUL_MED if '2021' in f else AZUL for f in imp_df['feature']]
        bars = ax.barh(imp_df['nome'][::-1], imp_df['importance'][::-1],
                       color=cores_imp[::-1], edgecolor='white', height=0.6)
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
        ax.set_xlabel('Importância no modelo')
        ax.set_title('Top 10 indicadores mais relevantes para a previsão', fontweight='bold')
        legenda = [mpatches.Patch(color=AZUL_MED, label='Indicador de 2021'),
                   mpatches.Patch(color=AZUL,     label='Indicador de 2020')]
        ax.legend(handles=legenda, fontsize=8, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")
        st.markdown("### Resumo dos dados inseridos")
        col_s1, col_s2 = st.columns(2)
        medias_ref = {'IAN':{'2020':7.43,'2021':6.90},'IDA':{'2020':6.32,'2021':5.43},
                      'IEG':{'2020':7.68,'2021':6.82},'IAA':{'2020':8.37,'2021':8.15},
                      'IPS':{'2020':6.74,'2021':6.84},'IPP':{'2020':7.07,'2021':7.58},
                      'IPV':{'2020':7.24,'2021':7.41},'INDE':{'2020':7.30,'2021':6.89}}
        vals_inp = {'IAN':(ian_20,ian_21),'IDA':(ida_20,ida_21),'IEG':(ieg_20,ieg_21),
                    'IAA':(iaa_20,iaa_21),'IPS':(ips_20,ips_21),'IPP':(ipp_20,ipp_21),
                    'IPV':(ipv_20,ipv_21),'INDE':(inde_20,inde_21)}
        for col_s, ano, idx in [(col_s1,'2020',0),(col_s2,'2021',1)]:
            with col_s:
                st.caption(f"**Indicadores {ano}**")
                rows = []
                for ind in INDICADORES_ML:
                    val = vals_inp[ind][idx]
                    med = medias_ref[ind][ano]
                    rows.append({'Indicador':ind, f'Valor {ano}':val, f'Média {ano}':med,
                                 'Situação':'✅ Acima' if val >= med else '⚠️ Abaixo'})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# PÁGINA 5 — ALUNOS EM RISCO
# ══════════════════════════════════════════════
elif pagina == "Alunos em Risco":
    st.title("Triagem — Alunos em Risco de Defasagem")

    with st.expander("📖 O que é cada Fase? Clique para entender o filtro"):
        st.markdown("""
| Fase | Corresponde a | Descrição |
|---|---|---|
| **0** | 1º e 2º ano | Alfabetização — base da leitura e escrita |
| **1** | 3º e 4º ano | Consolidação da leitura e matemática básica |
| **2** | 5º e 6º ano | Transição para o fundamental II |
| **3** | 7º e 8º ano | Aprofundamento — fase crítica para engajamento |
| **4** | 9º ano | Preparação para o ensino médio |
| **5** | 1º EM | Início do ensino médio |
| **6** | 2º EM | Consolidação do ensino médio |
| **7** | 3º EM | Preparação para vestibular e mercado de trabalho |
""")

    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        threshold = st.slider("Threshold de risco mínimo", 0.1, 0.9, 0.3, 0.05,
                               help="Alunos com probabilidade acima desse valor serão listados")
    with col_f2:
        pedra_filtro = st.multiselect("Filtrar por Pedra (2022)", PEDRAS_ORDEM, default=PEDRAS_ORDEM)
    with col_f3:
        fases_disp = sorted(df['FASE_2022'].dropna().unique().tolist())
        fase_filtro = st.multiselect("Filtrar por Fase", [int(f) for f in fases_disp],
                                     default=[int(f) for f in fases_disp])

    st.markdown("---")
    d_risco = df[
        (df['PROB_RISCO'] >= threshold) &
        (df['PEDRA_2022'].isin(pedra_filtro)) &
        (df['FASE_2022'].isin(fase_filtro))
    ].copy()

    d_risco['FAIXA_RISCO'] = pd.cut(d_risco['PROB_RISCO'],
        bins=[0, 0.3, 0.5, 1.0],
        labels=['Moderado (30-50%)', 'Alto (50-70%)', 'Muito alto (>70%)'])

    cols_exibir = ['NOME','PEDRA_2022','FASE_2022','INDE_2022','IAN_2022',
                   'IEG_2022','IDA_2022','PROB_RISCO']
    d_exibir = d_risco[cols_exibir].copy()
    d_exibir['PROB_RISCO'] = (d_exibir['PROB_RISCO'] * 100).round(1).astype(str) + '%'
    d_exibir = d_exibir.sort_values('PROB_RISCO', ascending=False)
    d_exibir.columns = ['Aluno','Pedra','Fase','INDE 2022','IAN 2022',
                        'IEG 2022','IDA 2022','Prob. Risco (%)']

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: st.metric("Alunos sinalizados", len(d_risco))
    with col_s2: st.metric("% da base total", f"{len(d_risco)/len(df)*100:.1f}%")
    with col_s3: st.metric("INDE médio do grupo", f"{d_risco['INDE_2022'].mean():.2f}")

    st.markdown("---")
    st.dataframe(d_exibir, use_container_width=True, height=400)

    csv = d_exibir.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Exportar lista como CSV",
        data=csv,
        file_name=f'alunos_em_risco_threshold_{threshold}.csv',
        mime='text/csv',
        use_container_width=True
    )

    st.markdown("---")
    st.markdown('<p class="secao-titulo">Distribuição de faixas de risco</p>', unsafe_allow_html=True)
    faixas = d_risco['FAIXA_RISCO'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(faixas.index, faixas.values,
                  color=[AZUL_CLA, VERM_CLA, VERMELHO], edgecolor='white', width=0.4)
    ax.bar_label(bars, fmt='%d', padding=4, fontsize=11, fontweight='bold')
    ax.set_ylabel('Nº de alunos')
    ax.grid(axis='y', alpha=0.3)
    sns.despine()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════
# PÁGINA 6 — EFETIVIDADE
# ══════════════════════════════════════════════
elif pagina == "Efetividade":
    st.title("Efetividade do Programa")
    st.markdown("Análise do impacto da Passos Mágicos no desenvolvimento dos alunos ao longo de 2020–2022.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="secao-titulo">INDE médio por Pedra ao longo dos anos</p>', unsafe_allow_html=True)
        inde_pedra = df_long.groupby(['ANO','PEDRA'], observed=True)['INDE'].mean().reset_index().dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        for pedra, cor in PEDRAS_CORES.items():
            d = inde_pedra[inde_pedra['PEDRA'] == pedra]
            if not d.empty:
                ax.plot(d['ANO'], d['INDE'], 'o-', color=cor, linewidth=2.5, markersize=9, label=pedra)
                for _, row in d.iterrows():
                    ax.annotate(f'{row["INDE"]:.2f}', xy=(row['ANO'], row['INDE']),
                                xytext=(0, 10), textcoords='offset points',
                                ha='center', fontsize=8, color=cor, fontweight='bold')
        ax.set_xticks([2020, 2021, 2022])
        ax.set_ylim(3, 10)
        ax.set_ylabel('INDE médio', color=CINZA_T)
        ax.legend(title='Pedra', fontsize=9)
        ax.grid(alpha=0.3)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<p class="secao-titulo">Mobilidade de Pedras (2021 → 2022)</p>', unsafe_allow_html=True)
        df['PEDRA_2022_NUM'] = df['PEDRA_2022'].map({'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4})
        df['PEDRA_2021_NUM'] = df['PEDRA_2021'].map({'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4})
        melhora = int((df['PEDRA_2022_NUM'] > df['PEDRA_2021_NUM']).sum())
        piora   = int((df['PEDRA_2022_NUM'] < df['PEDRA_2021_NUM']).sum())
        estavel = int((df['PEDRA_2022_NUM'] == df['PEDRA_2021_NUM']).sum())
        total   = melhora + piora + estavel
        fig, ax = plt.subplots(figsize=(6, 4))
        vals    = [melhora, estavel, piora]
        rotulos = [f'Subiu\n{melhora} ({melhora/total*100:.1f}%)',
                   f'Manteve\n{estavel} ({estavel/total*100:.1f}%)',
                   f'Desceu\n{piora} ({piora/total*100:.1f}%)']
        ax.pie(vals, labels=rotulos, colors=[AZUL_MED, CINZA, MARSALA],
               startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2),
               textprops=dict(fontsize=9, color=CINZA_T))
        ax.set_title('Mobilidade 2021 → 2022', fontweight='bold', color=CINZA_T)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown('<p class="secao-titulo">INDE médio — bolsistas vs não bolsistas</p>', unsafe_allow_html=True)
        d_bolsa = df.dropna(subset=['BOLSISTA_2022','INDE_2022']).copy()
        d_bolsa['Grupo'] = d_bolsa['BOLSISTA_2022'].map({1:'Bolsista', 0:'Não bolsista'})
        grupos_b  = ['Não bolsista', 'Bolsista']
        medias_b  = d_bolsa.groupby('Grupo')['INDE_2022'].mean()
        ns_b      = d_bolsa.groupby('Grupo')['INDE_2022'].count()
        vals_b    = [medias_b.get(g, 0) for g in grupos_b]
        ns_vals_b = [ns_b.get(g, 0) for g in grupos_b]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars_b = ax.bar(grupos_b, vals_b, color=[CINZA, MARSALA], edgecolor='white', width=0.45)
        for bar_b, val_b, n_b in zip(bars_b, vals_b, ns_vals_b):
            ax.text(bar_b.get_x()+bar_b.get_width()/2, val_b+0.08,
                    f'{val_b:.2f}\n(n={int(n_b)})',
                    ha='center', fontsize=10, fontweight='bold', color=CINZA_T)
        media_geral_b = d_bolsa['INDE_2022'].mean()
        ax.axhline(media_geral_b, color=VERM_CLA, linestyle='--',
                   linewidth=1.5, label=f'Média geral: {media_geral_b:.2f}')
        ax.set_ylabel('INDE médio 2022', color=CINZA_T)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=8)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown('<p class="secao-titulo">Variação dos indicadores 2020 → 2022</p>', unsafe_allow_html=True)
        inds_ev = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV']
        medias_ev = {ind: round(df[f'{ind}_2022'].mean() - df[f'{ind}_2020'].mean(), 3) for ind in inds_ev}
        ev_df  = pd.Series(medias_ev).sort_values()
        cores_ev = [MARSALA if v < 0 else AZUL_MED for v in ev_df.values]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars_ev = ax.barh(ev_df.index, ev_df.values, color=cores_ev, edgecolor='white', height=0.5)
        ax.axvline(0, color=CINZA_T, linewidth=1)
        ax.bar_label(bars_ev, fmt='%.2f', padding=4, fontsize=9, fontweight='bold', color=CINZA_T)
        ax.set_xlabel('Variação média (2022 − 2020)', color=CINZA_T)
        leg_el = [mpatches.Patch(color=AZUL_MED, label='Melhora'),
                  mpatches.Patch(color=MARSALA,  label='Queda')]
        ax.legend(handles=leg_el, fontsize=8)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col5:
        st.markdown('<p class="secao-titulo">% em risco por faixa etária</p>', unsafe_allow_html=True)
        df['FAIXA_ET'] = pd.cut(df['IDADE_ALUNO_2020'],
            bins=[0,10,13,16,30], labels=['Até 10','11–13','14–16','17+'])
        risco_faixa = df.dropna(subset=['FAIXA_ET','EM_RISCO'])
        pct_risco_f = risco_faixa.groupby('FAIXA_ET', observed=True)['EM_RISCO'].mean() * 100
        ns_faixa    = risco_faixa.groupby('FAIXA_ET', observed=True)['EM_RISCO'].count()
        fig, ax = plt.subplots(figsize=(5, 4))
        cores_faixa = [AZUL_CLA, AZUL_MED, VERM, MARSALA]
        bars_f = ax.bar(pct_risco_f.index, pct_risco_f.values,
                        color=cores_faixa, edgecolor='white', width=0.5)
        for bar_f, val_f, n_f in zip(bars_f, pct_risco_f.values, ns_faixa.values):
            ax.text(bar_f.get_x()+bar_f.get_width()/2, val_f+0.4,
                    f'{val_f:.1f}%\n(n={int(n_f)})',
                    ha='center', fontsize=9, fontweight='bold', color=CINZA_T)
        ax.set_ylabel('% em risco', color=CINZA_T)
        ax.set_ylim(0, 30)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    col6, col7, col8 = st.columns(3)

    with col6:
        st.markdown('<p class="secao-titulo">% Ponto de Virada por Pedra (2022)</p>', unsafe_allow_html=True)
        d_pv     = df.dropna(subset=['PEDRA_2022','PONTO_VIRADA_2022'])
        pv_pedra = d_pv.groupby('PEDRA_2022', observed=True)['PONTO_VIRADA_2022'].mean() * 100
        pv_pedra = pv_pedra.reindex(PEDRAS_ORDEM).dropna()
        fig, ax  = plt.subplots(figsize=(5, 4))
        bars_pv  = ax.bar(pv_pedra.index, pv_pedra.values,
                          color=[PEDRAS_CORES[p] for p in pv_pedra.index],
                          edgecolor='white', width=0.5)
        ax.bar_label(bars_pv, fmt='%.1f%%', padding=4, fontsize=10, fontweight='bold', color=CINZA_T)
        ax.set_ylabel('% que atingiu o PV', color=CINZA_T)
        ax.set_ylim(0, 45)
        ax.tick_params(axis='x', rotation=10)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col7:
        st.markdown('<p class="secao-titulo">INDE médio por tempo na Passos Mágicos</p>', unsafe_allow_html=True)
        d_tempo = df[['ANOS_PM_2020','INDE_2020']].dropna()
        it = d_tempo.groupby('ANOS_PM_2020')['INDE_2020'].agg(['mean','count']).reset_index()
        it.columns = ['anos','media','n']
        it = it[it['n'] >= 10]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(it['anos'], it['media'], 'o-', color=AZUL_MED, linewidth=2.5, markersize=9)
        for _, row in it.iterrows():
            ax.annotate(f'{row["media"]:.2f}\n(n={int(row["n"])})',
                        xy=(row['anos'], row['media']),
                        xytext=(0, 14), textcoords='offset points',
                        ha='center', fontsize=8, color=AZUL_MED, fontweight='bold')
        ax.set_xlabel('Anos na Passos Mágicos', color=CINZA_T)
        ax.set_ylabel('INDE médio', color=CINZA_T)
        ax.set_ylim(5, 10)
        ax.grid(alpha=0.3)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col8:
        st.markdown('<p class="secao-titulo">IDA médio por Pedra e ano</p>', unsafe_allow_html=True)
        ida_pedra = df_long.groupby(['ANO','PEDRA'], observed=True)['IDA'].mean().reset_index().dropna()
        fig, ax = plt.subplots(figsize=(5, 4))
        for pedra, cor in PEDRAS_CORES.items():
            d = ida_pedra[ida_pedra['PEDRA'] == pedra]
            if not d.empty:
                ax.plot(d['ANO'], d['IDA'], 'o--', color=cor, linewidth=1.8,
                        markersize=7, label=pedra, alpha=0.9)
        ax.set_xticks([2020, 2021, 2022])
        ax.set_ylim(2, 10)
        ax.set_ylabel('IDA médio', color=CINZA_T)
        ax.legend(title='Pedra', fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()
