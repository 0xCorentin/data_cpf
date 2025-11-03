import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des données CPF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_cpf_data(file_content: bytes, separator: str = ';', encoding: str = 'latin-1') -> pd.DataFrame:
    """Charge les données CPF depuis le contenu d'un fichier uploadé"""
    try:
        from io import BytesIO
        file_buffer = BytesIO(file_content)
        df = pd.read_csv(file_buffer, sep=separator, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

def validate_file_format(df: pd.DataFrame) -> tuple[bool, list]:
    """Valide que le fichier contient les colonnes nécessaires"""
    required_columns = [
        'annee_mois', 'annee', 'mois', 'type_referentiel',
        'intitule_certification', 'raison_sociale_of_contractant',
        'entrees_formation', 'sorties_realisation_partielle', 'sorties_realisation_totale'
    ]
    
    # Nettoyer les noms de colonnes pour la comparaison
    df_columns = [col.strip().lower() for col in df.columns]
    required_columns_lower = [col.lower() for col in required_columns]
    
    missing_columns = []
    for req_col in required_columns_lower:
        if req_col not in df_columns:
            missing_columns.append(req_col)
    
    is_valid = len(missing_columns) == 0
    return is_valid, missing_columns

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et prépare les données"""
    if df.empty:
        return df
    
    # Nettoyer les noms de colonnes (enlever les espaces)
    df.columns = df.columns.str.strip()
    
    # Valider le format
    is_valid, missing_columns = validate_file_format(df)
    if not is_valid:
        st.error(f"❌ Colonnes manquantes dans le fichier : {', '.join(missing_columns)}")
        st.info("Vérifiez que votre fichier contient toutes les colonnes requises.")
        return pd.DataFrame()
    
    # Convertir les colonnes numériques
    numeric_columns = ['entrees_formation', 'sorties_realisation_partielle', 'sorties_realisation_totale']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convertir la date
    if 'date_chargement' in df.columns:
        df['date_chargement'] = pd.to_datetime(df['date_chargement'], dayfirst=True, errors='coerce')
    
    return df

def calculate_kpis(df: pd.DataFrame) -> dict:
    """Calcule les KPIs principaux"""
    if df.empty:
        return {}
    
    kpis = {
        'total_entrees': int(df['entrees_formation'].sum()),
        'total_sorties_totales': int(df['sorties_realisation_totale'].sum()),
        'total_sorties_partielles': int(df['sorties_realisation_partielle'].sum()),
        'nb_organismes': df['raison_sociale_of_contractant'].nunique(),
        'nb_certifications': df['intitule_certification'].nunique(),
        'taux_reussite': (df['sorties_realisation_totale'].sum() / df['entrees_formation'].sum() * 100) if df['entrees_formation'].sum() > 0 else 0
    }
    
    return kpis

def create_certification_chart(df: pd.DataFrame) -> go.Figure:
    """Crée un graphique des top certifications"""
    cert_data = df.groupby('intitule_certification').agg({
        'entrees_formation': 'sum',
        'sorties_realisation_totale': 'sum'
    }).reset_index()
    
    cert_data = cert_data.sort_values('entrees_formation', ascending=False).head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Entrées en formation',
        x=cert_data['intitule_certification'],
        y=cert_data['entrees_formation'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Sorties avec réalisation totale',
        x=cert_data['intitule_certification'],
        y=cert_data['sorties_realisation_totale'],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Top 10 des certifications par nombre d'entrées",
        xaxis_tickangle=-45,
        barmode='group',
        height=500
    )
    
    return fig

def create_monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Crée un graphique de tendance mensuelle"""
    monthly_data = df.groupby('annee_mois').agg({
        'entrees_formation': 'sum',
        'sorties_realisation_totale': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_data['annee_mois'],
        y=monthly_data['entrees_formation'],
        mode='lines+markers',
        name='Entrées en formation',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=monthly_data['annee_mois'],
        y=monthly_data['sorties_realisation_totale'],
        mode='lines+markers',
        name='Sorties réussite totale',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="Évolution mensuelle des entrées et sorties",
        xaxis_title="Mois",
        yaxis_title="Nombre",
        height=400
    )
    
    return fig

def analyze_market_segments(df: pd.DataFrame) -> dict:
    """Analyse les segments de marché par type de référentiel"""
    segments = {}
    
    for ref_type in df['type_referentiel'].unique():
        segment_data = df[df['type_referentiel'] == ref_type]
        
        # Présence AFPA dans ce segment
        afpa_in_segment = segment_data[segment_data['raison_sociale_of_contractant'] == 'AFPA ENTREPRISES']
        afpa_presence = len(afpa_in_segment) > 0
        
        # Leaders du segment
        segment_leaders = segment_data.groupby('raison_sociale_of_contractant')['entrees_formation'].sum().sort_values(ascending=False).head(3)
        
        segments[ref_type] = {
            'total_entries': segment_data['entrees_formation'].sum(),
            'afpa_presence': afpa_presence,
            'afpa_entries': afpa_in_segment['entrees_formation'].sum() if afpa_presence else 0,
            'leaders': segment_leaders.to_dict(),
            'market_size': len(segment_data),
            'nb_organisms': segment_data['raison_sociale_of_contractant'].nunique()
        }
    
    return segments

def create_competitor_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Crée un graphique de comparaison des principaux concurrents"""
    # Analyser les top 10 organismes (incluant AFPA)
    top_orgs = df.groupby('raison_sociale_of_contractant').agg({
        'entrees_formation': 'sum',
        'sorties_realisation_totale': 'sum',
        'intitule_certification': 'nunique'
    }).reset_index()
    
    top_orgs['taux_reussite'] = (top_orgs['sorties_realisation_totale'] / top_orgs['entrees_formation'] * 100).round(1)
    top_orgs = top_orgs.sort_values('entrees_formation', ascending=False).head(10)
    
    # Créer le graphique en barres
    fig = go.Figure()
    
    # Colorer AFPA différemment
    colors = ['red' if org == 'AFPA ENTREPRISES' else 'lightblue' for org in top_orgs['raison_sociale_of_contractant']]
    
    fig.add_trace(go.Bar(
        x=top_orgs['raison_sociale_of_contractant'],
        y=top_orgs['entrees_formation'],
        name='Entrées formation',
        marker_color=colors,
        text=top_orgs['taux_reussite'],
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top 10 Organismes - Volume d'Entrées et Taux de Réussite",
        xaxis_title="Organisme de Formation",
        yaxis_title="Nombre d'Entrées",
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    return fig

def main():
    """Fonction principale de l'application Streamlit"""
    
    st.title("📊 Analyse des données CPF - Organismes de Formation")
    st.markdown("---")
    
    # Section de chargement de fichier
    st.subheader("📁 Chargement des données")
    
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier de données CPF",
        type=['csv'],
        help="Sélectionnez un fichier CSV contenant les données CPF avec les colonnes requises"
    )
    
    if uploaded_file is None:
        st.info("👆 Veuillez charger un fichier CSV pour commencer l'analyse.")
        st.markdown("### 📋 Format de fichier attendu")
        st.markdown("Le fichier CSV doit contenir les colonnes suivantes :")
        
        expected_columns = [
            "annee_mois", "annee", "mois", "type_referentiel", 
            "intitule_certification", "raison_sociale_of_contractant",
            "entrees_formation", "sorties_realisation_partielle", 
            "sorties_realisation_totale", "date_chargement"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            for col in expected_columns[:5]:
                st.write(f"• `{col}`")
        with col2:
            for col in expected_columns[5:]:
                st.write(f"• `{col}`")
        
        st.markdown("### 📝 Instructions")
        st.write("1. Le fichier doit être au format CSV")
        st.write("2. Le séparateur peut être `;` ou `,`")
        st.write("3. L'encodage peut être UTF-8 ou Latin-1")
        st.write("4. La première ligne doit contenir les noms des colonnes")
        
        return
    
    # Chargement des données depuis le fichier uploadé
    with st.spinner("Chargement des données..."):
        try:
            # Essayer de détecter le séparateur et l'encodage
            file_content = uploaded_file.read()
            
            # Essayer UTF-8 d'abord
            try:
                content_str = file_content.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError:
                content_str = file_content.decode('latin-1')
                encoding = 'latin-1'
            
            # Détecter le séparateur
            if ';' in content_str[:1000]:
                separator = ';'
            else:
                separator = ','
            
            # Rembobiner le fichier
            uploaded_file.seek(0)
            
            # Charger avec la fonction adaptée
            df = load_cpf_data(file_content, separator, encoding)
            df = clean_data(df)
            
            # Afficher les informations sur le fichier chargé
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Lignes", f"{len(df):,}")
            with col_info2:
                st.metric("Colonnes", len(df.columns))
            with col_info3:
                st.metric("Séparateur", f"'{separator}'")
            with col_info4:
                st.metric("Encodage", encoding)
            
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès !")
            
            # Afficher un aperçu des données
            with st.expander("👁️ Aperçu des données (5 premières lignes)"):
                st.dataframe(df.head(), width='stretch')
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier : {e}")
            st.info("Vérifiez que votre fichier respecte le format attendu.")
            return
    
    if df.empty:
        st.error("❌ Le fichier chargé ne contient pas de données valides.")
        return
    
    # Sidebar pour les filtres
    st.sidebar.header("🔍 Filtres")
    
    # Filtre par organisme de formation
    all_organisms = ['Tous'] + sorted(df['raison_sociale_of_contractant'].unique())
    selected_organism = st.sidebar.selectbox(
        "Organisme de formation:",
        all_organisms,
        index=all_organisms.index('AFPA ENTREPRISES') if 'AFPA ENTREPRISES' in all_organisms else 0
    )
    
    # Filtre par année
    years = sorted(df['annee'].unique())
    selected_years = st.sidebar.multiselect(
        "Années:",
        years,
        default=years
    )
    
    # Filtre par type de référentiel
    ref_types = ['Tous'] + sorted(df['type_referentiel'].unique())
    selected_ref_type = st.sidebar.selectbox(
        "Type de référentiel:",
        ref_types
    )
    
    # Filtre par intitulé de certification avec recherche
    st.sidebar.subheader("🎓 Certification")
    
    # Champ de recherche pour les certifications
    search_certification = st.sidebar.text_input(
        "Rechercher une certification:",
        placeholder="Tapez pour rechercher...",
        help="Recherche dans les noms de certification"
    )
    
    # Filtrer les certifications selon la recherche
    all_certifications_raw = sorted(df['intitule_certification'].unique())
    if search_certification:
        filtered_certifications = [cert for cert in all_certifications_raw 
                                 if search_certification.lower() in cert.lower()]
        all_certifications = ['Toutes'] + filtered_certifications
    else:
        all_certifications = ['Toutes'] + all_certifications_raw
    
    selected_certification = st.sidebar.selectbox(
        "Intitulé de certification:",
        all_certifications,
        help="Sélectionnez une certification spécifique pour filtrer les données"
    )
    
    # Afficher le nombre de certifications trouvées
    if search_certification:
        st.sidebar.caption(f"🔍 {len(all_certifications)-1} certification(s) trouvée(s)")
    
    # Application des filtres
    filtered_df = df.copy()
    
    if selected_organism != 'Tous':
        filtered_df = filtered_df[filtered_df['raison_sociale_of_contractant'] == selected_organism]
    
    if selected_years:
        filtered_df = filtered_df[filtered_df['annee'].isin(selected_years)]
    
    if selected_ref_type != 'Tous':
        filtered_df = filtered_df[filtered_df['type_referentiel'] == selected_ref_type]
    
    if selected_certification != 'Toutes':
        filtered_df = filtered_df[filtered_df['intitule_certification'] == selected_certification]
    
    st.sidebar.markdown(f"**Données filtrées:** {len(filtered_df):,} lignes")
    
    # Affichage des KPIs
    kpis = calculate_kpis(filtered_df)
    
    if kpis:
        st.subheader("📈 KPIs Principaux")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Entrées Formation",
                value=f"{kpis['total_entrees']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Sorties Réussite Totale",
                value=f"{kpis['total_sorties_totales']:,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Taux de Réussite",
                value=f"{kpis['taux_reussite']:.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Nb Certifications",
                value=f"{kpis['nb_certifications']:,}",
                delta=None
            )
        
        # Deuxième ligne de KPIs
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                label="Sorties Partielles",
                value=f"{kpis['total_sorties_partielles']:,}",
                delta=None
            )
        
        with col6:
            st.metric(
                label="Nb Organismes",
                value=f"{kpis['nb_organismes']:,}",
                delta=None
            )
        
        with col7:
            taux_abandon = ((kpis['total_entrees'] - kpis['total_sorties_totales'] - kpis['total_sorties_partielles']) / kpis['total_entrees'] * 100) if kpis['total_entrees'] > 0 else 0
            st.metric(
                label="Taux d'Abandon",
                value=f"{taux_abandon:.1f}%",
                delta=None
            )
        
        with col8:
            taux_partiel = (kpis['total_sorties_partielles'] / kpis['total_entrees'] * 100) if kpis['total_entrees'] > 0 else 0
            st.metric(
                label="Taux Réussite Partielle",
                value=f"{taux_partiel:.1f}%",
                delta=None
            )
    
    # Graphiques
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📊 Top Certifications")
        if not filtered_df.empty:
            fig_cert = create_certification_chart(filtered_df)
            st.plotly_chart(fig_cert, width='stretch')
    
    with col_chart2:
        st.subheader("📈 Tendance Mensuelle")
        if not filtered_df.empty:
            fig_trend = create_monthly_trend(filtered_df)
            st.plotly_chart(fig_trend, width='stretch')
    
    # Tableau détaillé
    st.markdown("---")
    st.subheader("📋 Données Détaillées")
    
    if not filtered_df.empty:
        # Agrégation par certification
        detail_data = filtered_df.groupby(['intitule_certification', 'raison_sociale_of_contractant']).agg({
            'entrees_formation': 'sum',
            'sorties_realisation_totale': 'sum',
            'sorties_realisation_partielle': 'sum'
        }).reset_index()
        
        detail_data['taux_reussite'] = (detail_data['sorties_realisation_totale'] / detail_data['entrees_formation'] * 100).round(1)
        
        st.dataframe(
            detail_data.sort_values('entrees_formation', ascending=False),
            width='stretch'
        )
        
        # Option de téléchargement
        csv = detail_data.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"cpf_analyse_{selected_organism}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Informations sur la certification sélectionnée
    if selected_certification != 'Toutes':
        st.markdown("---")
        st.subheader(f"🎓 Focus Certification: {selected_certification}")
        
        cert_data = filtered_df
        if not cert_data.empty:
            col_cert1, col_cert2, col_cert3 = st.columns(3)
            
            with col_cert1:
                st.metric("Organismes proposant cette certification", 
                         cert_data['raison_sociale_of_contractant'].nunique())
            
            with col_cert2:
                cert_entries = cert_data['entrees_formation'].sum()
                cert_success = cert_data['sorties_realisation_totale'].sum()
                cert_success_rate = (cert_success / cert_entries * 100) if cert_entries > 0 else 0
                st.metric("Taux de réussite certification", f"{cert_success_rate:.1f}%")
            
            with col_cert3:
                st.metric("Total entrées cette certification", f"{cert_entries:,}")
            
            # Top organismes pour cette certification
            if len(cert_data) > 1:
                st.write("**Top organismes pour cette certification:**")
                top_orgs_cert = cert_data.groupby('raison_sociale_of_contractant')['entrees_formation'].sum().sort_values(ascending=False).head(5)
                for org, count in top_orgs_cert.items():
                    st.write(f"• {org}: {count:,} entrées")
    
    # Informations sur AFPA ENTREPRISES si sélectionné
    if selected_organism == 'AFPA ENTREPRISES':
        st.markdown("---")
        st.subheader("🎯 Focus AFPA ENTREPRISES")
        
        afpa_data = filtered_df
        
        if not afpa_data.empty:
            # Top 5 des certifications AFPA
            top_afpa_cert = afpa_data.groupby('intitule_certification')['entrees_formation'].sum().sort_values(ascending=False).head()
            
            col_afpa1, col_afpa2 = st.columns(2)
            
            with col_afpa1:
                st.write("**Top 5 Certifications AFPA:**")
                for cert, count in top_afpa_cert.items():
                    st.write(f"• {cert}: {count:,} entrées")
            
            with col_afpa2:
                # Répartition par type de référentiel
                ref_repartition = afpa_data.groupby('type_referentiel')['entrees_formation'].sum()
                fig_pie = px.pie(
                    values=ref_repartition.values,
                    names=ref_repartition.index,
                    title="Répartition par type de référentiel"
                )
                st.plotly_chart(fig_pie, width='stretch')
    
    # Analyse concurrentielle AFPA
    st.markdown("---")
    st.subheader("🏆 Analyse Concurrentielle AFPA ENTREPRISES")
    
    # Identifier les certifications où AFPA n'est pas présent
    afpa_certifications = set(df[df['raison_sociale_of_contractant'] == 'AFPA ENTREPRISES']['intitule_certification'].unique())
    all_certifications = set(df['intitule_certification'].unique())
    missing_certifications = all_certifications - afpa_certifications
    
    # Analyser les opportunités manquées (certifications populaires sans AFPA)
    missing_opportunities = df[df['intitule_certification'].isin(missing_certifications)]
    top_missing = missing_opportunities.groupby('intitule_certification')['entrees_formation'].sum().sort_values(ascending=False).head(10)
    
    # Analyser les concurrents principaux
    competitors_analysis = df[df['raison_sociale_of_contractant'] != 'AFPA ENTREPRISES'].groupby('raison_sociale_of_contractant').agg({
        'entrees_formation': 'sum',
        'sorties_realisation_totale': 'sum',
        'intitule_certification': 'nunique'
    }).reset_index()
    
    competitors_analysis['taux_reussite'] = (competitors_analysis['sorties_realisation_totale'] / 
                                           competitors_analysis['entrees_formation'] * 100).round(1)
    
    top_competitors = competitors_analysis.sort_values('entrees_formation', ascending=False).head(10)
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.subheader("🎯 Opportunités Manquées")
        st.write("**Top 10 certifications populaires où AFPA n'est pas présent:**")
        
        if not top_missing.empty:
            for i, (cert, entries) in enumerate(top_missing.items(), 1):
                # Trouver le leader sur cette certification
                cert_leaders = df[df['intitule_certification'] == cert].groupby('raison_sociale_of_contractant')['entrees_formation'].sum().sort_values(ascending=False).head(1)
                leader = cert_leaders.index[0] if len(cert_leaders) > 0 else "N/A"
                leader_entries = cert_leaders.iloc[0] if len(cert_leaders) > 0 else 0
                
                with st.expander(f"{i}. {cert[:50]}... ({entries:,} entrées totales)"):
                    st.write(f"**Leader:** {leader}")
                    st.write(f"**Entrées leader:** {leader_entries:,}")
                    
                    # Top 3 organismes sur cette certification
                    top_orgs = df[df['intitule_certification'] == cert].groupby('raison_sociale_of_contractant')['entrees_formation'].sum().sort_values(ascending=False).head(3)
                    st.write("**Top 3 organismes:**")
                    for j, (org, count) in enumerate(top_orgs.items(), 1):
                        st.write(f"  {j}. {org}: {count:,}")
    
    with col_comp2:
        st.subheader("🏢 Principaux Concurrents")
        st.write("**Top 10 concurrents par volume d'entrées:**")
        
        for i, row in top_competitors.iterrows():
            with st.expander(f"{row.name + 1}. {row['raison_sociale_of_contractant'][:40]}..."):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Entrées", f"{int(row['entrees_formation']):,}")
                with col_b:
                    st.metric("Taux réussite", f"{row['taux_reussite']}%")
                with col_c:
                    st.metric("Nb certifs", f"{int(row['intitule_certification'])}")
    
    # Analyse comparative détaillée
    st.subheader("📊 Comparaison AFPA vs Concurrents")
    
    # Obtenir les stats AFPA
    afpa_stats = df[df['raison_sociale_of_contractant'] == 'AFPA ENTREPRISES'].agg({
        'entrees_formation': 'sum',
        'sorties_realisation_totale': 'sum',
        'intitule_certification': 'nunique'
    })
    afpa_taux = (afpa_stats['sorties_realisation_totale'] / afpa_stats['entrees_formation'] * 100).round(1)
    
    col_comp_a, col_comp_b, col_comp_c, col_comp_d = st.columns(4)
    
    with col_comp_a:
        st.metric(
            "Position AFPA (Entrées)",
            value=f"#{df.groupby('raison_sociale_of_contractant')['entrees_formation'].sum().rank(ascending=False, method='min').get('AFPA ENTREPRISES', 'N/A'):.0f}",
            help="Position d'AFPA ENTREPRISES par rapport aux autres organismes"
        )
    
    with col_comp_b:
        market_share = (afpa_stats['entrees_formation'] / df['entrees_formation'].sum() * 100).round(2)
        st.metric(
            "Part de marché AFPA",
            value=f"{market_share}%",
            help="Pourcentage des entrées totales du marché"
        )
    
    with col_comp_c:
        avg_competitor_rate = competitors_analysis['taux_reussite'].mean().round(1)
        delta_rate = afpa_taux - avg_competitor_rate
        st.metric(
            "Taux vs Concurrents",
            value=f"{afpa_taux}%",
            delta=f"{delta_rate:+.1f}% vs moyenne",
            help="Taux de réussite AFPA vs moyenne des concurrents"
        )
    
    with col_comp_d:
        avg_certifs = competitors_analysis['intitule_certification'].mean().round(0)
        delta_certifs = afpa_stats['intitule_certification'] - avg_certifs
        st.metric(
            "Diversité vs Concurrents",
            value=f"{int(afpa_stats['intitule_certification'])}",
            delta=f"{delta_certifs:+.0f} vs moyenne",
            help="Nombre de certifications AFPA vs moyenne des concurrents"
        )
    
    # Graphique de positionnement concurrentiel
    st.subheader("📈 Positionnement Concurrentiel")
    
    # Créer un scatter plot des concurrents
    fig_scatter = go.Figure()
    
    # Points des concurrents
    fig_scatter.add_trace(go.Scatter(
        x=competitors_analysis['entrees_formation'],
        y=competitors_analysis['taux_reussite'],
        mode='markers',
        name='Concurrents',
        text=competitors_analysis['raison_sociale_of_contractant'],
        hovertemplate='<b>%{text}</b><br>Entrées: %{x:,}<br>Taux réussite: %{y}%<extra></extra>',
        marker=dict(size=8, color='lightblue', opacity=0.6)
    ))
    
    # Point AFPA
    fig_scatter.add_trace(go.Scatter(
        x=[afpa_stats['entrees_formation']],
        y=[afpa_taux],
        mode='markers',
        name='AFPA ENTREPRISES',
        text=['AFPA ENTREPRISES'],
        hovertemplate='<b>%{text}</b><br>Entrées: %{x:,}<br>Taux réussite: %{y}%<extra></extra>',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig_scatter.update_layout(
        title="Position AFPA vs Concurrents (Volume vs Performance)",
        xaxis_title="Nombre d'entrées en formation",
        yaxis_title="Taux de réussite (%)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_scatter, width='stretch')
    
    # Graphique de comparaison des concurrents
    st.subheader("🏢 Comparaison des Top Organismes")
    fig_competitors = create_competitor_comparison_chart(df)
    st.plotly_chart(fig_competitors, width='stretch')
    
    # Analyse par segment de marché
    st.subheader("🎯 Analyse par Segment de Marché")
    segments = analyze_market_segments(df)
    
    # Créer un tableau de synthèse des segments
    segment_summary = []
    for ref_type, data in segments.items():
        segment_summary.append({
            'Type de Référentiel': ref_type,
            'Taille du Marché (entrées)': f"{data['total_entries']:,}",
            'Nb Organismes': data['nb_organisms'],
            'Présence AFPA': '✅' if data['afpa_presence'] else '❌',
            'Entrées AFPA': f"{data['afpa_entries']:,}" if data['afpa_presence'] else '0',
            'Leader du Segment': list(data['leaders'].keys())[0] if data['leaders'] else 'N/A',
            'Entrées Leader': f"{list(data['leaders'].values())[0]:,}" if data['leaders'] else '0'
        })
    
    segment_df = pd.DataFrame(segment_summary)
    segment_df = segment_df.sort_values('Taille du Marché (entrées)', ascending=False, key=lambda x: x.str.replace(',', '').astype(int))
    
    st.dataframe(segment_df, width='stretch')
    
    # Identifier les segments d'opportunité
    opportunity_segments = []
    for ref_type, data in segments.items():
        if not data['afpa_presence'] and data['total_entries'] > 1000:  # Segments sans AFPA avec plus de 1000 entrées
            opportunity_segments.append({
                'segment': ref_type,
                'entries': data['total_entries'],
                'leader': list(data['leaders'].keys())[0] if data['leaders'] else 'N/A',
                'leader_entries': list(data['leaders'].values())[0] if data['leaders'] else 0
            })
    
    if opportunity_segments:
        st.warning("⚠️ **Segments d'opportunité identifiés** (>1000 entrées, sans AFPA)")
        opportunity_df = pd.DataFrame(opportunity_segments)
        opportunity_df = opportunity_df.sort_values('entries', ascending=False)
        
        for _, opp in opportunity_df.head(5).iterrows():
            st.write(f"🎯 **{opp['segment']}**: {opp['entries']:,} entrées/an (Leader: {opp['leader']}, {opp['leader_entries']:,} entrées)")

if __name__ == "__main__":
    main()