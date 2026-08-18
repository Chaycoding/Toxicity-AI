from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import matplotlib.pyplot as plt

st.set_page_config(page_title="DILI Prediction Engine", layout="wide")

st.markdown("""
    <style>
    /* Dark Theme Core Adjustments */
    .stApp {
        background-color: #050B14 !important;
        color: #f1f5f9 !important;
    }
    
    /* Premium Glassmorphic Cards for Lipinski Metrics */
    [data-testid="stMetric"] {
        background: rgba(11, 18, 33, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        padding: 24px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    
    /* Precise Hover Accent Shift instead of aggressive bounce */
    [data-testid="stMetric"]:hover {
        border-color: rgba(14, 165, 233, 0.4) !important;
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.1) !important;
    }
    
    /* Technical Monospace Labels (e.g., "MOLECULAR WEIGHT") */
    [data-testid="stMetricLabel"] * {
        color: #94a3b8 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }

    /* Clear High-Contrast Values */
    [data-testid="stMetricValue"] * {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 28px !important;
    }
    
    /* Clean up input borders to match slate borders */
    div[data-baseweb="input"] {
        background-color: #0B1221 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
    }
    
    input {
        color: #f1f5f9 !important;
    }
    
    hr {
        border-color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('dili_ensemble_model.joblib')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Error: 'dili_ensemble_model.joblib' not found. Please ensure the model is exported and in the same folder.")
    model_loaded = False

# -----------------------------------------------------------------------------
# Feature Extraction Pipeline
def extract_features(smiles):
    """
    Converts SMILES to an RDKit Mol object and extracts the 2048-bit fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    fp_array = list(fp)
    features_df = pd.DataFrame([fp_array])
    
    # Calculate Complete Lipinski's Rules + TPSA strictly for the UI
    thermo_metrics = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),      # Lipinski Rule: <= 5
        'HBA': Descriptors.NumHAcceptors(mol),   # Lipinski Rule: <= 10
        'TPSA': Descriptors.TPSA(mol)
    }
    
    return features_df, thermo_metrics

# -----------------------------------------------------------------------------
# User Interface & Main App Logic
# -----------------------------------------------------------------------------
# Technical status header mimicking the portfolio style
st.markdown("""
    <span style="inline-flex; items-center; gap: 8px; padding: 4px 12px; border-radius: 9999px; background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.2); color: #0ea5e9; font-family: monospace; font-size: 11px; tracking-content: 0.15em; text-transform: uppercase;">
        ● Computational Inference Node Active
    </span>
    """, unsafe_allow_html=True)

st.title("DILI Assessment Engine")
st.markdown("Predictive toxicology model screening structural topologies for drug-induced liver injury risks.")

st.markdown("---")

st.subheader("Chemical Input Structure")
smiles_input = st.text_input("Enter SMILES String:", placeholder="e.g., CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")

if smiles_input and model_loaded:
    with st.spinner('Extracting structural properties and processing graphs...'):
        
        # Run extraction pipeline
        features_df, thermo_metrics = extract_features(smiles_input)
        
        if features_df is None:
            st.error("Invalid SMILES topology string. Unable to map chemical graph structure.")
        else:
            mol = Chem.MolFromSmiles(smiles_input)
            img = Draw.MolToImage(mol, size=(350, 350)) 
            
            st.markdown("<br>", unsafe_allow_html=True)
            top_col1, top_col2 = st.columns([1, 2.5]) 
            
            with top_col1:
                st.subheader("2D Graph Representation")
                st.image(img, use_container_width=False) 
                
            with top_col2:
                st.subheader("Model Predictive Verdict")
                
                proba = model.predict_proba(features_df)[0]
                toxic_proba = proba[1] * 100  
                
                st.markdown(f"### Probability Vector: <span style='color:#0ea5e9;'>{toxic_proba:.1f}%</span>", unsafe_allow_html=True)
                st.progress(int(toxic_proba))
                
                st.markdown("<br>", unsafe_allow_html=True)
                if toxic_proba >= 50.0:
                    st.markdown("""
                        <div style="background: rgba(231, 76, 60, 0.1); border: 1px solid rgba(231, 76, 60, 0.3); padding: 16px; border-radius: 12px; color: #fca5a5;">
                            🛑 <strong>CRITICAL RISK:</strong> High correlation detected with known topological motifs triggering Drug-Induced Liver Injury (DILI).
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="background: rgba(46, 204, 113, 0.1); border: 1px solid rgba(46, 204, 113, 0.3); padding: 16px; border-radius: 12px; color: #a7f3d0;">
                            ✅ <strong>CLEARANCE APPROVED:</strong> Molecular sequence passes baseline hepatic toxicology classification metrics.
                        </div>
                        """, unsafe_allow_html=True)
            
            # -----------------------------------------------------------------
            # BOTTOM ROW: Unified Lipinski Profiler
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("Pharmacokinetic & ADME Profiling")
            
            met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
            
            with met_col1:
                st.metric("Molecular Wt.", f"{thermo_metrics['MW']:.1f} Da")
            with met_col2:
                st.metric("LogP (Lipophilicity)", f"{thermo_metrics['LogP']:.2f}")
            with met_col3:
                st.metric("H-Bond Donors", f"{thermo_metrics['HBD']}")
            with met_col4:
                st.metric("H-Bond Acceptors", f"{thermo_metrics['HBA']}")
            with met_col5:
                st.metric("TPSA (Polarity)", f"{thermo_metrics['TPSA']:.1f} Å²")
            
st.markdown("---")
st.caption("Integrated ML Pipeline Engine | RDKit & Scikit-Learn Validation Pipeline | Evaluation Matrix: AUC-ROC")