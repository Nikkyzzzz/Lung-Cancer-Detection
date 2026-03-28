import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
import shutil

from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

DRIVE_MODELS_FOLDER_URL = "https://drive.google.com/drive/folders/16BN3OgZ6OGRNXGq8DSywC8SUu6WXi3X6?usp=sharing"

# Show the page title and description.
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --bg-primary: #060b12;
        --bg-secondary: #0d1622;
        --bg-panel: #122033;
        --text-primary: #e6f4ff;
        --text-muted: #c6d7e8;
        --accent: #2dd4bf;
        --accent-2: #0ea5e9;
        --danger: #f87171;
    }

    @keyframes heroIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.35); }
        70% { box-shadow: 0 0 0 10px rgba(45, 212, 191, 0); }
        100% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0); }
    }

    @keyframes signalMove {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    @keyframes sectionIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes gradientDrift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes orbFloatA {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(20px, -16px, 0) scale(1.08); }
        100% { transform: translate3d(0, 0, 0) scale(1); }
    }

    @keyframes orbFloatB {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(-24px, 14px, 0) scale(1.05); }
        100% { transform: translate3d(0, 0, 0) scale(1); }
    }

    @keyframes titleShimmer {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    @keyframes borderPulse {
        0% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.28); }
        70% { box-shadow: 0 0 0 10px rgba(45, 212, 191, 0); }
        100% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0); }
    }

    @keyframes resultPop {
        0% { opacity: 0; transform: translateY(8px) scale(0.98); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes resultGlowAlert {
        0% { box-shadow: 0 0 0 rgba(248, 113, 113, 0.0); }
        50% { box-shadow: 0 0 28px rgba(248, 113, 113, 0.22); }
        100% { box-shadow: 0 0 0 rgba(248, 113, 113, 0.0); }
    }

    @keyframes resultGlowSafe {
        0% { box-shadow: 0 0 0 rgba(45, 212, 191, 0.0); }
        50% { box-shadow: 0 0 24px rgba(45, 212, 191, 0.2); }
        100% { box-shadow: 0 0 0 rgba(45, 212, 191, 0.0); }
    }

    @keyframes loadBar {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(180%); }
    }

    @keyframes chartRise {
        0% { opacity: 0; transform: translateY(12px) scale(0.985); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes borderSweep {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(14, 165, 233, 0.18), transparent 32%),
            radial-gradient(circle at 92% 18%, rgba(45, 212, 191, 0.14), transparent 28%),
            linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
        background-size: 180% 180%;
        animation: gradientDrift 24s ease infinite;
        color: var(--text-primary);
    }

    .bg-orbs {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .bg-orbs span {
        position: absolute;
        border-radius: 999px;
        filter: blur(2px);
        opacity: 0.4;
    }

    .bg-orb-a {
        width: 240px;
        height: 240px;
        top: 12%;
        left: -40px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.42), rgba(14, 165, 233, 0));
        animation: orbFloatA 10s ease-in-out infinite;
    }

    .bg-orb-b {
        width: 300px;
        height: 300px;
        bottom: 6%;
        right: -60px;
        background: radial-gradient(circle, rgba(45, 212, 191, 0.38), rgba(45, 212, 191, 0));
        animation: orbFloatB 12s ease-in-out infinite;
    }

    .bg-orb-c {
        width: 180px;
        height: 180px;
        top: 58%;
        left: 45%;
        background: radial-gradient(circle, rgba(125, 248, 235, 0.22), rgba(125, 248, 235, 0));
        animation: orbFloatA 14s ease-in-out infinite;
    }

    .main .block-container,
    [data-testid="block-container"],
    .stMainBlockContainer {
        max-width: none !important;
        width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08111d, #0a1728);
        border-right: 1px solid rgba(45, 212, 191, 0.2);
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text-primary);
    }

    [data-testid="stMarkdownContainer"] p {
        color: var(--text-muted);
        line-height: 1.55;
    }

    .med-hero {
        background: linear-gradient(125deg, rgba(14, 165, 233, 0.24), rgba(45, 212, 191, 0.2));
        border: 1px solid rgba(45, 212, 191, 0.42);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin: 0.2rem 0 1rem 0;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        animation: heroIn 0.75s ease-out;
        position: relative;
        overflow: hidden;
    }

    .med-hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(200, 255, 248, 0.14), transparent);
        animation: signalMove 2.8s linear infinite;
        pointer-events: none;
        mix-blend-mode: screen;
    }

    .med-badge {
        display: inline-block;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #b8fff7;
        border: 1px solid rgba(45, 212, 191, 0.4);
        border-radius: 999px;
        padding: 0.18rem 0.7rem;
        margin-bottom: 0.45rem;
        background: rgba(45, 212, 191, 0.08);
        animation: pulseGlow 2.4s infinite;
    }

    .med-title {
        font-size: clamp(1.8rem, 3.2vw, 2.7rem);
        line-height: 1.18;
        margin: 0;
        background: linear-gradient(90deg, #f7fdff, #c0f6ff, #f7fdff);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: titleShimmer 4s linear infinite;
    }

    .med-sub {
        margin-top: 0.55rem;
        color: #d7e7f5;
    }

    .med-chip-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.5rem 0 0.8rem 0;
    }

    .med-chip {
        font-size: 0.78rem;
        color: #d5f6ff;
        border: 1px solid rgba(14, 165, 233, 0.45);
        background: rgba(14, 165, 233, 0.12);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        backdrop-filter: blur(2px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .med-chip:hover {
        transform: translateY(-1px);
        border-color: rgba(125, 248, 235, 0.78);
        box-shadow: 0 8px 18px rgba(14, 165, 233, 0.24);
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.4rem 0 1rem 0;
    }

    .feature-card {
        background: linear-gradient(145deg, rgba(11, 23, 38, 0.92), rgba(10, 20, 34, 0.8));
        border: 1px solid rgba(45, 212, 191, 0.24);
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.22s ease, border-color 0.22s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        border-color: rgba(125, 248, 235, 0.6);
    }

    .feature-card-title {
        color: #d6f7ff;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }

    .feature-card-sub {
        color: #b9d2e8;
        font-size: 0.78rem;
    }

    .section-banner {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.5rem 0 0.8rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #e8f6ff;
    }

    .section-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: linear-gradient(180deg, #2dd4bf, #0ea5e9);
        animation: borderPulse 2s infinite;
    }

    .model-health-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.55rem 0 0.6rem 0;
    }

    .health-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.74rem;
        border-radius: 999px;
        padding: 0.2rem 0.55rem;
        border: 1px solid rgba(125, 248, 235, 0.45);
        color: #dff8ff;
        background: rgba(8, 28, 41, 0.85);
    }

    .health-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
    }

    .health-ready .health-dot { background: #2dd4bf; }
    .health-warn .health-dot { background: #f59e0b; }
    .health-cpu .health-dot { background: #60a5fa; }
    .health-gpu .health-dot { background: #34d399; }

    .inference-loader {
        position: relative;
        margin: 0.35rem 0 0.7rem 0;
        border-radius: 10px;
        border: 1px solid rgba(14, 165, 233, 0.35);
        background: rgba(7, 20, 33, 0.82);
        overflow: hidden;
    }

    .inference-loader-text {
        padding: 0.52rem 0.72rem 0.38rem 0.72rem;
        color: #d7eefb;
        font-size: 0.84rem;
    }

    .inference-loader-track {
        height: 4px;
        background: rgba(30, 58, 85, 0.6);
        overflow: hidden;
    }

    .inference-loader-bar {
        height: 100%;
        width: 55%;
        background: linear-gradient(90deg, #2dd4bf, #0ea5e9, #2dd4bf);
        animation: loadBar 1.2s linear infinite;
        border-radius: 999px;
    }

    .result-card {
        border-radius: 14px;
        border: 1px solid rgba(142, 222, 255, 0.26);
        padding: 0.9rem 1rem;
        margin: 0.35rem 0 0.7rem 0;
        background: linear-gradient(140deg, rgba(11, 26, 41, 0.95), rgba(8, 19, 31, 0.88));
        animation: resultPop 0.35s ease-out;
    }

    .result-card .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.5rem;
    }

    .result-title {
        color: #e8f8ff;
        font-weight: 700;
        font-size: 1.03rem;
    }

    .result-confidence {
        font-size: 0.78rem;
        border: 1px solid rgba(142, 222, 255, 0.35);
        padding: 0.16rem 0.48rem;
        border-radius: 999px;
        color: #dff6ff;
        white-space: nowrap;
    }

    .result-msg {
        color: #d7e8f7;
        line-height: 1.5;
    }

    .result-card.malignant {
        border-color: rgba(248, 113, 113, 0.55);
        background: linear-gradient(145deg, rgba(56, 18, 25, 0.84), rgba(34, 11, 18, 0.9));
        animation: resultPop 0.35s ease-out, resultGlowAlert 2.4s ease-in-out infinite;
    }

    .result-card.benign,
    .result-card.normal {
        border-color: rgba(45, 212, 191, 0.55);
        background: linear-gradient(145deg, rgba(8, 45, 40, 0.84), rgba(7, 31, 35, 0.9));
        animation: resultPop 0.35s ease-out, resultGlowSafe 2.6s ease-in-out infinite;
    }

    .result-card.other {
        border-color: rgba(14, 165, 233, 0.55);
    }

    .viz-section {
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
    }

    .viz-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #e8f6ff;
        margin-bottom: 0.2rem;
    }

    .viz-sub {
        color: #b7d4ea;
        margin-bottom: 0.7rem;
    }

    .chart-card {
        position: relative;
        border-radius: 14px;
        border: 1px solid rgba(125, 248, 235, 0.2);
        background: linear-gradient(145deg, rgba(9, 23, 38, 0.9), rgba(8, 18, 30, 0.84));
        padding: 0.55rem 0.55rem 0.35rem 0.55rem;
        overflow: hidden;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.26);
        animation: chartRise 0.5s ease both;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    
    .chart-caption {
        color: #cde4f6;
        font-size: 0.8rem;
        margin: 0.2rem 0.3rem 0.25rem 0.3rem;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: rgba(8, 16, 28, 0.96) !important;
        border: 1px solid rgba(45, 212, 191, 0.46) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 0 0 1px rgba(14, 165, 233, 0.12) inset;
    }

    [data-baseweb="select"] [role="option"],
    [data-baseweb="select"] [role="listbox"] {
        background-color: #0a1322 !important;
        color: #e7f4ff !important;
    }

    /* Dropdown popovers are rendered in a portal outside the select container */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="popover"] [role="option"] {
        background-color: #091321 !important;
        color: #eaf5ff !important;
        border-color: rgba(45, 212, 191, 0.4) !important;
    }

    div[data-baseweb="popover"] [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(14, 127, 194, 0.55), rgba(10, 162, 217, 0.5)) !important;
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] [role="option"]:hover {
        background-color: rgba(14, 165, 233, 0.25) !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(12, 24, 40, 0.95), rgba(9, 19, 32, 0.95)) !important;
        border: 1px solid rgba(45, 212, 191, 0.46) !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        animation: borderPulse 3.2s infinite;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(125, 248, 235, 0.88) !important;
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.2);
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #e6f4ff !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: #b6d6ef !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #dff5ff, #c7efff) !important;
        color: #00324a !important;
        border: 1px solid rgba(4, 68, 99, 0.25) !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(90deg, #ecf9ff, #d7f5ff) !important;
        color: #00273b !important;
    }

    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"] {
        color: #dceeff !important;
        opacity: 1 !important;
        font-weight: 600;
    }

    [data-testid="stFileUploaderFile"] * {
        color: #dceeff !important;
    }

    [data-baseweb="tag"] {
        background: linear-gradient(90deg, #0e7fc2, #0aa2d9) !important;
        border: 1px solid rgba(142, 222, 255, 0.45) !important;
        color: #eef9ff !important;
    }

    [data-baseweb="tag"] * {
        color: #eef9ff !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent) !important;
    }

    .stButton button {
        background: linear-gradient(90deg, #1290d8, #0fb7ac);
        border: none;
        color: #f8fcff;
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.2s ease, filter 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35);
    }

    .stButton button:hover {
        transform: translateY(-1px);
        filter: brightness(1.12);
        box-shadow: 0 10px 24px rgba(45, 212, 191, 0.35);
    }

    .stButton button:disabled {
        opacity: 0.65;
        box-shadow: none;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(45, 212, 191, 0.25);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
    }

    /* Chart corner controls (view data / fullscreen) */
    [data-testid="stElementToolbar"] {
        background: linear-gradient(135deg, rgba(10, 23, 38, 0.95), rgba(8, 18, 31, 0.95)) !important;
        border: 1px solid rgba(45, 212, 191, 0.45) !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35) !important;
    }

    [data-testid="stElementToolbar"] button,
    [data-testid="stElementToolbar"] [role="button"] {
        color: #dff4ff !important;
        background: transparent !important;
        border-radius: 8px !important;
    }

    /* Force dark skin on individual chart action buttons */
    [data-testid="stElementToolbarButton"],
    [data-testid="stElementToolbar"] button[kind="icon"],
    [data-testid="stElementToolbar"] button[title],
    [data-testid="stElementToolbar"] button[aria-label],
    button[title*="View fullscreen"],
    button[title*="View data"],
    button[aria-label*="fullscreen"],
    button[aria-label*="data"] {
        background: linear-gradient(135deg, rgba(11, 24, 39, 0.98), rgba(8, 18, 31, 0.98)) !important;
        border: 1px solid rgba(45, 212, 191, 0.4) !important;
        color: #dff4ff !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    [data-testid="stElementToolbar"] button:hover,
    [data-testid="stElementToolbar"] [role="button"]:hover {
        background: rgba(14, 165, 233, 0.2) !important;
        color: #ffffff !important;
    }

    [data-testid="stElementToolbar"] svg,
    [data-testid="stElementToolbar"] path {
        fill: #dff4ff !important;
        color: #dff4ff !important;
        stroke: transparent !important;
    }

    [data-testid="stElementToolbarButton"]:hover,
    [data-testid="stElementToolbar"] button[kind="icon"]:hover,
    button[title*="View fullscreen"]:hover,
    button[title*="View data"]:hover,
    button[aria-label*="fullscreen"]:hover,
    button[aria-label*="data"]:hover {
        background: linear-gradient(135deg, rgba(19, 61, 89, 0.95), rgba(13, 80, 96, 0.9)) !important;
        border-color: rgba(125, 248, 235, 0.75) !important;
        color: #ffffff !important;
    }

    [data-testid="stElementToolbarButton"]:focus,
    [data-testid="stElementToolbar"] button:focus,
    [data-testid="stElementToolbarButton"]:focus-visible,
    [data-testid="stElementToolbar"] button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.35) !important;
    }

    /* Final fallback: hide chart corner actions (view data / fullscreen) */
    [data-testid="stVegaLiteChart"] [data-testid="stElementToolbar"],
    [data-testid="stVegaLiteChart"] [data-testid="stElementToolbarButton"],
    [data-testid="stVegaLiteChart"] button[title*="View fullscreen"],
    [data-testid="stVegaLiteChart"] button[title*="View data"],
    [data-testid="stVegaLiteChart"] button[aria-label*="fullscreen"],
    [data-testid="stVegaLiteChart"] button[aria-label*="data"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* Absolute fallback: disable Streamlit element toolbar globally */
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbarButton"],
    button[title*="View fullscreen"],
    button[title*="View data"],
    button[aria-label*="fullscreen"],
    button[aria-label*="data"] {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Dropdown menu opened from chart corner controls */
    div[data-baseweb="popover"] [role="menu"],
    div[data-baseweb="popover"] [role="menuitem"],
    div[data-baseweb="popover"] [data-testid="stMarkdownContainer"] {
        background: #0a1523 !important;
        color: #e9f6ff !important;
        border-color: rgba(45, 212, 191, 0.4) !important;
    }

    div[data-baseweb="popover"] [role="menuitem"]:hover {
        background: rgba(14, 165, 233, 0.2) !important;
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] [role="menuitem"] svg,
    div[data-baseweb="popover"] [role="menuitem"] path {
        fill: #dff4ff !important;
        color: #dff4ff !important;
    }

    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > div {
        animation: sectionIn 0.35s ease both;
    }

    [data-testid="stHeader"] {
        background: rgba(5, 12, 21, 0.6);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(45, 212, 191, 0.22);
    }

    /* Restore all Streamlit chrome: header, toolbar, menu, etc. */

    .stApp {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .main .block-container {
        padding-top: 1.2rem !important;
    }

    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .med-hero {
            padding: 0.9rem 1rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp,
        .med-hero,
        .med-badge,
        .med-title,
        .bg-orbs span,
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > div,
        [data-testid="stFileUploaderDropzone"],
        .result-card,
        .inference-loader-bar {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bg-orbs" aria-hidden="true">
        <span class="bg-orb-a"></span>
        <span class="bg-orb-b"></span>
        <span class="bg-orb-c"></span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="med-hero">
        <div class="med-badge">Clinical Decision Support</div>
        <h1 class="med-title">Lung Cancer Detection Dashboard</h1>
        <p class="med-sub">Leveraging Vision Transformers and Machine Learning for Early Lung Cancer Detection</p>
    </div>
    <div class="med-chip-row">
        <span class="med-chip">CT + Histopathology</span>
        <span class="med-chip">Hybrid AI Models</span>
        <span class="med-chip">Early Risk Guidance</span>
    </div>
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-card-title">Precision Pipeline</div>
            <div class="feature-card-sub">Transformer and CNN ensemble guidance for robust screening.</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-title">Clinical Clarity</div>
            <div class="feature-card-sub">Fast confidence and diagnosis outputs with reduced interaction friction.</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-title">Real-Time Feel</div>
            <div class="feature-card-sub">Smooth motion, hover feedback, and high-contrast medical theme.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    Lung cancer is a significant contributor to 
    cancer-related mortality. With recent advancements in 
    Computer Vision, Vision Transformers have gained traction 
    and shown remarkable success in medical image analysis. This 
    study explored the potential of Vision Transformer models (ViT, 
    CVT, CCT ViT, Parallel ViT, Efficient ViT) compared to 
    established state-of-the-art architectures (CNN) for lung 
    cancer detection via medical imaging modalities, including CT 
    and Histopathological scans. This work evaluated the impact of 
    data availability and different training approaches on model 
    performance. The training approaches included but were not 
    limited to, Supervised Learning and Transfer Learning. 
    Established evaluation metrics such as accuracy, recall, 
    precision, F1-score, and area under the ROC curve (AUC
    ROC) assessed model performance in terms of detection 
    efficacy, data validity, and computational efficiency. ViT 
    achieved an accuracy of 94% on a balanced dataset and an 
    accuracy of 87% on an imbalanced dataset trained from the 
    ground up. Cost-sensitive evaluation metrics, such as cost 
    matrix and weighted loss, analysed model performance by 
    considering the real-world implications of different types of 
    errors, especially in cases where misdiagnosing a cancer case 
    is far more critical.
    """ 
)

"---"
# --------------------------------------------------------------

# st.subheader("CT Scans of Lung Cancer")
# st.image("images/Lung Cancer Images/CT/CT.png", caption="Sample CT Scan Images Used for Model Training in Lung Cancer Detection")

# st.subheader("Histopathological Images of Lung Cancer")
# st.image("images/Lung Cancer Images/Histopathological/Histopathological.png", caption="Sample  Images Used for Model Training in Lung Cancer Detection")


# --------------------------------------------------------------

# Define image dimensions and preprocess function based on your model training
IMG_SIZE = (244, 244)  # Match to the input size your model was trained on

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        epsilon = tf.keras.backend.epsilon()  
        return 2 * ((precision * recall) / (precision + recall + epsilon))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
       
def preprocess_cnn_image(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0) # Add a batch dimension
    return image_array

# Preprocess image for PyTorch ViT models
def preprocess_vit_image(image, target_size):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

def get_vit_input_size(model):
    """Infer expected ViT input size from positional embeddings and patch dimensions."""
    if not hasattr(model, "pos_embedding"):
        return IMG_SIZE

    num_patches = model.pos_embedding.shape[1] - 1
    if num_patches <= 0:
        return IMG_SIZE

    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        return IMG_SIZE

    try:
        patch_module = model.to_patch_embedding[1]
        if hasattr(patch_module, "normalized_shape"):
            patch_dim = patch_module.normalized_shape[0]
        elif hasattr(patch_module, "in_features"):
            patch_dim = patch_module.in_features
        else:
            return IMG_SIZE

        channels = 3
        patch_size = int((patch_dim / channels) ** 0.5)
        image_size = grid_size * patch_size
        return (image_size, image_size)
    except Exception:
        return IMG_SIZE

def print_deduction(status, confidence=None, output=None):
    target = output if output is not None else st

    severity = "other"
    title = "Prediction"
    message = "Prediction complete."

    if status == 'Benign':
        severity = "benign"
        title = "Benign"
        message = "The image shows a benign case. No malignancy detected, but regular monitoring is advised."
    elif status == 'Malignant':
        severity = "malignant"
        title = "Malignant"
        message = "The image indicates a malignant lung cancer case. Immediate medical attention is recommended."
    elif status == 'Normal':
        severity = "normal"
        title = "Normal"
        message = "The image appears normal with no signs of lung cancer."
    elif status == "Malignant_ACA":
        severity = "malignant"
        title = "Malignant - Adenocarcinoma"
        message = "The image indicates an Adenocarcinoma (ACA) case. Further evaluation and treatment should be discussed with a healthcare professional."
    elif status == "Malignant_SCC":
        severity = "malignant"
        title = "Malignant - Squamous Cell Carcinoma"
        message = "The image indicates Squamous Cell Carcinoma (SCC). Prompt medical intervention is necessary, and treatment options should be explored with a specialist."

    confidence_text = f"{confidence:.2f}%" if confidence is not None else "N/A"
    target.markdown("#### Prediction Result")
    target.markdown(
        f"""
        <div class="result-card {severity}">
            <div class="result-header">
                <div class="result-title">{title}</div>
                <div class="result-confidence">Confidence: {confidence_text}</div>
            </div>
            <div class="result-msg">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_names = ["CNN Base Model", "CNN Hybrid Model", "ViT Base Model", "ViT CVT Model", "ViT Parallel Model"]

# Load Keras models based on user selection
model_paths = {
    "CNN Base Model": "models/cnn_model_2.keras",
    "CNN Hybrid Model": "models/cnn_model_2.keras",
    "ViT Base Model": "models/vit_ground_up_ct_model.pth",
    "ViT CVT Model": "models/vit_cvt_ground_up_ct_model.pth",
    "ViT Parallel Model": "models/vit_parallel_ground_up_ct_model.pth",
    "ViT Histopathological Model": "models/vit_ground_up_histopathological_model.pth"
}

_drive_models_download_attempted = False

def _download_models_folder_from_drive_once():
    """Attempt one-time download of missing artifacts from the shared Drive folder."""
    global _drive_models_download_attempted
    if _drive_models_download_attempted:
        return
    _drive_models_download_attempted = True

    try:
        import gdown
    except ImportError:
        st.error(
            "Missing dependency 'gdown'. Install it with: pip install gdown, "
            "then rerun the app to auto-download models from Google Drive."
        )
        return

    os.makedirs("models", exist_ok=True)
    try:
        with st.spinner("Downloading missing model files from Google Drive. This can take a few minutes on first run..."):
            gdown.download_folder(
                url=DRIVE_MODELS_FOLDER_URL,
                output="models",
                quiet=True,
                remaining_ok=True,
            )
    except Exception as exc:
        st.error(f"Could not download model files from Google Drive: {exc}")

def _ensure_local_artifact(file_path):
    if os.path.exists(file_path):
        return True
    _download_models_folder_from_drive_once()
    if not os.path.exists(file_path):
        st.error(
            f"Missing required file: {file_path}. "
            f"Please verify it exists in the Drive folder: {DRIVE_MODELS_FOLDER_URL}"
        )
        return False
    return True

def _build_vit_from_state_dict(state_dict):
    """Recreate the lucidrains-style ViT from a saved state_dict."""
    from vit_model import ViT

    if "pos_embedding" not in state_dict or "to_patch_embedding.1.weight" not in state_dict:
        raise ValueError("Unsupported checkpoint format: missing ViT keys.")

    pos_embedding = state_dict["pos_embedding"]
    patch_ln_weight = state_dict["to_patch_embedding.1.weight"]
    mlp_head_weight = state_dict["mlp_head.weight"]

    num_patches = pos_embedding.shape[1] - 1
    dim = pos_embedding.shape[2]
    patch_dim = patch_ln_weight.shape[0]
    channels = 3
    patch_size = int((patch_dim / channels) ** 0.5)
    image_size = int((num_patches ** 0.5) * patch_size)

    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.append(int(parts[2]))
    if not layer_indices:
        raise ValueError("Unsupported checkpoint format: cannot infer transformer depth.")
    depth = max(layer_indices) + 1

    qkv_weight = state_dict["transformer.layers.0.0.to_qkv.weight"]
    inner_dim = qkv_weight.shape[0] // 3
    dim_head = 64
    heads = max(1, inner_dim // dim_head)

    mlp_dim = state_dict["transformer.layers.0.1.net.1.weight"].shape[0]
    num_classes = mlp_head_weight.shape[0]

    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dim_head=dim_head
    )
    model.load_state_dict(state_dict)
    return model

def _build_cvt_from_state_dict(state_dict):
    """Recreate the project CvT variant from a saved state_dict."""
    from cvt_model import CvT

    num_classes = state_dict["to_logits.2.weight"].shape[0]

    model = CvT(
        num_classes=num_classes,
        s1_emb_dim=128,
        s1_emb_kernel=7,
        s1_emb_stride=4,
        s1_proj_kernel=3,
        s1_kv_proj_stride=2,
        s1_heads=2,
        s1_depth=2,
        s1_mlp_mult=4,
        s2_emb_dim=256,
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=4,
        s2_depth=2,
        s2_mlp_mult=4,
        s3_emb_dim=512,
        s3_emb_kernel=3,
        s3_emb_stride=2,
        s3_proj_kernel=3,
        s3_kv_proj_stride=2,
        s3_heads=10,
        s3_depth=2,
        s3_mlp_mult=4,
        dropout=0.1,
        channels=3
    )
    model.load_state_dict(state_dict)
    return model

def _build_parallel_vit_from_state_dict(state_dict):
    """Recreate the project ParallelViT variant from a saved state_dict."""
    from parallel_vit_model import ParallelViT

    pos_embedding = state_dict["pos_embedding"]
    num_patches = pos_embedding.shape[1] - 1
    dim = pos_embedding.shape[2]

    patch_linear_weight = state_dict["to_patch_embedding.1.weight"]
    patch_dim = patch_linear_weight.shape[1]
    channels = 3
    patch_size = int((patch_dim / channels) ** 0.5)
    image_size = int((num_patches ** 0.5) * patch_size)

    layer_indices = []
    branch_indices = []
    for key in state_dict.keys():
        if not key.startswith("transformer.layers."):
            continue

        parts = key.split(".")
        # Expected pattern examples:
        # transformer.layers.<layer_idx>.<block_idx>.fns.<branch_idx>.to_qkv.weight
        # transformer.layers.<layer_idx>.<block_idx>.fns.<branch_idx>.net.1.weight
        if len(parts) > 2 and parts[2].isdigit():
            layer_indices.append(int(parts[2]))

        if len(parts) > 5 and parts[4] == "fns" and parts[5].isdigit():
            branch_indices.append(int(parts[5]))

    if not layer_indices:
        raise ValueError("Unsupported ParallelViT checkpoint format.")

    depth = max(layer_indices) + 1
    num_parallel_branches = max(branch_indices) + 1 if branch_indices else 2

    qkv_weight = state_dict["transformer.layers.0.0.fns.0.to_qkv.weight"]
    inner_dim = qkv_weight.shape[0] // 3
    dim_head = 64
    heads = max(1, inner_dim // dim_head)

    mlp_dim = state_dict["transformer.layers.0.1.fns.0.net.1.weight"].shape[0]
    num_classes = state_dict["mlp_head.1.weight"].shape[0]

    model = ParallelViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        num_parallel_branches=num_parallel_branches,
        channels=channels,
        dim_head=dim_head,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.load_state_dict(state_dict)
    return model

def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
        if "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
            return loaded_obj["model_state_dict"]
        return loaded_obj
    return None

def _build_model_from_state_dict(state_dict):
    state_keys = set(state_dict.keys())

    if "transformer.layers.0.0.fns.0.to_qkv.weight" in state_keys:
        return _build_parallel_vit_from_state_dict(state_dict)

    if "pos_embedding" in state_keys and "to_patch_embedding.1.weight" in state_keys:
        return _build_vit_from_state_dict(state_dict)

    if "to_logits.2.weight" in state_keys and any(k.startswith("layers.") for k in state_keys):
        return _build_cvt_from_state_dict(state_dict)

    raise ValueError("Unsupported checkpoint format for configured models.")

def _materialize_lfs_pointer_if_possible(file_path):
    """Replace a Git LFS pointer file with its local object if already downloaded."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [f.readline().strip() for _ in range(3)]
    except Exception:
        return

    if not lines or not lines[0].startswith("version https://git-lfs.github.com/spec/v1"):
        return

    oid_line = next((line for line in lines if line.startswith("oid sha256:")), None)
    if not oid_line:
        return

    oid = oid_line.split("oid sha256:", 1)[1].strip()
    if len(oid) < 4:
        return

    lfs_object_path = os.path.join(".git", "lfs", "objects", oid[:2], oid[2:4], oid)
    if not os.path.exists(lfs_object_path):
        return

    if os.path.getsize(file_path) < os.path.getsize(lfs_object_path):
        shutil.copyfile(lfs_object_path, file_path)

@st.cache_resource(show_spinner=False)
def loadModel(model_name, model_signature=None):
    model_path = model_paths.get(model_name)

    if not model_path:
        st.error(f"Unknown model selection: '{model_name}'.")
        return None, None
        
    if model_path and _ensure_local_artifact(model_path):
        if model_path.endswith(".keras"):
            # Load TensorFlow model

            model = tf.keras.models.load_model(model_path)
            return model, 'tf'
        
        elif model_path.endswith(".pth"):
            # Load PyTorch model
            # Ensure CUDA-saved checkpoints can be loaded on CPU-only machines.
            try:
                import torch

                _materialize_lfs_pointer_if_possible(model_path)
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    loaded_obj = torch.load(model_path, map_location=target_device)
                except Exception:
                    # PyTorch>=2.6 defaults weights_only=True; retry for trusted local checkpoints.
                    loaded_obj = torch.load(model_path, map_location=target_device, weights_only=False)

                if isinstance(loaded_obj, torch.nn.Module):
                    model = loaded_obj
                else:
                    state_dict = _extract_state_dict(loaded_obj)
                    if isinstance(state_dict, dict):
                        model = _build_model_from_state_dict(state_dict)
                    else:
                        raise TypeError(f"Unsupported checkpoint type: {type(loaded_obj)}")

                model.to(target_device)
                model.eval()  # Set model to evaluation mode
                return model, 'torch'
            except Exception as exc:
                exc_msg = str(exc)
                if "invalid load key, 'v'" in exc_msg:
                    exc_msg = (
                        "Checkpoint file is not a valid PyTorch binary (likely a Git LFS pointer). "
                        "Please download the real .pth weights file."
                    )
                st.error(f"Failed to load PyTorch model '{model_name}': {exc_msg}")
                return None, None
    else:
        st.error(f"Model file not found: {model_path}")
            
    return None, None
    
# Run model on uploaded image
def run_model(model_name, image, output=None):
    target = output if output is not None else st

    loader_slot = target.empty()
    loader_slot.markdown(
        """
        <div class="inference-loader">
            <div class="inference-loader-text">Running AI inference pipeline...</div>
            <div class="inference-loader-track">
                <div class="inference-loader-bar"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = model_paths.get(model_name)
    model_signature = None
    if model_path and os.path.exists(model_path):
        model_signature = (os.path.getmtime(model_path), os.path.getsize(model_path))

    try:
        model, framework = loadModel(model_name, model_signature)
        if model:
            if framework == 'tf':
                # Preprocess and predict using TensorFlow model
                processed_image = preprocess_cnn_image(image)
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions) * 100  # Get confidence as a percentage

            elif framework == 'torch':
                # Preprocess and predict using PyTorch model
                import torch

                vit_input_size = get_vit_input_size(model)
                processed_image = preprocess_vit_image(image, vit_input_size)
                model_device = next(model.parameters()).device
                processed_image = processed_image.to(model_device)
                with torch.no_grad():
                    predictions = model(processed_image)
                predicted_class = predictions.argmax(dim=1).item()
                confidence = torch.softmax(predictions, dim=1)[0, predicted_class].item() * 100  # Confidence for PyTorch

            # Define class labels (adjust these to match your model's output)
            class_labels = ["Normal", "Benign", "Malignant", "Malignant_ACA", "Malignant_SCC"]
            status = class_labels[predicted_class]

            # Display the result with confidence
            print_deduction(status, confidence, output=output)
        else:
            target.error(f"Model '{model_name}' could not be loaded.")
    finally:
        loader_slot.empty()

st.markdown("<div class='section-banner'><span class='section-dot'></span>Clinical Workspace</div>", unsafe_allow_html=True)
image_col, survey_col = st.columns([1.2, 1.0], gap="large")

uploaded_file = None
selected_image_model = None
predict_image_clicked = False

with image_col:
    st.markdown("#### Image Model Controls")
    image_choice = st.selectbox(
        "**Choose Image Type**",
        options=["CT-Scan Image", "Histopathological Image"],
        key="img_type"
    )

    if image_choice == "CT-Scan Image":
        selected_image_model = st.selectbox(
            "**Choose Model**",
            options=sorted(model_names),
            key="img_model_ct"
        )
    else:
        selected_image_model = st.selectbox(
            "**Choose Model**",
            options=["ViT Histopathological Model"],
            key="img_model_hist"
        )

    uploaded_file = st.file_uploader(
        "**Choose an image...**",
        type=["jpg", "jpeg", "png"],
        key="img_upload"
    )

    selected_model_path = model_paths.get(selected_image_model, "")
    model_ready = os.path.exists(selected_model_path)
    model_framework = "TensorFlow" if selected_model_path.endswith(".keras") else "PyTorch"
    runtime_device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    st.markdown(
        f"""
        <div class="model-health-row">
            <div class="health-badge {'health-ready' if model_ready else 'health-warn'}">
                <span class="health-dot"></span>
                {'Local Model Ready' if model_ready else 'Model will fetch from Drive'}
            </div>
            <div class="health-badge {'health-gpu' if runtime_device == 'GPU' else 'health-cpu'}">
                <span class="health-dot"></span>
                Runtime: {runtime_device}
            </div>
            <div class="health-badge health-ready">
                <span class="health-dot"></span>
                Framework: {model_framework}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predict_image_clicked = st.button("Predict Image", key="predict_image_button")

    prediction_output = st.container()
    if predict_image_clicked:
        if uploaded_file is None:
            with prediction_output:
                st.markdown("#### Prediction Result")
                st.warning("Please upload an image first.")
        else:
            image_for_prediction = Image.open(uploaded_file)
            run_model(selected_image_model, image_for_prediction, output=prediction_output)
    else:
        with prediction_output:
            if uploaded_file is None:
                st.markdown("#### Prediction Result")
                st.caption("Upload an image and click Predict Image to view results here.")

    st.markdown("#### Preview and Diagnosis")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=420)
    else:
        st.info("Upload an image to preview here.")

"---"
# --------------------------------------------------------------

st.subheader("🔍 Exploring Lung Cancer")
st.write(
    """
    This section visualises data from the [Exploring Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer/data).
    The effectiveness of cancer prediction system can inform individuals of their cancer risk with low cost and it will help people to make a more informed decision based on their cancer risk status. 
    Just click on the widgets below to explore!
    """
)

cancer_directory = "data/survey_lung_cancer.csv"
@st.cache_data
def load_data():
    lung_df = pd.read_csv(cancer_directory)
    lung_df.columns = lung_df.columns.str.replace('_', ' ').str.strip().str.title()
    return lung_df

lung_df = load_data()

# Mapping dictionary for binary columns (1: No, 2: Yes)
binary_mapping = {1: "No", 2: "Yes", "YES": "Yes", "NO": "No", "M": "Male", "F": "Female"}

columns = ["Gender", "Age"]

# Apply mapping to each binary column
for col in lung_df.columns:
    if col != "Age":
        lung_df[col] = lung_df[col].map(binary_mapping)

# Gender selection with mapped values
genders = st.multiselect(
    "**Select Gender**",
    options=lung_df["Gender"].unique().tolist(),
    default=["Male", "Female"]
)

main_features = ["Smoking", "Peer Pressure", "Chronic Disease", "Alcohol Consuming"]
main_symptoms = ["Yellow Fingers", "Anxiety", "Fatigue", "Allergy", "Wheezing", "Coughing", "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain"]

# Features multiselect with relevant features
features = st.multiselect(
    "**Select Features**",
    options=main_features,
    default=main_features
)

# Symptoms multiselect based on symptom columns
symptoms = st.multiselect(
    "**Select Symptoms**",
    options=main_symptoms,
    default=main_symptoms
)

# Age slider based on the dataset's age range (1-120)
ages = st.slider(
    "**Select Age Range**", 
    min_value=1, 
    max_value=120, 
    value=(20, 50)
)

# Filter the dataframe based on widget inputs
lung_df_filtered = lung_df[
    (lung_df["Gender"].isin(genders)) &
    (lung_df["Age"].between(ages[0], ages[1])) 
]

lung_df_filtered = lung_df_filtered.sort_values(by="Age", ascending=True)

# Select only the necessary columns based on user input
columns_to_display = ["Age", "Gender", "Lung Cancer"] + features + symptoms
lung_df_filtered = lung_df_filtered[columns_to_display]

st.dataframe(
    lung_df_filtered,
    width='stretch',
    column_config={"Age": st.column_config.TextColumn("Age")},
)

# --------------------------------------------------------------
# LUNG CANCER STATISTICS

"---"

st.markdown(
    """
    <div class="viz-section">
        <div class="viz-title">Interactive Cancer Analytics</div>
        <div class="viz-sub">Hover over chart elements to spotlight key trends and compare distributions instantly.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pie chart for Lung Cancer status
lung_cancer_counts = lung_df_filtered['Lung Cancer'].value_counts().reset_index()
lung_cancer_counts.columns = ['Status', 'Count']
lung_cancer_counts['Percent'] = (lung_cancer_counts['Count'] / lung_cancer_counts['Count'].sum()) * 100

pie_highlight = alt.selection_point(fields=['Status'], on='mouseover', empty=True)

lung_cancer_chart = (
    alt.Chart(lung_cancer_counts)
    .mark_arc(innerRadius=68, cornerRadius=7, padAngle=0.02, stroke='#081320', strokeWidth=2)
    .encode(
        theta=alt.Theta(field='Count', type='quantitative'),
        color=alt.Color(
            field='Status',
            type='nominal',
            legend=alt.Legend(title='Lung Cancer Status'),
            scale=alt.Scale(range=['#2dd4bf', '#0ea5e9', '#f87171'])
        ),
        opacity=alt.condition(pie_highlight, alt.value(1), alt.value(0.65)),
        tooltip=[
            alt.Tooltip('Status:N', title='Status'),
            alt.Tooltip('Count:Q', title='Cases'),
            alt.Tooltip('Percent:Q', title='Share', format='.1f')
        ]
    )
    .add_params(pie_highlight)
    .properties(title='Lung Cancer Status Distribution', height=320)
    .configure(background='transparent')
    .configure_view(strokeOpacity=0)
    .configure_title(color='#e6f4ff', fontSize=18)
    .configure_axis(labelColor='#c3d8ea', titleColor='#e6f4ff', gridColor='#29405a')
    .configure_legend(labelColor='#c3d8ea', titleColor='#e6f4ff')
)

gender_counts = lung_df_filtered['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

bar_highlight = alt.selection_point(fields=['Gender'], on='mouseover', empty=True)

bar_base = alt.Chart(gender_counts).encode(
    x=alt.X('Gender:N', title='Gender', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('Count:Q', title='Count'),
    tooltip=[alt.Tooltip('Gender:N'), alt.Tooltip('Count:Q')]
)

bar_chart = bar_base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
    color=alt.condition(
        bar_highlight,
        alt.Color('Gender:N', scale=alt.Scale(range=['#2dd4bf', '#0ea5e9']), legend=None),
        alt.value('#3a5875')
    )
).add_params(bar_highlight)

bar_labels = bar_base.mark_text(
    dy=-8,
    color='#dff6ff',
    fontSize=12,
    fontWeight='bold'
).encode(text='Count:Q')

gender_chart = (
    (bar_chart + bar_labels)
    .properties(title='Gender Distribution', height=320)
    .configure(background='transparent')
    .configure_view(strokeOpacity=0)
    .configure_title(color='#e6f4ff', fontSize=18)
    .configure_axis(labelColor='#c3d8ea', titleColor='#e6f4ff', gridColor='#29405a')
    .configure_legend(labelColor='#c3d8ea', titleColor='#e6f4ff')
)

chart_col_1, chart_col_2 = st.columns(2, gap='large')

with chart_col_1:
    
    st.altair_chart(lung_cancer_chart, width='stretch')
    st.markdown("<div class='chart-caption'>Status ratio is shown as a donut chart with hover spotlight.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col_2:
    
    st.altair_chart(gender_chart, width='stretch')
    st.markdown("<div class='chart-caption'>Hover bars to emphasize each cohort and compare case volume.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

"---"
# --------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_models():
    # Load the models
    aux_paths = [
        'models/lr_model.pkl',
        'models/knn_model.pkl',
        'models/label_encoder.pkl',
        'models/scaler.pkl',
    ]
    for aux_path in aux_paths:
        if not _ensure_local_artifact(aux_path):
            st.error(f"Required model artifact not found: {aux_path}")
            st.stop()

    lr_model = joblib.load('models/lr_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return lr_model, knn_model, label_encoder, scaler


with survey_col:
    st.markdown("#### 📋 Survey Prediction")
    st.caption("Compact risk-form controls for quick patient screening.")

    age = st.slider("**Select Age**", min_value=1, max_value=120, value=30, key="survey_age")
    gender = st.selectbox("**Select Gender**", options=["Male", "Female"], key="survey_gender")

    feature_inputs = {}
    survey_features = [
        feature for feature in lung_df.columns
        if feature not in columns and feature != "Lung Cancer"
    ]

    feat_col_1, feat_col_2 = st.columns(2, gap="small")
    for idx, feature in enumerate(survey_features):
        target_col = feat_col_1 if idx % 2 == 0 else feat_col_2
        with target_col:
            feature_inputs[feature] = st.selectbox(
                f"**{feature}?**",
                options=["No", "Yes"],
                key=f"survey_feature_{idx}"
            )

    survey_model_choice = st.selectbox(
        "**Choose Model**",
        options=["Logistic Regression", "K-Nearest Neighbors"],
        key="survey_model_choice"
    )

    if st.button("Predict Survey", key="predict_survey_button"):
        lr_model, knn_model, label_encoder, scaler = load_models()
        selected_model = lr_model if survey_model_choice == "Logistic Regression" else knn_model

        input_data = {
            "Gender": 1 if gender == "Male" else 0,
            "Age": age,
            **feature_inputs,
            "Lung Cancer": "No"
        }

        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if col not in columns:
                input_df[col] = label_encoder.transform(input_df[col])

        del input_df["Lung Cancer"]
        input_df = scaler.transform(input_df)

        prediction = selected_model.predict(input_df)
        result = "Likely to have lung cancer." if prediction[0] == 1 else "Unlikely to have lung cancer."
        st.success(result)

# --------------------------------------------------------------


st.subheader("🌍 Lung Cancer Research")



st.subheader("🔗 References")

st.write(
    "- **[Lung Cancer DataSet](https://www.kaggle.com/datasets/yusufdede/lung-cancer-dataset), Yusuf Dede (2018)**"
)

st.write(
    "- **[Lung and Colon Cancer Histopathological Image Dataset (LC25000)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data), Borkowski AA (2019)**"
)

st.write(
    "- **[The IQ-OTH/NCCD lung cancer dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/data), Alyasriy (2023)**"
)
