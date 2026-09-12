"""
Streamlit Web Dashboard for TurboLife AI.
Refined Turbofan Engine RUL Command Center & Predictive Maintenance Platform.
Optimized with Streamlit @st.cache_resource and @st.cache_data for instant sub-second re-renders,
zero redundant model reloads, and persistent fleet-level inference caching.
"""

import os
import sys
from io import BytesIO

# Ensure config is imported first to set project-isolated environment variables
import config

import json
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import preprocess
import model as model_module
import predict as predict_module

# ==========================================
# 1. Streamlit Page Configuration & Premium Aviation Theme
# ==========================================
st.set_page_config(
    page_title="TurboLife AI – Turbofan Engine RUL Command Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom High-End Aviation CSS Theme
st.markdown(
    """
    <style>
    /* Clean Chrome: Hide deploy button, hamburger menu and footer only */
    .stDeployButton,
    .stAppDeployButton,
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }

    /* Force Dark Color Rendering Globally Across Browser & System Modes */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stMain"], .stApp {
        background: #060a14 !important;
        color: #e8efff !important;
        color-scheme: dark !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    /* Fix Streamlit Top Header/Toolbar: Never White */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: #060a14 !important;
        border-bottom: 1px solid #16233d !important;
    }

    /* Keep Sidebar visible, styled, and responsive */
    section[data-testid="stSidebar"] {
        background-color: #090e1a !important;
        border-right: 1px solid #182338 !important;
        color: #e8efff !important;
    }
    section[data-testid="stSidebar"] * {
        color-scheme: dark !important;
    }

    /* Ensure Collapse / Expand toggle buttons are styled in cyan and visible */
    [data-testid="stSidebarCollapseButton"] button {
        color: #38bdf8 !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        color: #38bdf8 !important;
    }

    /* Global Typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Responsive Main Content Container */
    @media (min-width: 769px) {
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2.4rem !important;
            padding-right: 2.4rem !important;
            max-width: 1480px;
        }
    }
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2.2rem !important;
            padding-left: 14px !important;
            padding-right: 14px !important;
            max-width: 100% !important;
        }
        section[data-testid="stSidebar"] {
            width: 88vw !important;
            min-width: 88vw !important;
            max-width: 95vw !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6) !important;
            z-index: 999999 !important;
        }
    }

    /* Sidebar Content Spacing & Grouped Panels */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #0d1527;
        border: 1px solid #1a2742;
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar-header-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.84rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }
    .sidebar-footer {
        text-align: center;
        font-size: 0.76rem;
        color: #64748b;
        padding: 10px 4px 4px 4px;
        line-height: 1.45;
    }

    /* Header Card - Curved Flight Path & Responsive Stacking */
    .header-bar {
        position: relative;
        background: radial-gradient(ellipse at 18% 30%, rgba(56, 189, 248, 0.09) 0%, transparent 60%), linear-gradient(135deg, #101a2e 0%, #0c1322 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 16px;
        padding: 48px 32px 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.32);
        box-sizing: border-box;
        overflow: hidden;
        min-height: 140px;
        width: 100%;
    }
    .header-flight-path {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }
    .header-content-left {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .header-brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .header-icon-badge {
        width: 48px;
        height: 48px;
        min-width: 48px;
        border-radius: 50%;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.22);
    }
    .header-text-container {
        display: flex;
        flex-direction: column;
    }
    .header-title-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .header-main-title {
        font-size: 1.75rem; /* ~28px */
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #cbd5e1;
        line-height: 1.2;
    }
    .header-desc-line {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .header-status-slot {
        position: relative;
        z-index: 2;
    }
    .status-pill-online {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid #10b981;
        color: #10b981;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        letter-spacing: 0.3px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .status-pill-offline {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid #ef4444;
        color: #ef4444;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* Desktop Subtitle Divider */
    @media (min-width: 769px) {
        .header-subtitle::before {
            content: "|";
            color: #475569;
            font-weight: 300;
            margin-right: 12px;
            margin-left: 2px;
        }
    }

    /* Mobile Header Stacking */
    @media (max-width: 768px) {
        .header-bar {
            flex-direction: column !important;
            align-items: flex-start !important;
            padding: 20px 16px 18px 16px !important;
            min-height: auto !important;
            gap: 14px !important;
        }
        .header-content-left {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
            width: 100% !important;
        }
        .header-brand-row {
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            width: 100% !important;
        }
        .header-main-title {
            font-size: 1.5rem !important;
        }
        .header-subtitle {
            font-size: 1rem !important;
        }
        .header-desc-line {
            font-size: 0.84rem !important;
        }
        .header-status-slot {
            width: 100% !important;
            display: flex !important;
            justify-content: flex-start !important;
        }
        .status-pill-online, .status-pill-offline {
            width: 100% !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }
    }

    /* Tab Page Headers */
    .tab-header-box {
        margin-bottom: 20px;
        padding-top: 6px;
    }
    .tab-main-heading {
        font-size: 1.45rem; /* ~23px */
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.3px;
        margin-bottom: 4px;
    }
    .tab-sub-heading {
        font-size: 0.95rem; /* ~15px */
        color: #94a3b8;
        line-height: 1.45;
    }
    @media (max-width: 768px) {
        .tab-main-heading {
            font-size: 1.25rem !important;
        }
        .tab-sub-heading {
            font-size: 0.86rem !important;
        }
    }

    /* Fleet KPI Summary Cards */
    .kpi-card {
        background: linear-gradient(145deg, #101a2e 0%, #0c1322 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 22px;
        text-align: left;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
        min-height: 96px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    .kpi-total::before { background: #38bdf8; }
    .kpi-healthy::before { background: #10b981; }
    .kpi-monitor::before { background: #f59e0b; }
    .kpi-high::before { background: #ef4444; }

    .kpi-label {
        font-size: 0.9rem; /* ~14.5px */
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.2px;
    }
    .kpi-val {
        font-size: 2.1rem; /* ~34px */
        font-weight: 800;
        color: #f8fafc;
        margin-top: 2px;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Mobile KPI Card 2x2 Grid Conversion */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"]:has(.kpi-card) {
            display: grid !important;
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 12px !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.kpi-card) > div[data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 auto !important;
        }
        .kpi-card {
            padding: 14px 14px !important;
            min-height: 84px !important;
        }
        .kpi-val {
            font-size: 1.6rem !important;
        }
        .kpi-label {
            font-size: 0.8rem !important;
        }
        .kpi-sub {
            font-size: 0.72rem !important;
        }
    }
    @media (max-width: 380px) {
        div[data-testid="stHorizontalBlock"]:has(.kpi-card) {
            grid-template-columns: 1fr !important;
        }
    }

    /* Selected Engine Diagnostic Header Banner */
    .engine-overview-box {
        background: #131d31;
        border: 1px solid #223252;
        border-radius: 14px;
        padding: 14px 22px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    .engine-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }

    /* Primary 4 Metric Cards for Selected Engine */
    .engine-metric-unit {
        text-align: center;
        background: #0d1527;
        border: 1px solid #1a2742;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 98px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }
    .engine-metric-label {
        font-size: 0.84rem;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .engine-metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 3px;
        line-height: 1.2;
    }
    .engine-metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Mobile Single-Column Stack for Metric Units */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"]:has(.engine-metric-unit) {
            display: grid !important;
            grid-template-columns: 1fr !important;
            gap: 10px !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.engine-metric-unit) > div[data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 auto !important;
        }
        .engine-metric-unit {
            padding: 14px 16px !important;
            min-height: auto !important;
        }
        .engine-metric-val {
            font-size: 1.45rem !important;
        }
    }

    /* Dedicated Evaluation Details Panel */
    .evaluation-panel {
        background: rgba(15, 24, 42, 0.85);
        border: 1px solid #223252;
        border-left: 4px solid #38bdf8;
        border-radius: 14px;
        padding: 14px 20px;
        margin-top: 16px;
        margin-bottom: 6px;
    }
    .eval-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .eval-metrics-row {
        font-size: 0.92rem;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .eval-divider {
        color: #475569;
    }
    .eval-disclaimer {
        color: #94a3b8;
        font-size: 0.8rem;
        font-style: italic;
    }
    @media (max-width: 768px) {
        .evaluation-panel {
            padding: 12px 14px !important;
        }
        .eval-metrics-row {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 6px !important;
            font-size: 0.85rem !important;
        }
        .eval-divider {
            display: none !important;
        }
    }

    /* Risk Badges */
    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.16);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-monitor {
        background-color: rgba(245, 158, 11, 0.16);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.16);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }

    /* Recommendation Cards */
    .rec-card-healthy {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.09) 0%, #131d31 100%);
        border-left: 5px solid #10b981;
        border-top: 1px solid #1e2d4a;
        border-right: 1px solid #1e2d4a;
        border-bottom: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 18px 24px;
        margin: 22px 0 24px 0;
    }
    .rec-card-monitor {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.09) 0%, #131d31 100%);
        border-left: 5px solid #f59e0b;
        border-top: 1px solid #1e2d4a;
        border-right: 1px solid #1e2d4a;
        border-bottom: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 18px 24px;
        margin: 22px 0 24px 0;
    }
    .rec-card-high {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.09) 0%, #131d31 100%);
        border-left: 5px solid #ef4444;
        border-top: 1px solid #1e2d4a;
        border-right: 1px solid #1e2d4a;
        border-bottom: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 18px 24px;
        margin: 22px 0 24px 0;
    }
    .rec-headline {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .rec-body {
        font-size: 0.94rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    @media (max-width: 768px) {
        .rec-card-healthy, .rec-card-monitor, .rec-card-high {
            padding: 14px 16px !important;
            margin: 16px 0 !important;
        }
        .rec-headline {
            font-size: 0.96rem !important;
        }
        .rec-body {
            font-size: 0.88rem !important;
        }
    }

    /* Consistent Analytics Cards */
    .analytics-card-container {
        background: linear-gradient(145deg, #0e172a 0%, #090e1a 100%);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.28);
        margin-bottom: 16px;
        box-sizing: border-box;
    }
    .panel-heading {
        font-size: 1.02rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
        min-height: 28px;
    }
    .gauge-interpretation-box {
        background: #080d18;
        border: 1px solid #1a2742;
        border-radius: 12px;
        padding: 14px 20px;
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.55;
        margin-top: 10px;
    }
    @media (max-width: 768px) {
        .analytics-card-container {
            padding: 16px 14px !important;
            border-radius: 14px !important;
        }
        .panel-heading {
            font-size: 0.95rem !important;
        }
        .gauge-interpretation-box {
            padding: 10px 14px !important;
            font-size: 0.84rem !important;
        }
    }

    /* Workflow Pipeline Process Cards */
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 16px 0 20px 0;
        position: relative;
    }
    .workflow-card {
        background: #0f182c;
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 14px;
        padding: 16px 14px;
        text-align: center;
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 112px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .workflow-card-num {
        font-size: 0.76rem;
        font-weight: 800;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .workflow-card-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .workflow-card-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.35;
    }
    .workflow-arrow-badge {
        position: absolute;
        right: -10px;
        top: 50%;
        transform: translateY(-50%);
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #090e1a;
        border: 1px solid #38bdf8;
        color: #38bdf8;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 5;
    }
    @media (max-width: 900px) {
        .workflow-grid {
            grid-template-columns: 1fr !important;
            gap: 10px !important;
        }
        .workflow-arrow-badge {
            display: none !important;
        }
    }

    /* Spec details card */
    .spec-block {
        background: #0d1527;
        border: 1px solid #1a2742;
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
        box-sizing: border-box;
    }
    .spec-block-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .spec-list {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.65;
        margin: 0;
        padding-left: 18px;
    }

    /* Red-accent urgent engines card */
    .urgent-engines-card {
        background: linear-gradient(145deg, #16121f 0%, #0d1527 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 5px solid #ef4444;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 22px;
        box-sizing: border-box;
    }
    @media (max-width: 768px) {
        .urgent-engines-card {
            padding: 14px 14px !important;
            border-radius: 14px !important;
        }
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Touch Targets - 44px Min Height for Mobile Accessibility */
    button[kind="secondary"],
    button[kind="primary"],
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        background-color: #10192d !important;
        color: #e8efff !important;
        border: 1px solid #1e2d4a !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        min-height: 44px !important;
        padding: 10px 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #16233d !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Input Fields & Select Boxes */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    input, select, textarea {
        background-color: #0d1527 !important;
        color: #e8efff !important;
        border-color: #1a2742 !important;
        color-scheme: dark !important;
        min-height: 44px !important;
        box-sizing: border-box !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #0d1527 !important;
        color: #e8efff !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #16233d !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #10192d !important;
        color: #e8efff !important;
        border-color: #1e2d4a !important;
        min-height: 44px !important;
        min-width: 44px !important;
    }

    /* Radio Controls, Checkboxes, and Toggles */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stToggle"] label {
        color: #e8efff !important;
        min-height: 44px !important;
        display: inline-flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] > div,
    div[data-testid="stToggle"] > div,
    div[data-testid="stCheckbox"] > div {
        color-scheme: dark !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #0d1527 !important;
        border: 1px solid #1a2742 !important;
        border-radius: 14px !important;
        color-scheme: dark !important;
    }
    div[data-testid="stExpander"] summary {
        color: #e8efff !important;
        background-color: #0d1527 !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #38bdf8 !important;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background-color: #0d1527 !important;
        color: #cbd5e1 !important;
    }

    /* Tabs Horizontal Scroll on Small Screens */
    div[data-testid="stTabs"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: thin !important;
        scrollbar-color: #1e2d4a transparent !important;
        padding-bottom: 6px !important;
        gap: 8px !important;
        width: 100% !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        flex-shrink: 0 !important;
        white-space: nowrap !important;
        padding: 10px 16px !important;
        font-size: 0.95rem !important;
        min-height: 44px !important;
        color: #94a3b8 !important;
        background-color: transparent !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    div[data-testid="stTabs"] {
        color-scheme: dark !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] section {
        background-color: #0d1527 !important;
        border-color: #1a2742 !important;
        color: #e8efff !important;
    }

    /* Responsive Column Collapse on Mobile */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]),
        div[data-testid="stHorizontalBlock"]:has(.spec-block),
        div[data-testid="stHorizontalBlock"]:has([data-testid="stRadio"]) {
            display: flex !important;
            flex-direction: column !important;
            gap: 14px !important;
        }
        div[data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]) > div[data-testid="column"],
        div[data-testid="stHorizontalBlock"]:has(.spec-block) > div[data-testid="column"],
        div[data-testid="stHorizontalBlock"]:has([data-testid="stRadio"]) > div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. Optimized Caching & Data Loading Layer
# ==========================================

# Cache 1: Model & Scaler Artifacts (Loaded once per server process using @st.cache_resource)
@st.cache_resource(show_spinner=False)
def load_system_artifacts():
    """
    Loads and caches the trained Deep LSTM neural network model, fitted MinMaxScaler,
    and metadata JSON. Loaded once per server session with @st.cache_resource
    to avoid re-importing weights or allocating GPU/CPU memory on every UI rerun.
    """
    model, scaler, metadata = None, None, None
    if os.path.exists(config.MODEL_FILE):
        try:
            model = model_module.load_trained_model(config.MODEL_FILE)
        except Exception as e:
            st.error(f"Error loading model: {e}")

    if os.path.exists(config.SCALER_FILE):
        try:
            scaler = preprocess.load_scaler(config.SCALER_FILE)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")

    if os.path.exists(config.METADATA_FILE):
        try:
            metadata = model_module.load_model_metadata(config.METADATA_FILE)
        except Exception as e:
            st.error(f"Error loading metadata: {e}")

    return model, scaler, metadata


# Cache 2: Static NASA C-MAPSS FD001 Telemetry Dataset (Disk read & parsing cached with @st.cache_data)
@st.cache_data(show_spinner=False)
def load_default_test_data() -> pd.DataFrame:
    """
    Loads and caches the NASA C-MAPSS FD001 benchmark test telemetry dataset.
    Uses @st.cache_data to prevent disk reads and space-separated text parsing
    on every user interaction.
    """
    preprocess.ensure_dataset_available()
    return preprocess.load_cmapss_file(config.TEST_DATA_FILE)


# Cache 3: Custom Uploaded Telemetry Parser (Raw bytes to parsed DataFrame cached with @st.cache_data)
@st.cache_data(show_spinner=False)
def parse_uploaded_telemetry(file_bytes: bytes) -> pd.DataFrame:
    """
    Parses and caches custom uploaded telemetry files from raw bytes.
    Avoids re-parsing text streams on widget changes.
    """
    df = pd.read_csv(BytesIO(file_bytes), sep=r"\s+", header=None).dropna(axis=1, how="all")
    cols = config.ALL_COLUMNS[: df.shape[1]]
    df.columns = cols
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


# Cache 4: Fleet Predictions (Inference over all 100 engines cached with @st.cache_data)
@st.cache_data(show_spinner=False)
def get_cached_fleet_predictions(
    test_df: pd.DataFrame,
    _model: Any,
    _scaler: Any,
    _metadata: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Computes and caches Remaining Useful Life (RUL) inference for all fleet units.
    Uses @st.cache_data with ignored model/scaler hashing (_model, _scaler) to avoid
    re-running 100-engine neural network inference on every UI widget interaction.
    """
    return predict_module.predict_fleet_rul(
        test_df=test_df,
        model=_model,
        scaler=_scaler,
        metadata=_metadata,
    )


# Cache 5: Friendly Sensor Label Formatter (Cached with @st.cache_data)
@st.cache_data(show_spinner=False)
def format_sensor_label(s_col: str) -> str:
    """Generates friendly aeronautical engineering label with raw column in brackets (cached)."""
    meta = config.SENSOR_METADATA.get(s_col, {})
    if meta and "name" in meta:
        return f"{meta['name']} [{s_col}]"
    return s_col


# Cache 6: Precomputed Sensor Channel Mapping Dictionary (Cached with @st.cache_data)
@st.cache_data(show_spinner=False)
def get_cached_sensor_label_map(sensor_cols: Tuple[str, ...]) -> Dict[str, str]:
    """
    Precomputes and caches dictionary mapping friendly sensor labels to raw column names.
    Cached with @st.cache_data to avoid dictionary reconstruction on every render.
    """
    label_map = {}
    for s_col in sensor_cols:
        label_map[format_sensor_label(s_col)] = s_col
    return label_map


def build_compact_rul_gauge(predicted_rul: float) -> go.Figure:
    """
    Builds a clean, centered radial gauge for Remaining Useful Life (0 to 125 cycles).
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_rul,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Predicted Remaining Useful Life", "font": {"size": 15, "color": "#cbd5e1"}},
            number={"suffix": " cyc", "font": {"size": 32, "color": "#38bdf8", "family": "Arial"}},
            gauge={
                "axis": {
                    "range": [0, config.RUL_CAP],
                    "tickwidth": 1.2,
                    "tickcolor": "#64748b",
                    "tickmode": "array",
                    "tickvals": [0, 30, 80, 125],
                    "ticktext": ["0", "30 (Red)", "80 (Yel)", "125"],
                    "tickfont": {"size": 10.5, "color": "#94a3b8"},
                },
                "bar": {"color": "#38bdf8", "thickness": 0.28},
                "bgcolor": "#080d18",
                "borderwidth": 1,
                "bordercolor": "#1e2d4a",
                "steps": [
                    {"range": [0, config.MONITOR_THRESHOLD], "color": "rgba(239, 68, 68, 0.35)"},        # 0-30 High Risk
                    {"range": [config.MONITOR_THRESHOLD, config.HEALTHY_THRESHOLD], "color": "rgba(245, 158, 11, 0.35)"}, # 31-80 Monitor
                    {"range": [config.HEALTHY_THRESHOLD, config.RUL_CAP], "color": "rgba(16, 185, 129, 0.35)"},   # 81-125 Healthy
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": config.MONITOR_THRESHOLD,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        autosize=True,
        height=240,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def plot_sensor_telemetry_clean(
    engine_df: pd.DataFrame,
    sensor_cols: List[str],
    current_cycle: int,
    normalized_mode: bool = True,
) -> go.Figure:
    """
    Renders clean Plotly time-series degradation trends across the full container width.
    """
    fig = go.Figure()
    
    color_palette = ["#38bdf8", "#f59e0b", "#10b981", "#a855f7", "#ec4899"]
    
    for idx, s_col in enumerate(sensor_cols[:3]):  # Enforce max 3 sensors
        meta = config.SENSOR_METADATA.get(s_col, {"name": s_col, "unit": ""})
        unit_str = f" [{meta['unit']}]" if meta["unit"] and meta["unit"] != "-" else ""
        friendly_label = f"{meta['name']} [{s_col}]"
        
        raw_values = engine_df[s_col].values
        
        if normalized_mode:
            min_val = np.min(raw_values)
            max_val = np.max(raw_values)
            if max_val > min_val:
                y_plot = (raw_values - min_val) / (max_val - min_val)
            else:
                y_plot = np.zeros_like(raw_values)
            y_hover = [f"{v:.3f} (Raw: {r:.2f}{unit_str})" for v, r in zip(y_plot, raw_values)]
        else:
            y_plot = raw_values
            y_hover = [f"{r:.2f}{unit_str}" for r in raw_values]

        fig.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=y_plot,
                mode="lines+markers",
                name=friendly_label,
                line=dict(width=2.4, color=color_palette[idx % len(color_palette)]),
                marker=dict(size=3.5),
                customdata=y_hover,
                hovertemplate="Cycle %{x}<br>" + friendly_label + ": %{customdata}<extra></extra>",
            )
        )

    # Vertical line indicating the final / current flight cycle
    fig.add_vline(
        x=current_cycle,
        line_width=1.5,
        line_dash="dash",
        line_color="#38bdf8",
        annotation_text=f"Current ({current_cycle})",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#38bdf8"),
    )

    y_axis_title = "Normalized Trend (0.0 to 1.0 Range)" if normalized_mode else "Raw Sensor Engineering Units"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080d18",
        autosize=True,
        xaxis=dict(
            title=dict(text="Flight Cycle (Operating Missions)", font=dict(size=12, color="#94a3b8")),
            gridcolor="#152136",
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=10.5, color="#cbd5e1"),
        ),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(size=12, color="#94a3b8")),
            gridcolor="#152136",
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=10.5, color="#cbd5e1"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#cbd5e1"),
        ),
        height=400,
        margin=dict(l=40, r=15, t=20, b=75),
    )
    return fig


# ==========================================
# 3. Main Command Center Application
# ==========================================
def main():
    # Load Model Artifacts with a small loading spinner only on first load
    if "model_loaded_once" not in st.session_state:
        with st.spinner("Initializing TurboLife AI Deep LSTM Model & Artifacts..."):
            model, scaler, metadata = load_system_artifacts()
        st.session_state["model_loaded_once"] = True
    else:
        model, scaler, metadata = load_system_artifacts()

    model_ready = model is not None and scaler is not None and metadata is not None

    # Initialize Session State Defaults
    if "selected_engine_id" not in st.session_state:
        st.session_state["selected_engine_id"] = 1

    # ----------------------------------------------------
    # SECTION 1: Top Header Bar (Clean Intentional Flight Path)
    # ----------------------------------------------------
    online_pill_html = (
        '<div class="status-pill-online">● Model Loaded • Inference Ready</div>'
        if model_ready
        else '<div class="status-pill-offline">● Model Not Loaded • Train First</div>'
    )

    st.markdown(
        f"""
        <div class="header-bar">
            <svg class="header-flight-path" viewBox="0 0 1000 140" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="flightPathGrad" x1="100%" y1="0%" x2="0%" y2="0%">
                        <stop offset="0%" stop-color="#0284c7" stop-opacity="0.05"/>
                        <stop offset="35%" stop-color="#0ea5e9" stop-opacity="0.45"/>
                        <stop offset="75%" stop-color="#38bdf8" stop-opacity="0.85"/>
                        <stop offset="100%" stop-color="#38bdf8" stop-opacity="1.0"/>
                    </linearGradient>
                    <filter id="pathGlow" x="-10%" y="-10%" width="120%" height="120%">
                        <feGaussianBlur stdDeviation="2" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                <path d="M 965 28 C 720 38, 460 20, 280 21 C 230 22, 195 23, 170 24" stroke="url(#flightPathGrad)" stroke-width="2" fill="none" filter="url(#pathGlow)"/>
                <circle cx="170" cy="24" r="7" fill="#38bdf8" opacity="0.22" filter="url(#pathGlow)"/>
                <circle cx="170" cy="24" r="4.2" fill="#38bdf8"/>
                <circle cx="170" cy="24" r="2" fill="#ffffff"/>
            </svg>
            <div class="header-content-left">
                <div class="header-brand-row">
                    <div class="header-icon-badge">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
                        </svg>
                    </div>
                    <span class="header-main-title">TurboLife AI</span>
                </div>
                <div class="header-text-container">
                    <div class="header-title-row">
                        <span class="header-subtitle">Predictive Maintenance Command Center</span>
                    </div>
                    <div class="header-desc-line">LSTM-based remaining useful life prediction for turbofan engines</div>
                </div>
            </div>
            <div class="header-status-slot">
                {online_pill_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Collapsible Detailed System Information
    with st.expander("About this system (Architecture & C-MAPSS Physics)", expanded=False):
        st.markdown(
            """
            **TurboLife AI** applies deep recurrent sequence learning to predict the Remaining Useful Life (RUL in flight cycles) 
            of aircraft turbofan engines before reaching critical wear limits.
            
            - **Temporal Dynamics**: Rather than evaluating isolated sensor snapshots, our **2-Layer LSTM Network** processes sliding windows of the last **30 flight cycles** across 14 thermodynamic sensor channels (compressor temperatures, turbine pressures, rotor speeds, fuel flow ratios).
            - **Piecewise Linear Target**: Early in engine life (>125 cycles), engines operate nominally without observable degradation. The target RUL is capped at **125 cycles** to focus gradient learning on the exponential wear regime.
            - **Risk Tiers**:
              - <span style="color: #10b981; font-weight: bold;">●</span> **Healthy** (>80 cycles): Continue normal scheduled operations.
              - <span style="color: #f59e0b; font-weight: bold;">●</span> **Monitor** (31–80 cycles): Plan inspection during the next maintenance window.
              - <span style="color: #ef4444; font-weight: bold;">●</span> **High Risk** (≤30 cycles): Schedule urgent maintenance action.
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # SIDEBAR: Structured, Consistent Command Panels
    # ----------------------------------------------------
    with st.sidebar:
        # Group 1: Command Controls
        with st.container(border=True):
            st.markdown(
                """
                <div class="sidebar-header-badge">
                    <span>Command Controls</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Reset Dashboard", width="stretch"):
                st.session_state["selected_engine_id"] = 1
                st.rerun()

        # Group 2: Evaluation Settings
        with st.container(border=True):
            st.markdown(
                """
                <div class="sidebar-header-badge">
                    <span>Evaluation Settings</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            evaluation_mode = st.toggle(
                "Evaluation Mode",
                value=True,
                help="Show NASA C-MAPSS benchmark ground-truth RUL and prediction error metrics.",
            )
            if evaluation_mode:
                st.markdown(
                    "<div style='font-size: 0.78rem; color: #94a3b8; line-height: 1.4; margin-top: 4px;'>Actual RUL is available only because this is a labelled NASA benchmark dataset. In real maintenance, actual RUL is unknown before failure.</div>",
                    unsafe_allow_html=True,
                )

        # Group 3: Telemetry Source
        with st.container(border=True):
            st.markdown(
                """
                <div class="sidebar-header-badge">
                    <span>Telemetry Source</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            data_source = st.radio(
                "Select Data Mode:",
                ["NASA C-MAPSS FD001 Test Fleet", "Upload Custom Test File (.txt / .csv)", "Demo Engine Generator"],
                index=0,
                label_visibility="collapsed",
            )

            test_df = None
            if data_source == "NASA C-MAPSS FD001 Test Fleet":
                try:
                    test_df = load_default_test_data()
                except Exception as e:
                    st.error(f"Error loading C-MAPSS test data: {e}")
            elif data_source == "Upload Custom Test File (.txt / .csv)":
                uploaded_file = st.file_uploader("Upload Telemetry File", type=["txt", "csv"])
                if uploaded_file is not None:
                    try:
                        test_df = parse_uploaded_telemetry(uploaded_file.getvalue())
                        st.success(f"Loaded {test_df['engine_id'].nunique()} custom engines.")
                    except Exception as e:
                        st.error(f"Failed to parse uploaded file: {e}")
            else:
                preprocess.ensure_dataset_available()
                test_df = load_default_test_data()

        # Group 4: Aircraft Engine Selection
        with st.container(border=True):
            st.markdown(
                """
                <div class="sidebar-header-badge">
                    <span>Aircraft Engine Selection</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            selected_engine_id = 1
            if test_df is not None and not test_df.empty:
                available_engines = sorted(test_df["engine_id"].unique().tolist())
                
                # Ensure state is valid
                if st.session_state["selected_engine_id"] not in available_engines:
                    st.session_state["selected_engine_id"] = available_engines[0]

                # Quick Number Input / Jump
                quick_id = st.number_input(
                    "Quick ID Jump:",
                    min_value=min(available_engines),
                    max_value=max(available_engines),
                    value=st.session_state["selected_engine_id"],
                    step=1,
                )
                if quick_id in available_engines and quick_id != st.session_state["selected_engine_id"]:
                    st.session_state["selected_engine_id"] = quick_id
                    st.rerun()

                st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
                selected_engine_id = st.selectbox(
                    "Select Engine Unit:",
                    options=available_engines,
                    index=available_engines.index(st.session_state["selected_engine_id"]),
                    label_visibility="collapsed",
                )
                st.session_state["selected_engine_id"] = selected_engine_id

        # Sidebar Footer
        st.markdown(
            """
            <div class="sidebar-footer">
                <b>TurboLife AI Control System v1.0</b><br/>
                Created by Charan Sai • NASA C-MAPSS Benchmark
            </div>
            """,
            unsafe_allow_html=True,
        )

    # If Model is missing and not trained yet
    if not model_ready:
        st.error("Trained model artifacts are missing (`models/rul_lstm.keras` or `scaler.joblib`).")
        st.info("Run `python train.py` in your terminal to train the Deep LSTM model on the NASA CMAPSS dataset.")
        return

    if test_df is None or test_df.empty:
        st.info("Please select or upload a valid telemetry dataset in the sidebar.")
        return

    # Execute Cached Fleet Predictions (computed once per dataset, not on every widget interaction)
    try:
        fleet_results = get_cached_fleet_predictions(
            test_df=test_df,
            _model=model,
            _scaler=scaler,
            _metadata=metadata,
        )
    except Exception as e:
        st.error(f"Inference computation error: {e}")
        return

    if fleet_results is None or fleet_results.empty:
        st.error("Unable to generate fleet predictions for the current telemetry data.")
        return

    # ----------------------------------------------------
    # SECTION A: Fleet Summary Strip (4 Balanced KPI Cards)
    # ----------------------------------------------------
    total_engines = len(fleet_results)
    healthy_count = (fleet_results["risk_level"] == config.RISK_HEALTHY).sum()
    monitor_count = (fleet_results["risk_level"] == config.RISK_MONITOR).sum()
    high_risk_count = (fleet_results["risk_level"] == config.RISK_HIGH).sum()

    f_col1, f_col2, f_col3, f_col4 = st.columns(4, gap="medium")
    
    with f_col1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-total">
                <div class="kpi-label">Total Fleet Units</div>
                <div class="kpi-val">{total_engines}</div>
                <div class="kpi-sub">Active C-MAPSS Engines</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        h_pct = round((healthy_count / total_engines) * 100, 1)
        st.markdown(
            f"""
            <div class="kpi-card kpi-healthy">
                <div class="kpi-label"><span style="color: #10b981;">●</span> Healthy</div>
                <div class="kpi-val" style="color: #10b981;">{healthy_count}</div>
                <div class="kpi-sub">{h_pct}% of fleet (&gt;80 cyc)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        m_pct = round((monitor_count / total_engines) * 100, 1)
        st.markdown(
            f"""
            <div class="kpi-card kpi-monitor">
                <div class="kpi-label"><span style="color: #f59e0b;">●</span> Monitor</div>
                <div class="kpi-val" style="color: #f59e0b;">{monitor_count}</div>
                <div class="kpi-sub">{m_pct}% of fleet (31-80 cyc)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col4:
        r_pct = round((high_risk_count / total_engines) * 100, 1)
        st.markdown(
            f"""
            <div class="kpi-card kpi-high">
                <div class="kpi-label"><span style="color: #ef4444;">●</span> High Risk</div>
                <div class="kpi-val" style="color: #ef4444;">{high_risk_count}</div>
                <div class="kpi-sub">{r_pct}% of fleet (&le;30 cyc)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # SECTION D: 3 Core Analytics Tabs
    # ----------------------------------------------------
    tab_diag, tab_fleet, tab_model = st.tabs(
        [
            "Engine Health & Sensor Analysis",
            "Fleet Health Overview",
            "Model Architecture & Data",
        ]
    )

    # ----------------------------------------------------
    # TAB 1: Engine Health & Sensor Analysis
    # ----------------------------------------------------
    with tab_diag:
        st.markdown(
            """
            <div class="tab-header-box">
                <div class="tab-main-heading">Selected Engine Health & Telemetry Diagnostics</div>
                <div class="tab-sub-heading">Unit-level wear trajectory, remaining useful life predictions, and multi-channel thermodynamic telemetry.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        engine_data = fleet_results[fleet_results["engine_id"] == selected_engine_id].iloc[0]
        engine_telemetry = test_df[test_df["engine_id"] == selected_engine_id]

        pred_rul = float(engine_data["predicted_rul"])
        current_cycle = int(engine_data["current_cycle"])
        risk_level = str(engine_data["risk_level"])
        recommendation = str(engine_data["recommendation"])
        actual_rul = engine_data.get("actual_rul", None)
        rul_error = engine_data.get("error", None)

        badge_class = (
            "badge-healthy"
            if risk_level == config.RISK_HEALTHY
            else "badge-monitor"
            if risk_level == config.RISK_MONITOR
            else "badge-high"
        )

        st.markdown(
            f"""
            <div class="engine-overview-box">
                <div class="engine-title">
                    <span>Selected Engine Diagnostic: <b>Unit #{selected_engine_id}</b></span>
                    <span class="{badge_class}">{risk_level.upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Primary Metric Cards in Row
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        
        with m1:
            st.markdown(
                f"""
                <div class="engine-metric-unit">
                    <div class="engine-metric-label">Engine Identifier</div>
                    <div class="engine-metric-val">Unit #{selected_engine_id}</div>
                    <div class="engine-metric-sub">NASA C-MAPSS FD001</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        with m2:
            st.markdown(
                f"""
                <div class="engine-metric-unit">
                    <div class="engine-metric-label">Current Flight Cycle</div>
                    <div class="engine-metric-val">{current_cycle}</div>
                    <div class="engine-metric-sub">Missions Completed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="engine-metric-unit">
                    <div class="engine-metric-label">Predicted RUL</div>
                    <div class="engine-metric-val" style="color: #38bdf8;">{pred_rul:.1f} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">cycles</span></div>
                    <div class="engine-metric-sub">Maintenance Planning Indicator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            st.markdown(
                f"""
                <div class="engine-metric-unit">
                    <div class="engine-metric-label">Operational Risk Status</div>
                    <div style="margin-top: 4px;"><span class="{badge_class}">{risk_level}</span></div>
                    <div class="engine-metric-sub">Maintenance Action Tier</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dedicated Evaluation Details Panel
        if evaluation_mode and actual_rul is not None and not pd.isna(actual_rul):
            err_sign = "+" if rul_error >= 0 else ""
            abs_err = abs(rul_error)
            st.markdown(
                f"""
                <div class="evaluation-panel">
                    <div class="eval-title">NASA Benchmark Evaluation Data</div>
                    <div class="eval-metrics-row">
                        <span><b>Actual RUL (Ground Truth):</b> <span style="color: #38bdf8;">{actual_rul:.1f} cycles</span></span>
                        <span class="eval-divider">•</span>
                        <span><b>Prediction Error (&Delta;):</b> <span style="color: #f8fafc;">{err_sign}{rul_error:.1f} cycles</span></span>
                        <span class="eval-divider">•</span>
                        <span><b>Absolute Error (|&Delta;|):</b> <span style="color: #f8fafc;">{abs_err:.1f} cycles</span></span>
                        <span class="eval-divider">•</span>
                        <span class="eval-disclaimer">Actual RUL is available only for benchmark evaluation.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recommendation Card
        rec_class = (
            "rec-card-healthy"
            if risk_level == config.RISK_HEALTHY
            else "rec-card-monitor"
            if risk_level == config.RISK_MONITOR
            else "rec-card-high"
        )
        rec_icon = (
            "<span style=\"color: #10b981;\">●</span>"
            if risk_level == config.RISK_HEALTHY
            else ("<span style=\"color: #f59e0b;\">●</span>" if risk_level == config.RISK_MONITOR else "<span style=\"color: #ef4444;\">●</span>")
        )

        if risk_level == config.RISK_HEALTHY:
            rec_title = "Continue Normal Scheduled Flight Operations"
        elif risk_level == config.RISK_MONITOR:
            rec_title = "Plan Maintenance Inspection in Next Window"
        else:
            rec_title = "Schedule Urgent Maintenance Action Immediately"

        st.markdown(
            f"""
            <div class="{rec_class}">
                <div class="rec-headline">
                    {rec_icon} Maintenance Action Protocol: <b>{rec_title}</b>
                </div>
                <div class="rec-body">
                    {recommendation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ROW 1: Dedicated Centered RUL Radial Gauge Card
        st.markdown(
            """
            <div class="analytics-card-container">
                <div class="panel-heading" style="justify-content: center;">
                    Remaining Useful Life (RUL) Radial Gauge Meter
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        g_col1, g_col2, g_col3 = st.columns([1.2, 2.6, 1.2])
        with g_col2:
            st.plotly_chart(
                build_compact_rul_gauge(pred_rul),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
                key="diag_rul_gauge",
            )
            st.markdown(
                f"""
                <div class="gauge-interpretation-box" style="text-align: center;">
                    <b>Unit #{selected_engine_id}</b> is projected to operate safely for <b>{pred_rul:.1f} more flight cycles</b> before reaching the <b>{risk_level}</b> maintenance threshold.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # ROW 2: Dedicated Full-Width Historical Telemetry Card
        active_sensors = tuple(metadata.get("feature_columns", [c for c in test_df.columns if c.startswith("sensor_")]))
        sensor_label_map = get_cached_sensor_label_map(active_sensors)
        friendly_options = list(sensor_label_map.keys())

        preferred_defaults_raw = [s for s in ["sensor_3", "sensor_4", "sensor_7"] if s in active_sensors]
        if len(preferred_defaults_raw) < 3:
            preferred_defaults_raw = list(active_sensors[:3])
        preferred_defaults_friendly = [format_sensor_label(s) for s in preferred_defaults_raw]

        st.markdown(
            """
            <div class="analytics-card-container">
                <div class="panel-heading">
                    Historical Telemetry Degradation Trends
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Compact horizontal control strip directly above the graph
        c_view, c_sel = st.columns([1.1, 1.9], gap="medium")
        
        with c_view:
            telemetry_mode = st.radio(
                "Telemetry View Mode:",
                ["Normalized Degradation Trends", "Raw Sensor Values"],
                index=0,
                horizontal=True,
                help="Normalized mode scales curves (0.0 to 1.0) so multi-range sensors can be visually compared.",
            )
        
        with c_sel:
            selected_friendly = st.multiselect(
                "Telemetry Channels (Max 3):",
                options=friendly_options,
                default=preferred_defaults_friendly[:3],
                max_selections=3,
                label_visibility="visible",
            )

        selected_raw_cols = [sensor_label_map[f] for f in selected_friendly if f in sensor_label_map]

        if selected_raw_cols:
            is_normalized = (telemetry_mode == "Normalized Degradation Trends")
            st.plotly_chart(
                plot_sensor_telemetry_clean(
                    engine_df=engine_telemetry,
                    sensor_cols=selected_raw_cols,
                    current_cycle=current_cycle,
                    normalized_mode=is_normalized,
                ),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
                key="diag_sensor_telemetry",
            )
        else:
            st.info("Select 1 to 3 telemetry channels above to visualize degradation curves.")

    # ----------------------------------------------------
    # TAB 2: Fleet Health Overview Tab
    # ----------------------------------------------------
    with tab_fleet:
        st.markdown(
            """
            <div class="tab-header-box">
                <div class="tab-main-heading">Fleet Health Distribution & Prioritization</div>
                <div class="tab-sub-heading">Global fleet risk stratification, prioritized overhaul schedules, and inspection register across all 100 test units.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        fc1, fc2 = st.columns([35, 65], gap="large")
        
        with fc1:
            # Donut Chart
            donut_df = pd.DataFrame({
                "Risk Tier": [config.RISK_HIGH, config.RISK_MONITOR, config.RISK_HEALTHY],
                "Engine Count": [high_risk_count, monitor_count, healthy_count],
            })
            fig_donut = px.pie(
                donut_df,
                values="Engine Count",
                names="Risk Tier",
                color="Risk Tier",
                color_discrete_map={
                    config.RISK_HIGH: config.RISK_COLORS[config.RISK_HIGH],
                    config.RISK_MONITOR: config.RISK_COLORS[config.RISK_MONITOR],
                    config.RISK_HEALTHY: config.RISK_COLORS[config.RISK_HEALTHY],
                },
                hole=0.5,
                title="Fleet Risk Proportion",
            )
            fig_donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                autosize=True,
                height=300,
                margin=dict(l=15, r=15, t=35, b=15),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10.5)),
            )
            st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False, "responsive": True}, key="donut_fleet_risk")

        with fc2:
            # Sorted Bar Chart
            sorted_fleet = fleet_results.sort_values(by="predicted_rul").reset_index(drop=True)
            fig_bar = px.bar(
                sorted_fleet,
                x="engine_id",
                y="predicted_rul",
                color="risk_level",
                color_discrete_map={
                    config.RISK_HIGH: config.RISK_COLORS[config.RISK_HIGH],
                    config.RISK_MONITOR: config.RISK_COLORS[config.RISK_MONITOR],
                    config.RISK_HEALTHY: config.RISK_COLORS[config.RISK_HEALTHY],
                },
                title="Fleet RUL Ranking (All 100 Engines)",
                labels={"engine_id": "Engine ID", "predicted_rul": "Predicted RUL (Cycles)", "risk_level": "Risk Tier"},
            )
            fig_bar.add_hline(y=config.MONITOR_THRESHOLD, line_dash="dash", line_color="#ef4444", annotation_text="High Risk (30)", annotation_font_size=10)
            fig_bar.add_hline(y=config.HEALTHY_THRESHOLD, line_dash="dash", line_color="#10b981", annotation_text="Healthy (80)", annotation_font_size=10)
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#080d18",
                autosize=True,
                height=310,
                margin=dict(l=35, r=15, t=35, b=35),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10.5)),
            )
            st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False, "responsive": True}, key="bar_fleet_rul")

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # Most Urgent Engines (Top 5 Lowest RUL) in Red-Accent Card
        urgent_5 = fleet_results.sort_values(by="predicted_rul").head(5).copy()
        
        if evaluation_mode and "actual_rul" in urgent_5.columns:
            disp_cols = ["engine_id", "current_cycle", "predicted_rul", "risk_level", "actual_rul", "error"]
        else:
            disp_cols = ["engine_id", "current_cycle", "predicted_rul", "risk_level"]
            
        def highlight_risk(val):
            color = config.RISK_COLORS.get(val, None)
            if color:
                return f"background-color: {color}; color: #000; font-weight: bold;"
            return ""

        st.markdown(
            """
            <div class="urgent-engines-card">
                <div class="panel-heading" style="color: #ef4444; margin-bottom: 12px;">
                    Most Urgent Engines (Top 5 Critical Units Requiring Immediate Overhaul)
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            urgent_5[disp_cols].style.map(highlight_risk, subset=["risk_level"]),
            width="stretch",
            hide_index=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Complete Fleet Inspection Register")
        
        filter_risk = st.multiselect(
            "Filter Fleet Table by Risk Tier:",
            options=[config.RISK_HIGH, config.RISK_MONITOR, config.RISK_HEALTHY],
            default=[config.RISK_HIGH, config.RISK_MONITOR, config.RISK_HEALTHY],
        )

        filtered_fleet = fleet_results[fleet_results["risk_level"].isin(filter_risk)].sort_values(by="predicted_rul")
        
        if evaluation_mode and "actual_rul" in filtered_fleet.columns:
            table_cols = ["engine_id", "current_cycle", "predicted_rul", "risk_level", "actual_rul", "error", "recommendation"]
        else:
            table_cols = ["engine_id", "current_cycle", "predicted_rul", "risk_level", "recommendation"]

        st.dataframe(
            filtered_fleet[table_cols].style.map(highlight_risk, subset=["risk_level"]),
            width="stretch",
            height=320,
            hide_index=True,
        )

        # CSV Export Button
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        csv_bytes = filtered_fleet[table_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Fleet Prediction Report (CSV)",
            data=csv_bytes,
            file_name="turbolife_fleet_predictions.csv",
            mime="text/csv",
        )

    # ----------------------------------------------------
    # TAB 3: Model Architecture & Data Insights
    # ----------------------------------------------------
    with tab_model:
        st.markdown(
            """
            <div class="tab-header-box">
                <div class="tab-main-heading">Deep LSTM Architecture & C-MAPSS Physics</div>
                <div class="tab-sub-heading">Recurrent neural network specifications, temporal feature sliding windows, and piecewise linear wear modeling.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Connected Workflow Process Cards
        st.markdown(
            """
            <div class="analytics-card-container" style="margin-bottom: 24px;">
                <div class="panel-heading">
                    System Workflow Pipeline
                </div>
                <div class="workflow-grid">
                    <div class="workflow-card">
                        <div class="workflow-card-num">Step 1</div>
                        <div class="workflow-card-title">Sensor Telemetry</div>
                        <div class="workflow-card-sub">21 Thermodynamic Channels</div>
                        <div class="workflow-arrow-badge">→</div>
                    </div>
                    <div class="workflow-card">
                        <div class="workflow-card-num">Step 2</div>
                        <div class="workflow-card-title">30-Cycle Sequence</div>
                        <div class="workflow-card-sub">Temporal Sliding Window</div>
                        <div class="workflow-arrow-badge">→</div>
                    </div>
                    <div class="workflow-card">
                        <div class="workflow-card-num">Step 3</div>
                        <div class="workflow-card-title">Deep LSTM Network</div>
                        <div class="workflow-card-sub">64 → 32 Units + Dropout</div>
                        <div class="workflow-arrow-badge">→</div>
                    </div>
                    <div class="workflow-card">
                        <div class="workflow-card-num">Step 4</div>
                        <div class="workflow-card-title">RUL Regression</div>
                        <div class="workflow-card-sub">Capped at 125 Cycles</div>
                        <div class="workflow-arrow-badge">→</div>
                    </div>
                    <div class="workflow-card">
                        <div class="workflow-card-num">Step 5</div>
                        <div class="workflow-card-title">Maintenance Action</div>
                        <div class="workflow-card-sub">Tri-Tier Protocol Engine</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Equal Aligned Two Columns
        m_left, m_right = st.columns(2, gap="large")
        
        with m_left:
            st.markdown(
                f"""
                <div class="spec-block" style="margin-bottom: 20px;">
                    <div class="spec-block-title">LSTM Network Specifications</div>
                    <ul class="spec-list">
                        <li><b>Architecture:</b> 2-Layer Deep LSTM Regressor</li>
                        <li><b>Temporal Window:</b> {metadata.get('sequence_length', config.SEQUENCE_LENGTH)} consecutive flight cycles</li>
                        <li><b>Active Channels:</b> {len(metadata.get('feature_columns', []))} informative sensor streams</li>
                        <li><b>Layer 1:</b> LSTM ({config.LSTM_UNITS_1} Units, Dropout = {config.DROPOUT_RATE})</li>
                        <li><b>Layer 2:</b> LSTM ({config.LSTM_UNITS_2} Units, Dropout = {config.DROPOUT_RATE})</li>
                        <li><b>Dense Head:</b> {config.DENSE_UNITS} Units (ReLU activation)</li>
                        <li><b>Output:</b> 1 Continuous Linear Unit (Scalar RUL in cycles)</li>
                        <li><b>Optimizer:</b> Adam (Initial Learning Rate = {config.LEARNING_RATE})</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metrics = metadata.get("metrics", {})
            st.markdown(
                f"""
                <div class="spec-block">
                    <div class="spec-block-title">Training & Evaluation Performance</div>
                    <ul class="spec-list">
                        <li><b>Train Split MAE:</b> <code>{metrics.get('train_mae', 'N/A')} cycles</code></li>
                        <li><b>Validation Split MAE:</b> <code>{metrics.get('val_mae', 'N/A')} cycles</code></li>
                        <li><b>Test Fleet MAE:</b> <code>{metrics.get('test_mae', 'N/A')} cycles</code></li>
                        <li><b>Test Fleet RMSE:</b> <code>{metrics.get('test_rmse', 'N/A')} cycles</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_right:
            st.markdown(
                f"""
                <div class="spec-block" style="margin-bottom: 20px;">
                    <div class="spec-block-title">Piecewise Linear RUL Target Rationale</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                        In early flight cycles (&gt;125 cycles before failure), turbofan engines operate nominally with zero measurable degradation.<br/><br/>
                        TurboLife AI caps target RUL at <b>{config.RUL_CAP} cycles</b>:
                    </div>
                    <div style="background: #080d18; border: 1px solid #1a2742; border-radius: 8px; padding: 10px 14px; margin: 10px 0; text-align: center; color: #38bdf8; font-weight: 600;">
                        RUL<sub>target</sub> = min(t<sub>failure</sub> − t, 125)
                    </div>
                    <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                        This eliminates training on flat, uninformative noise and maximizes network sensitivity to the final exponential wear curve.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="spec-block">
                    <div class="spec-block-title">Industrial & Operational Disclaimers</div>
                    <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.55;">
                        • <b>Benchmark Dataset:</b> NASA C-MAPSS is a laboratory simulated turbofan dataset generated under controlled fault-injection regimes.<br/><br/>
                        • <b>Production Requirements:</b> Real-world deployment requires certified onboard ACMS / FOQA / QAR telemetry, maintenance records, and OEM airworthiness validation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Diagnostic Plots
        if os.path.exists(config.TRAINING_HISTORY_PLOT) or os.path.exists(config.ACTUAL_VS_PREDICTED_PLOT):
            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
            st.markdown("##### Diagnostic Training & Evaluation Figures")
            ip1, ip2 = st.columns(2, gap="large")
            with ip1:
                if os.path.exists(config.TRAINING_HISTORY_PLOT):
                    st.image(str(config.TRAINING_HISTORY_PLOT), caption="Loss & MAE Convergence Across Epochs")
            with ip2:
                if os.path.exists(config.ACTUAL_VS_PREDICTED_PLOT):
                    st.image(str(config.ACTUAL_VS_PREDICTED_PLOT), caption="Test Fleet Ground Truth vs Predicted RUL")


if __name__ == "__main__":
    main()
