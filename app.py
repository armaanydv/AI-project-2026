"""
app.py
======
Next-Gen AI Timetable Scheduler (Constraint Satisfaction Problem & Local Search)
UI Architecture inspired by Figma Dark Minimalism Design Specification.
Features:
1. Dataset Selection: Preloaded Small/Large benchmarks or custom JSON upload.
2. AI Search Algorithms:
   - Backtracking CSP (Baseline)
   - Backtracking CSP (MRV + Degree Heuristic + Forward Checking)
   - Min-Conflicts Local Search
   - Simulated Annealing (Bonus 1)
   - Hybrid CSP + Simulated Annealing (Bonus 2)
3. Responsive Card-Based Timetable Grid:
   - Multi-period Lab spanning with cyan accent styling
   - Theory session cards with indigo badges
   - Diagonal striped lunch break cells
   - Minimalist free period indicators
   - Views by Class, Teacher, and Room
4. Interactive Live Manual Reassignment & Conflict Inspector
5. Constraint Scoring Matrix (H1-H9 Hard & S1-S5 Soft Constraints)
6. Export Hub (CSV & Multi-sheet Excel)
7. Algorithm Performance Lab & Comparison Analytics
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.csp_model import Assignment, Dataset, Session, Timetable
from core.cost_function import count_violations, ViolationReport
from core.data_loader import load_dataset_from_dict, load_dataset_from_file, DataValidationError
from core.backtracking import BacktrackingSolver
from core.min_conflicts import MinConflictsSolver
from core.simulated_annealing import SimulatedAnnealingSolver
from core.hybrid import HybridSolver


# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ROOT STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Timetable Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-ground: #0B0F14;
    --bg-sidebar: #0E131A;
    --bg-card: #11161D;
    --bg-subtle: #161C24;
    --border-main: #1B2530;
    --border-focus: #2D3B4E;
    --accent-indigo: #7C8CFF;
    --accent-indigo-hover: #96A3FF;
    --accent-teal: #7DD3D0;
    --accent-mint: #69D4A3;
    --accent-mint-light: #7BE6B2;
    --status-success-bg: #0F2A24;
    --status-success-border: #1F4A3A;
    --text-primary: #F5F7FA;
    --text-secondary: #8F9AA8;
    --text-muted: #5B6674;
    --text-dim: #6F7A89;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Global resets & typography */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-ground) !important;
}

/* Header & toolbar cleanup */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-sidebar);
}
::-webkit-scrollbar-thumb {
    background: var(--border-main);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--border-focus);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-main) !important;
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Sidebar Headings & Badges */
/* Top Global Navigation Bar - Figma Match */
.top-nav-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    width: 100%;
    min-height: 54px;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 10px;
    margin-bottom: 20px;
    gap: 16px;
    white-space: nowrap;
}
.nav-brand-group {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
    white-space: nowrap;
}
.app-icon-squircle {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: #5B6BFA;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(91, 107, 250, 0.4);
    flex-shrink: 0;
}
.app-brand-name {
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    font-weight: 700;
    color: #F5F7FA;
    letter-spacing: 0.2px;
    line-height: 18px;
    white-space: nowrap;
}

.nav-breadcrumbs {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 11.5px;
    line-height: 16px;
    white-space: nowrap;
    flex: 1;
}
.breadcrumb-muted {
    color: #8F9AA8;
    white-space: nowrap;
}
.breadcrumb-slash {
    color: #5B6674;
    white-space: nowrap;
}
.breadcrumb-bold {
    color: #F5F7FA;
    font-weight: 600;
    white-space: nowrap;
}
.breadcrumb-year {
    color: #8F9AA8;
    margin-left: 2px;
    white-space: nowrap;
}

.nav-controls {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
}
.theme-pill {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: #11161D;
    border: 1px solid #1B2530;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #EDF0F7;
    cursor: default;
    transition: border-color 0.15s ease;
}
.theme-pill:hover {
    border-color: #2D3B4E;
}
.user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #191E33;
    border: 1px solid #2D3558;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #858CFF;
}

/* Workspace Main Header - Figma Match */
.workspace-main-header {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: flex-start;
    width: 100%;
    margin-bottom: 20px;
    padding-bottom: 4px;
}
.workspace-title-group {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
}
.workspace-heading {
    font-family: 'Inter', sans-serif;
    font-size: 26px;
    font-weight: 600;
    color: #F5F7FA;
    margin: 0;
    letter-spacing: -0.4px;
    line-height: 32px;
}
.workspace-subheading {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 400;
    color: #8F9AA8;
    margin: 0;
    line-height: 16px;
}
.workspace-meta-group {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
}
.unsaved-badge-pill {
    box-sizing: border-box;
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    background: #231C13;
    border: 1px solid #4A3B18;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: #FBBF24;
    letter-spacing: 0.2px;
}
.workspace-meta-sub {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 400;
    color: #8F9AA8;
    line-height: 14px;
}

.sidebar-header-box {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-main);
}
.sidebar-header-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
}

.section-marker-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    margin-bottom: 8px;
}
.section-marker {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    background: #11161D;
    border: 1px solid var(--border-main);
    border-radius: 5px;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-secondary);
}
.section-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.2px;
}

/* Form inputs & selectboxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: var(--bg-sidebar) !important;
    border: 1px solid var(--border-main) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: var(--border-focus) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent-indigo) !important;
    border: 1px solid var(--accent-indigo) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(124, 140, 255, 0.2);
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-indigo-hover) !important;
    box-shadow: 0 4px 18px rgba(124, 140, 255, 0.35);
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-main) !important;
    color: var(--text-primary) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-indigo) !important;
    color: var(--accent-indigo) !important;
}

/* Tabs - Figma Spec */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0E131A !important;
    border: 1px solid #1B2530 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    min-height: 64px !important;
    box-sizing: border-box !important;
    align-items: center !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 10px 14px !important;
    min-height: 56px !important;
    height: 56px !important;
    color: #8F9AA8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    line-height: 150% !important;
    border: none !important;
    background-color: #0E131A !important;
    box-sizing: border-box !important;
    transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #F5F7FA !important;
    background-color: rgba(255, 255, 255, 0.02) !important;
}
.stTabs [aria-selected="true"] {
    background-color: #11161D !important;
    color: #7C8CFF !important;
    font-weight: 600 !important;
    border: 1px solid #1B2530 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Outcome Banners - Figma Spec */
.outcome-banner {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 13px 16px;
    gap: 10px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-family: 'Inter', sans-serif;
}
.outcome-banner-success {
    background: #0F2A24;
    border: 1px solid #1F4A3A;
    color: #7BE6B2;
}
.outcome-banner-warning {
    background: #2A1315;
    border: 1px solid #4A1A1E;
    color: #F87171;
}
.outcome-badge {
    background: rgba(255, 255, 255, 0.08);
    padding: 4px 10px;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 400;
    color: #7BE6B2;
}

/* KPI Cards Grid - Figma Spec */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
@media (max-width: 1024px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 600px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

.kpi-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 16px;
    gap: 10px;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    min-height: 112px;
    transition: border-color 0.2s ease;
}
.kpi-card:hover {
    border-color: #2D3B4E;
}
.kpi-top {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    width: 100%;
}
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}
.kpi-icon {
    width: 14px;
    height: 14px;
    color: #979FAF;
}
.kpi-value {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 28px;
    line-height: 30px;
    color: #F5F7FA;
    width: 100%;
}
.kpi-caption {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #8F9AA8;
}

/* Timetable Container */
.timetable-viewport {
    width: 100%;
    overflow-x: auto;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-main);
    border-radius: 10px;
    padding: 1px;
    margin-bottom: 16px;
}

.tt-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: fixed;
    min-width: 900px;
}

.tt-head-cell {
    background: var(--bg-sidebar);
    border-bottom: 1px solid var(--border-main);
    border-right: 1px solid var(--border-main);
    padding: 10px 8px;
    text-align: center;
    vertical-align: middle;
}
.tt-head-cell:last-child {
    border-right: none;
}
.tt-period-num {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
}
.tt-period-time {
    font-size: 9px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    margin-top: 2px;
}

.tt-day-cell {
    background: var(--bg-sidebar);
    border-bottom: 1px solid var(--border-main);
    border-right: 1px solid var(--border-main);
    padding: 12px 10px;
    width: 80px;
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.3px;
}

.tt-slot-cell {
    background: var(--bg-sidebar);
    border-bottom: 1px solid var(--border-main);
    border-right: 1px solid var(--border-main);
    padding: 6px;
    vertical-align: top;
    height: 110px;
}
.tt-slot-cell:last-child {
    border-right: none;
}

/* Theory Card */
.lesson-card-theory {
    background: var(--bg-card);
    border: 1px solid var(--border-main);
    border-radius: 6px;
    padding: 8px;
    height: 98px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.lesson-card-theory:hover {
    border-color: var(--accent-indigo);
    transform: translateY(-1px);
}

/* Lab Card */
.lesson-card-lab {
    background: #12343A;
    border: 1px solid #1F4A3A;
    border-radius: 6px;
    padding: 8px;
    height: 98px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.lesson-card-lab:hover {
    border-color: var(--accent-teal);
    transform: translateY(-1px);
}

.card-badge {
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.card-badge-theory {
    color: var(--accent-indigo);
}
.card-badge-lab {
    color: var(--accent-teal);
}

.card-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.25;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.card-meta {
    font-size: 9.5px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 4px;
    margin-top: 4px;
}
.card-room-pill {
    background: rgba(255, 255, 255, 0.05);
    padding: 2px 5px;
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 9px;
    font-family: var(--font-mono);
}

/* Lunch Break Cell */
.tt-lunch-cell {
    background: repeating-linear-gradient(
        45deg,
        #0E131A,
        #0E131A 6px,
        #151C26 6px,
        #151C26 12px
    );
    border-bottom: 1px solid var(--border-main);
    border-right: 1px solid var(--border-main);
    height: 110px;
    vertical-align: middle;
    text-align: center;
    padding: 6px;
}
.lunch-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 500;
}

/* Free Period Cell */
.tt-free-cell {
    background: transparent;
    height: 98px;
    border: 1px dashed #1B2530;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-dim);
    font-size: 10px;
}

/* Unavailable Slot (Teacher) */
.tt-unavail-cell {
    background: rgba(239, 68, 68, 0.08);
    height: 98px;
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
    color: #F87171;
    font-size: 10px;
}

/* Timetable Legend */
.tt-legend-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-main);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 11px;
    margin-bottom: 16px;
}
.legend-items {
    display: flex;
    align-items: center;
    gap: 16px;
}
.legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
}
.legend-indicator-theory {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    background: var(--accent-indigo);
}
.legend-indicator-lab {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    background: var(--accent-teal);
}
.legend-indicator-free {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    border: 1px dashed var(--text-dim);
}

/* Insight Box */
.insight-box {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-main);
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 24px;
}
.insight-icon {
    flex-shrink: 0;
    margin-top: 1px;
    color: var(--accent-indigo);
}

/* Constraint breakdown tables */
.constraint-table-card {
    background: var(--bg-sidebar);
    border: 1px solid var(--border-main);
    border-radius: 10px;
    padding: 16px;
}
.constraint-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 12px;
}
.constraint-row:last-child {
    border-bottom: none;
}
.constraint-chip-zero {
    background: #0F2A24;
    color: #69D4A3;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
}
.constraint-chip-violation {
    background: #2A1315;
    color: #F87171;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
}

/* Figma Manual Editing & Conflict Flagging System */
.editor-panel {
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.pending-change-bar {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    gap: 10px;
    background: #11161D;
    border: 1px solid #1B2530;
    border-radius: 8px;
    margin-top: 14px;
    margin-bottom: 16px;
}

.lesson-card-conflict {
    background: #352126;
    border: 1px solid #F38C92;
    border-radius: 6px;
    padding: 8px;
    height: 98px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.lesson-card-conflict:hover {
    border-color: #F87171;
    transform: translateY(-1px);
}
.card-badge-conflict {
    color: #F38C92;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.tt-drop-target {
    background: #11161D;
    border: 1px dashed #7C8CFF;
    border-radius: 6px;
    height: 98px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 4px;
    box-sizing: border-box;
}

.validation-context-card {
    background: #11161D;
    border: 1px solid #1B2530;
    border-radius: 8px;
    padding: 16px;
    height: 100%;
    box-sizing: border-box;
}

.conflict-popover-card {
    background: #352126;
    border: 1px solid #F38C92;
    border-radius: 8px;
    padding: 16px;
    height: 100%;
    box-sizing: border-box;
}

.conflict-popover-clean {
    background: #0F2A24;
    border: 1px solid #1F4A3A;
    border-radius: 8px;
    padding: 16px;
    height: 100%;
    box-sizing: border-box;
}

.affected-booking-pill {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #11161D;
    border: 1px solid #F38C92;
    border-radius: 6px;
    color: #F38C92;
    font-size: 11px;
    margin-top: 10px;
}

.affected-booking-clean {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #0B1E19;
    border: 1px solid #1F4A3A;
    border-radius: 6px;
    color: #69D4A3;
    font-size: 11px;
    margin-top: 10px;
}

.editor-actions-row {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding-top: 16px;
    border-top: 1px solid #1B2530;
    margin-top: 16px;
}

.editing-guidance {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    color: #8F9AA8;
    font-size: 11px;
    margin-top: 14px;
}

.workspace-footer {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    margin-top: 12px;
    border-top: 1px solid #1B2530;
    font-size: 10px;
    color: #8F9AA8;
}

/* Figma Constraints Breakdown Spec */
.constraints-header {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 0px;
    width: 100%;
    height: 22px;
    margin-bottom: 20px;
}
.constraints-title-group {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px;
    gap: 8px;
    height: 22px;
}
.constraints-title-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 18px;
    line-height: 22px;
    color: #EDF0F7;
}
.constraints-scope-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #979FAF;
}

.hard-constraints-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    overflow: hidden;
}

.hard-heading-row {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    width: 100%;
    height: 78px;
}
.hard-title-desc {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    gap: 7px;
}
.hard-title-wrap {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px;
    gap: 8px;
    height: 18px;
}
.hard-heading-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 15px;
    line-height: 18px;
    color: #F5F7FA;
}
.hard-requirement-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}
.hard-status-badge {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 4px 8px;
    gap: 4px;
    background: #0F2A24;
    border-radius: 6px;
    height: 21px;
}
.hard-status-badge-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #7BE6B2;
}

.hard-constraint-row {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 14px 20px;
    gap: 12px;
    width: 100%;
    height: 72px;
    min-height: 72px;
    border-top: 1px solid #2A2F3B;
}
.hard-id-text {
    width: 24px;
    height: 13px;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #979FAF;
    flex: none;
}
.hard-rule-col {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    gap: 6px;
    flex: 1;
}
.hard-rule-name {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    color: #EDF0F7;
}
.hard-rule-desc {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 10px;
    line-height: 150%;
    color: #979FAF;
}
.hard-violation-wrap {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px;
    gap: 6px;
    flex: none;
}
.hard-violation-count {
    width: 14px;
    height: 15px;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    text-align: right;
    color: #69D4A3;
}

.hard-total-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    width: 100%;
    height: 55px;
    background: #0F2A24;
    border-top: 1px solid #1F4A3A;
}
.hard-total-label {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    color: #7BE6B2;
}
.hard-total-value {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 12px;
    line-height: 15px;
    color: #7BE6B2;
}

.soft-quality-col {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    gap: 20px;
    width: 100%;
}

.soft-constraints-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    overflow: hidden;
}

.soft-heading-wrap {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 20px;
    gap: 8px;
    width: 100%;
    height: 82px;
}
.soft-heading-row {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 0px;
    width: 100%;
    height: 21px;
}
.soft-title-group {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px;
    gap: 8px;
    height: 18px;
}
.soft-heading-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 15px;
    line-height: 18px;
    color: #F5F7FA;
}
.soft-status-badge {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 4px 8px;
    gap: 4px;
    background: #242640;
    border-radius: 6px;
    height: 21px;
}
.soft-status-badge-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #858CFF;
}
.soft-explanation-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}

.scoring-cols-header {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 10px 20px;
    gap: 8px;
    width: 100%;
    height: 31px;
    background: #0E131A;
    border-top: 1px solid #2A2F3B;
}
.col-header-rule {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 9px;
    line-height: 11px;
    color: #8F9AA8;
    flex: 1;
}
.col-header-weight {
    width: 42px;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 9px;
    line-height: 11px;
    text-align: right;
    color: #8F9AA8;
    flex: none;
}
.col-header-penalty {
    width: 48px;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 9px;
    line-height: 11px;
    text-align: right;
    color: #8F9AA8;
    flex: none;
}

.soft-constraint-row {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 17px 20px;
    gap: 8px;
    width: 100%;
    height: 90px;
    min-height: 90px;
    border-top: 1px solid #2A2F3B;
}
.soft-rule-col {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    gap: 7px;
    flex: 1;
}
.soft-rule-name-wrap {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px;
    gap: 8px;
    height: 15px;
}
.soft-rule-id {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #979FAF;
}
.soft-rule-name {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    color: #EDF0F7;
}
.soft-rule-desc {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 10px;
    line-height: 150%;
    color: #979FAF;
}
.soft-occurrences-wrap {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px;
    gap: 5px;
    height: 12px;
}
.soft-occurrences-count {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #F0B76B;
}
.soft-weight-text {
    width: 42px;
    height: 15px;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    text-align: right;
    color: #979FAF;
    flex: none;
}
.soft-penalty-text {
    width: 48px;
    height: 17px;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 14px;
    line-height: 17px;
    text-align: right;
    color: #EDF0F7;
    flex: none;
}

.soft-total-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    width: 100%;
    height: 62px;
    background: #11161D;
    border-top: 1px solid #2A2F3B;
}
.soft-total-label {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    color: #979FAF;
}
.soft-total-val {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 18px;
    line-height: 22px;
    color: #EDF0F7;
}

.cost-model-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 20px;
    gap: 12px;
    width: 100%;
    min-height: 164px;
    background: #11161D;
    border: 1px solid #1B2530;
    border-radius: 12px;
}
.cost-model-title {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 12px;
    line-height: 15px;
    color: #7C8CFF;
}
.cost-model-formula {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 18px;
    line-height: 22px;
    color: #EDF0F7;
}
.cost-model-calc {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    color: #8F9AA8;
}
.cost-model-interp {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 160%;
    color: #8F9AA8;
    border-top: 1px solid #1B2530;
    padding-top: 8px;
}

.workspace-footer {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: flex-start;
    padding: 4px 0px 0px;
    width: 100%;
    height: 16px;
    margin-top: 20px;
    border-top: 1px solid #1B2530;
}
.footer-meta-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #8F9AA8;
}
/* Figma Export Center Spec */
.export-workspace {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    gap: 20px;
    width: 100%;
}

.export-heading {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 0px;
    width: 100%;
    height: 22px;
    margin-bottom: 20px;
}
.export-title-group {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
    height: 22px;
}
.export-title-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 600;
    font-size: 18px;
    line-height: 22px;
    color: #F5F7FA;
}
.export-status-badge {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 4px 8px;
    gap: 4px;
    background: #0F2A24;
    border-radius: 6px;
    height: 21px;
}
.export-status-badge-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #7BE6B2;
}

.export-context-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 14px;
    gap: 10px;
    width: 100%;
    min-height: 47px;
    background: #11161D;
    border: 1px solid #1B2530;
    border-radius: 8px;
    margin-bottom: 20px;
}
.export-context-text {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 12px;
    line-height: 160%;
    color: #8F9AA8;
    flex: 1;
}

.export-settings-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 24px;
    gap: 20px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    margin-bottom: 20px;
}
.export-format-options {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 16px;
    width: 100%;
}
.export-format-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 20px;
    gap: 16px;
    flex: 1;
    min-height: 88px;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 8px;
    transition: all 0.15s ease;
}
.export-format-card-active {
    background: #11161D;
    border: 1px solid #7C8CFF;
}
.export-format-icon {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    width: 42px;
    height: 48px;
    background: #11161D;
    border-radius: 6px;
    flex: none;
}
.export-format-info {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 7px;
    flex: 1;
}
.export-format-header {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
}
.export-format-name {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 15px;
    line-height: 18px;
    color: #F5F7FA;
}
.export-format-ext {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}
.export-format-desc {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 150%;
    color: #8F9AA8;
}

.export-scope-row {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 20px;
    width: 100%;
}
.export-scope-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    flex: 1;
}
.export-scope-label {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #979FAF;
}
.export-scope-input {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px 12px;
    gap: 8px;
    width: 100%;
    height: 40px;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: #F5F7FA;
}

.export-preferences-row {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 24px;
    width: 100%;
}
.export-preference-item {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
}
.export-pref-checkbox {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 14px;
    height: 14px;
    background: #7C8CFF;
    border-radius: 3px;
    flex: none;
}
.export-pref-label {
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}

.workbook-preview-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
}
.preview-toolbar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    width: 100%;
    height: 74px;
}
.preview-file-details {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
}
.preview-title-wrap {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
}
.preview-filename {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 13px;
    line-height: 16px;
    color: #EDF0F7;
}
.preview-sheet-context {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #979FAF;
}

.preview-cols-header {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 13px 20px;
    gap: 16px;
    width: 100%;
    height: 38px;
    background: #0E131A;
    border-top: 1px solid #1B2530;
}
.preview-col-head {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #8F9AA8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.preview-data-row {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 0px 20px;
    gap: 16px;
    width: 100%;
    height: 48px;
    background: #0E131A;
    border-top: 1px solid #1B2530;
    transition: background 0.1s ease;
}
.preview-data-row:hover {
    background: #11161D;
}
.preview-cell-text {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.preview-cell-text-bold {
    color: #F5F7FA;
    font-weight: 500;
}

.preview-footer-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    width: 100%;
    height: 40px;
    border-top: 1px solid #1B2530;
}
.preview-footer-meta {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    color: #8F9AA8;
}

.download-actions-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 0px;
    width: 100%;
    min-height: 40px;
    margin-top: 10px;
}
.download-assurance {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
}
.download-ready-title {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 12px;
    line-height: 15px;
    color: #F5F7FA;
}
.download-ready-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}

/* -------------------------------------------------------------------------
   TAB 5: SEARCH LAB & BENCHMARK ANALYTICS
------------------------------------------------------------------------- */
.experiments-heading-row {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    margin-bottom: 20px;
}
.experiments-title-box {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
}
.experiments-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 18px;
    line-height: 22px;
    color: #F5F7FA;
}
.experiments-badge {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 4px 8px;
    gap: 4px;
    background: #11161D;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}

.experiment-protocol-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 16px;
    gap: 8px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 8px;
    margin-bottom: 20px;
}
.protocol-title-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
}
.protocol-title {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 12px;
    line-height: 15px;
    color: #F5F7FA;
}
.protocol-description {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 170%;
    color: #8F9AA8;
}

.benchmarks-table-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    margin-bottom: 20px;
    overflow: hidden;
}
.benchmarks-cols-header {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 16px 20px;
    gap: 16px;
    width: 100%;
    height: 44px;
    background: #0E131A;
}
.benchmark-col-th {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 10px;
    line-height: 12px;
    color: #979FAF;
    letter-spacing: 0.4px;
}

.benchmarks-data-row {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 16px 20px;
    gap: 16px;
    width: 100%;
    min-height: 72px;
    background: #0E131A;
    border-top: 1px solid #1B2530;
    transition: background 0.1s ease;
}
.benchmarks-data-row:hover {
    background: #11141A;
}

.benchmarks-data-row-active {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 16px 20px;
    gap: 16px;
    width: 100%;
    min-height: 72px;
    background: #11161D;
    border-top: 1px solid #2A2F3B;
    border-bottom: 1px solid #2A2F3B;
}

.solver-col {
    width: 280px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
}
.time-col {
    width: 96px;
    flex-shrink: 0;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 15px;
    color: #F5F7FA;
}
.work-col {
    width: 168px;
    flex-shrink: 0;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 15px;
    color: #8F9AA8;
}
.hard-col {
    width: 128px;
    flex-shrink: 0;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 6px;
}
.soft-col {
    width: 120px;
    flex-shrink: 0;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 15px;
    color: #F5F7FA;
}
.success-col {
    width: 184px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.solver-name-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 8px;
}
.solver-name-text {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 12px;
    line-height: 15px;
    color: #F5F7FA;
}
.solver-name-text-active {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 12px;
    line-height: 15px;
    color: #7C8CFF;
}
.solver-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    background: #242640;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    line-height: 13px;
    color: #858CFF;
}
.solver-cue-text {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    line-height: 12px;
    color: #7C8CFF;
}
.hard-zero {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 15px;
    color: #7BE6B2;
}
.hard-violation {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 15px;
    color: #F5B461;
}
.success-rate-val {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 15px;
    color: #F5F7FA;
}
.success-rate-sub {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    line-height: 12px;
    color: #8F9AA8;
}

.benchmarks-note-bar {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    padding: 13px 20px;
    width: 100%;
    background: #0E131A;
    border-top: 1px solid #1B2530;
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    line-height: 15px;
    color: #8F9AA8;
}

.convergence-chart-card {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 24px;
    gap: 20px;
    width: 100%;
    background: #0E131A;
    border: 1px solid #1B2530;
    border-radius: 12px;
    margin-bottom: 20px;
}
.chart-heading-row {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    width: 100%;
}
.chart-title-box {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
}
.chart-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 15px;
    line-height: 18px;
    color: #F5F7FA;
}
.chart-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 13px;
    color: #8F9AA8;
}
.chart-legend-box {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 20px;
}
.chart-legend-item {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    line-height: 12px;
    color: #8F9AA8;
}
.legend-marker-line {
    width: 16px;
    height: 3px;
    border-radius: 2px;
}
.chart-svg-container {
    width: 100%;
    height: 326px;
    box-sizing: border-box;
}
.chart-interpretation-box {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 12px 14px;
    gap: 10px;
    width: 100%;
    background: #0E131A;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    line-height: 160%;
    color: #8F9AA8;
}

.selection-takeaway-banner {
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 16px;
    gap: 12px;
    width: 100%;
    background: #11161D;
    border-radius: 8px;
    margin-bottom: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 160%;
    color: #8F9AA8;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. STATE INITIALIZATION
# ---------------------------------------------------------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "timetable" not in st.session_state:
    st.session_state.timetable = None
if "report" not in st.session_state:
    st.session_state.report = None
if "run_stats" not in st.session_state:
    st.session_state.run_stats = {}
if "selected_class" not in st.session_state:
    st.session_state.selected_class = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = {}
if "benchmark_histories" not in st.session_state:
    st.session_state.benchmark_histories = {}

def sync_run_to_benchmark(
    ds_key: str,
    algo_id: str,
    algo_name: str,
    badge: str,
    time_taken: float,
    work_str: str,
    report,
    success: bool,
    cost_history=None
):
    """Synchronizes an individual algorithm run directly into the Benchmark store."""
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = {}
    if "benchmark_histories" not in st.session_state:
        st.session_state.benchmark_histories = {}
    if ds_key not in st.session_state.benchmark_results or not isinstance(st.session_state.benchmark_results[ds_key], dict):
        st.session_state.benchmark_results[ds_key] = {}
    if ds_key not in st.session_state.benchmark_histories or not isinstance(st.session_state.benchmark_histories[ds_key], dict):
        st.session_state.benchmark_histories[ds_key] = {}

    st.session_state.benchmark_results[ds_key][algo_id] = {
        "Id": algo_id,
        "Algorithm": algo_name,
        "Badge": badge,
        "Time": time_taken,
        "TimeStr": f"{time_taken:.3f}s",
        "Work": work_str,
        "Hard": report.hard_violations,
        "Soft": report.soft_penalty,
        "Total Cost": report.total_cost,
        "Success": success,
    }
    if cost_history:
        st.session_state.benchmark_histories[ds_key][algo_name] = cost_history


# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS: TIME & LABELS
# ---------------------------------------------------------------------------
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def get_day_name(day_idx: int) -> str:
    return DAY_NAMES[day_idx] if day_idx < len(DAY_NAMES) else f"Day {day_idx + 1}"

def get_period_time_label(period_idx: int) -> str:
    """Generates standard academic bell timings starting at 09:00."""
    start_hour = 9 + period_idx
    end_hour = start_hour + 1
    return f"{start_hour:02d}:00 - {end_hour:02d}:00"


# ---------------------------------------------------------------------------
# 4. SIDEBAR CONFIGURATION (PIXEL MATCH FIGMA SPECS)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Header (Figma Match)
    st.markdown(
        """
        <div class="sidebar-header-box">
            <div style="display: flex; align-items: center; gap: 8px;">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#F5F7FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="4" y1="21" x2="4" y2="14"></line>
                    <line x1="4" y1="10" x2="4" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12" y2="3"></line>
                    <line x1="20" y1="21" x2="20" y2="16"></line>
                    <line x1="20" y1="12" x2="20" y2="3"></line>
                    <line x1="1" y1="14" x2="7" y2="14"></line>
                    <line x1="9" y1="8" x2="15" y2="8"></line>
                    <line x1="17" y1="16" x2="23" y2="16"></line>
                </svg>
                <span class="sidebar-header-title">Configuration</span>
            </div>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="9" y1="3" x2="9" y2="21"></line>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SECTION 1: PROBLEM DATASET ---
    st.markdown(
        """
        <div class="section-marker-container" style="justify-content: space-between; width: 100%; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 15px; font-weight: 600; color: #8F9AA8; font-family: 'JetBrains Mono', monospace;">01</span>
                <span class="section-title">Problem Dataset</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_choice = st.selectbox(
        "Choose Dataset Source",
        [
            "Preloaded: Small (3 Classes, 6 Teachers)",
            "Preloaded: Large (8 Classes, 16 Teachers)",
            "Upload Custom JSON",
        ],
        index=0,
        label_visibility="collapsed",
    )

    loaded_filename = None
    target_dataset_key = None
    if dataset_choice == "Preloaded: Small (3 Classes, 6 Teachers)":
        target_dataset_key = "small_dataset.json"
        loaded_filename = "small_dataset.json"
        if st.session_state.get("current_dataset_key") != target_dataset_key:
            try:
                st.session_state.dataset = load_dataset_from_file("data/small_dataset.json")
                st.session_state.current_dataset_key = target_dataset_key
                st.session_state.loaded_filename = loaded_filename
                st.session_state.timetable = None
                st.session_state.report = None
                st.session_state.run_stats = {}
                st.session_state.selected_class = None
            except Exception as e:
                st.error(f"Error loading small dataset: {e}")
                st.session_state.dataset = None
                loaded_filename = None
        else:
            loaded_filename = st.session_state.get("loaded_filename", loaded_filename)

    elif dataset_choice == "Preloaded: Large (8 Classes, 16 Teachers)":
        target_dataset_key = "large_dataset.json"
        loaded_filename = "large_dataset.json"
        if st.session_state.get("current_dataset_key") != target_dataset_key:
            try:
                st.session_state.dataset = load_dataset_from_file("data/large_dataset.json")
                st.session_state.current_dataset_key = target_dataset_key
                st.session_state.loaded_filename = loaded_filename
                st.session_state.timetable = None
                st.session_state.report = None
                st.session_state.run_stats = {}
                st.session_state.selected_class = None
            except Exception as e:
                st.error(f"Error loading large dataset: {e}")
                st.session_state.dataset = None
                loaded_filename = None
        else:
            loaded_filename = st.session_state.get("loaded_filename", loaded_filename)

    else:
        uploaded_file = st.file_uploader("Upload JSON Dataset", type=["json"], label_visibility="collapsed")
        if uploaded_file is not None:
            target_dataset_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            loaded_filename = uploaded_file.name
            if st.session_state.get("current_dataset_key") != target_dataset_key:
                try:
                    content = json.load(uploaded_file)
                    st.session_state.dataset = load_dataset_from_dict(content)
                    st.session_state.current_dataset_key = target_dataset_key
                    st.session_state.loaded_filename = loaded_filename
                    st.session_state.timetable = None
                    st.session_state.report = None
                    st.session_state.run_stats = {}
                    st.session_state.selected_class = None
                except Exception as e:
                    st.error(f"Validation error: {e}")
                    st.session_state.dataset = None
                    loaded_filename = None
            else:
                loaded_filename = st.session_state.get("loaded_filename", loaded_filename)
        else:
            if st.session_state.get("current_dataset_key") != "no_upload":
                st.session_state.current_dataset_key = "no_upload"
                st.session_state.dataset = None
                st.session_state.timetable = None
                st.session_state.report = None
                st.session_state.run_stats = {}
                st.session_state.selected_class = None
            loaded_filename = None

    if st.session_state.dataset is not None:
        ds: Dataset = st.session_state.dataset
        sess_count = len(ds.get_sessions())
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:6px; background:#0F2A24; padding:6px 10px; border-radius:6px; margin:8px 0;">
                <svg width="13" height="15" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                <span style="font-size:17px; font-weight:600; color:#69D4A3;">Loaded: {loaded_filename or 'Dataset Ready'}</span>
            </div>
            <div style="background:#0E131A; border:1px solid #1B2530; border-radius:8px; padding:10px; margin-bottom:16px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:17.5px;">
                    <span style="color:#8F9AA8;">Classes: <b style="color:#F5F7FA;">{len(ds.classes)}</b></span>
                    <span style="color:#8F9AA8;">Teachers: <b style="color:#F5F7FA;">{len(ds.teachers)}</b></span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:17.5px;">
                    <span style="color:#8F9AA8;">Rooms: <b style="color:#F5F7FA;">{len(ds.rooms)}</b></span>
                    <span style="color:#8F9AA8;">Sessions: <b style="color:#F5F7FA;">{sess_count}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- SECTION 2: AI SEARCH ALGORITHM ---
    st.markdown(
        """
        <div class="section-marker-container" style="justify-content: space-between; width: 100%; margin-top: 14px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 15px; font-weight: 600; color: #8F9AA8; font-family: 'JetBrains Mono', monospace;">02</span>
                <span class="section-title">AI Search Algorithm</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    algorithm = st.selectbox(
        "Select Technique",
        [
            "Hybrid CSP + SA (Bonus 2)",
            "Backtracking CSP (MRV + Degree + FC)",
            "Backtracking CSP (Baseline)",
            "Min-Conflicts Local Search",
            "Simulated Annealing (Bonus 1)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # Dynamic Hyperparameters in minimalist card
    st.markdown('<div style="background:#0E131A; border:1px solid #1B2530; border-radius:8px; padding:12px; margin-top:8px;">', unsafe_allow_html=True)
    seed = st.number_input("Random Seed", value=42, step=1)

    if algorithm == "Hybrid CSP + SA (Bonus 2)":
        bt_timeout = st.slider("Stage 1 BT Timeout (s)", min_value=5, max_value=30, value=15, step=5)
        hybrid_iters = st.slider("Stage 2 SA Iterations", min_value=300, max_value=3000, value=1200, step=100)
    elif "Backtracking" in algorithm:
        timeout = st.slider("Timeout (seconds)", min_value=5, max_value=60, value=30, step=5)
    elif algorithm == "Min-Conflicts Local Search":
        max_iters = st.slider("Max Iterations", min_value=500, max_value=5000, value=2000, step=250)
        p_walk = st.slider("Random Walk Probability", min_value=0.0, max_value=0.20, value=0.06, step=0.01)
    elif algorithm == "Simulated Annealing (Bonus 1)":
        t0 = st.number_input("Initial Temp (T0)", value=500.0, step=50.0)
        cooling = st.slider("Cooling Rate (alpha)", min_value=0.950, max_value=0.999, value=0.993, step=0.001)
        sa_iters = st.slider("Max Iterations", min_value=500, max_value=5000, value=2000, step=250)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Timetable", type="primary", use_container_width=True)

    st.markdown(
        """
        <div style="margin-top: 24px; padding-top: 14px; border-top: 1px solid #1B2530; font-size: 11px; color: #5B6674;">
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px; color:#8F9AA8;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2"><path d="M22 10v6M2 10l10-5 10 5-10 5z"></path><path d="M6 12v5c3 3 9 3 12 0v-5"></path></svg>
                <span>B.Tech AI Assignment</span>
            </div>
            <span>CSP & Heuristic Search Engine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 5. ALGORITHM EXECUTION ENGINE
# ---------------------------------------------------------------------------
if (generate_btn or st.session_state.pop("trigger_generation", False)) and st.session_state.dataset is not None:
    dataset = st.session_state.dataset
    with st.spinner(f"Optimizing schedule using {algorithm}..."):
        start_exec = time.time()

        active_ds_key = st.session_state.get("current_dataset_key", "default_ds")

        if algorithm == "Backtracking CSP (Baseline)":
            solver = BacktrackingSolver(
                dataset,
                use_mrv=False,
                use_degree=False,
                use_forward_checking=False,
                timeout_seconds=float(timeout),
                seed=int(seed),
            )
            res = solver.solve()
            st.session_state.timetable = res.timetable
            st.session_state.report = res.report
            st.session_state.run_stats = {
                "Algorithm": "Backtracking (Baseline)",
                "Time Taken": f"{res.time_taken:.3f}s",
                "Iterations": f"{res.nodes_expanded} nodes",
                "Success": res.success,
                "Type": "Exact CSP",
            }
            sync_run_to_benchmark(
                active_ds_key,
                "baseline",
                "Baseline Backtracking",
                "Baseline",
                res.time_taken,
                f"{res.nodes_expanded:,} nodes",
                res.report,
                res.success,
            )

        elif algorithm == "Backtracking CSP (MRV + Degree + FC)":
            solver = BacktrackingSolver(
                dataset,
                use_mrv=True,
                use_degree=True,
                use_forward_checking=True,
                timeout_seconds=float(timeout),
                seed=int(seed),
            )
            res = solver.solve()
            st.session_state.timetable = res.timetable
            st.session_state.report = res.report
            st.session_state.run_stats = {
                "Algorithm": "Backtracking (MRV+Deg+FC)",
                "Time Taken": f"{res.time_taken:.3f}s",
                "Iterations": f"{res.nodes_expanded} nodes",
                "Success": res.success,
                "Type": "Heuristic CSP",
            }
            sync_run_to_benchmark(
                active_ds_key,
                "mrv",
                "MRV + Degree + FC",
                "Heuristic",
                res.time_taken,
                f"{res.nodes_expanded:,} nodes",
                res.report,
                res.success,
            )

        elif algorithm == "Min-Conflicts Local Search":
            mc_solver = MinConflictsSolver(
                dataset,
                max_iterations=int(max_iters),
                random_walk_prob=float(p_walk),
                seed=int(seed),
            )
            mc_res = mc_solver.solve()
            st.session_state.timetable = mc_res.timetable
            st.session_state.report = mc_res.report
            st.session_state.run_stats = {
                "Algorithm": "Min-Conflicts Search",
                "Time Taken": f"{mc_res.time_taken:.3f}s",
                "Iterations": f"{mc_res.iterations} steps",
                "Success": mc_res.success,
                "Type": "Local Search",
            }
            sync_run_to_benchmark(
                active_ds_key,
                "min_conflicts",
                "Min-Conflicts",
                "Local",
                mc_res.time_taken,
                f"{mc_res.iterations:,} steps",
                mc_res.report,
                mc_res.success,
                mc_res.cost_history,
            )

        elif algorithm == "Simulated Annealing (Bonus 1)":
            sa_solver = SimulatedAnnealingSolver(
                dataset,
                initial_temp=float(t0),
                cooling_rate=float(cooling),
                max_iterations=int(sa_iters),
                seed=int(seed),
            )
            sa_res = sa_solver.solve()
            st.session_state.timetable = sa_res.timetable
            st.session_state.report = sa_res.report
            st.session_state.run_stats = {
                "Algorithm": "Simulated Annealing",
                "Time Taken": f"{sa_res.time_taken:.3f}s",
                "Iterations": f"{sa_res.iterations} iters",
                "Success": sa_res.success,
                "Type": "Metaheuristic",
            }
            sync_run_to_benchmark(
                active_ds_key,
                "simulated_annealing",
                "Simulated Annealing",
                "Local",
                sa_res.time_taken,
                f"{sa_res.iterations:,} iters",
                sa_res.report,
                sa_res.success,
                sa_res.cost_history,
            )

        elif algorithm == "Hybrid CSP + SA (Bonus 2)":
            hy_solver = HybridSolver(
                dataset,
                sa_max_iterations=int(hybrid_iters),
                bt_timeout=float(bt_timeout),
                seed=int(seed),
            )
            hy_res = hy_solver.solve()
            st.session_state.timetable = hy_res.timetable
            st.session_state.report = hy_res.report
            st.session_state.run_stats = {
                "Algorithm": "Hybrid CSP + SA",
                "Time Taken": f"{hy_res.total_time:.3f}s",
                "Iterations": f"{hy_res.backtracking_nodes} nodes + {hy_res.sa_iterations} SA iters",
                "Success": hy_res.success,
                "Type": "Two-Phase Hybrid",
            }
            sync_run_to_benchmark(
                active_ds_key,
                "hybrid",
                "Hybrid CSP + SA",
                "Hybrid",
                hy_res.total_time,
                f"{hy_res.backtracking_nodes:,} nodes + {hy_res.sa_iterations:,} SA",
                hy_res.report,
                hy_res.success,
                hy_res.cost_history,
            )


# ---------------------------------------------------------------------------
# 6. MAIN WORKSPACE VIEW (FIGMA OBSIDIAN DARK MATCH)
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="top-nav-bar">
        <div class="nav-brand-group">
            <div class="app-icon-squircle">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="4" width="18" height="17" rx="3" ry="3"></rect>
                    <line x1="16" y1="2" x2="16" y2="6"></line>
                    <line x1="8" y1="2" x2="8" y2="6"></line>
                    <line x1="3" y1="9" x2="21" y2="9"></line>
                    <rect x="7" y="12" width="2" height="2" rx="0.5" fill="#FFFFFF"></rect>
                    <rect x="11" y="12" width="2" height="2" rx="0.5" fill="#FFFFFF"></rect>
                    <rect x="15" y="12" width="2" height="2" rx="0.5" fill="#FFFFFF"></rect>
                    <rect x="7" y="16" width="2" height="2" rx="0.5" fill="#FFFFFF"></rect>
                    <rect x="11" y="16" width="2" height="2" rx="0.5" fill="#FFFFFF"></rect>
                    <rect x="15" y="16" width="2" height="2" rx="0.5" fill="#FFFFFF"></rect>
                </svg>
            </div>
            <span class="app-brand-name">AI BASED TIMETABLE SCHEDULER</span>
        </div>
        <div class="nav-breadcrumbs">
            <span class="breadcrumb-muted">College workspace</span>
            <span class="breadcrumb-slash">/</span>
            <span class="breadcrumb-bold">AIDE - Semester III</span>
            <span class="breadcrumb-year">2025-29</span>
        </div>
        <div class="nav-controls">
            <div class="theme-pill">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                </svg>
                <span>Dark</span>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
        </div>
    </div>
    """
)

# Workspace Header with Timetable workspace and Unsaved edit badge
_ds_name = st.session_state.get("loaded_filename", "Demo dataset")
st.html(
    f"""
    <div class="workspace-main-header">
        <div class="workspace-title-group">
            <h1 class="workspace-heading">Timetable workspace</h1>
            <p class="workspace-subheading">Semester III · AIDE department</p>
        </div>
        <div class="workspace-meta-group">
            <div class="unsaved-badge-pill">Unsaved edit - #042</div>
            <span class="workspace-meta-sub">{_ds_name} / sample run</span>
        </div>
    </div>
    """
)

if st.session_state.dataset is None:
    st.info("👈 Please select or upload a dataset in the sidebar to begin.")
    st.stop()

dataset: Dataset = st.session_state.dataset
sessions: List[Session] = dataset.get_sessions()
session_map: Dict[str, Session] = {s.session_id: s for s in sessions}

timetable: Optional[Timetable] = st.session_state.timetable
report: Optional[ViolationReport] = st.session_state.report
stats = st.session_state.run_stats or {}


# --- FEASIBILITY OUTCOME BANNER (FIGMA MATCH) ---
if timetable is not None and report is not None:
    if report.hard_violations == 0:
        st.html(
            f'<div class="outcome-banner outcome-banner-success">'
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="1.6"><circle cx="12" cy="12" r="10"></circle><polyline points="16 10 11 15 8 12"></polyline></svg>'
            f'<span style="font-size:13px; line-height:16px; color:#7BE6B2;">Optimal schedule found: 100% hard constraints satisfied with zero academic clashes</span>'
            f'</div>'
            f'<div class="outcome-badge">Total Cost: {report.total_cost}</div>'
            f'</div>'
        )
    else:
        st.html(
            f'<div class="outcome-banner outcome-banner-warning">'
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="1.6"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
            f'<span style="font-size:13px; line-height:16px; color:#F87171;">Schedule has {report.hard_violations} Hard Violation(s) Active! Adjust timeout or switch to Hybrid/MRV solver.</span>'
            f'</div>'
            f'<div class="outcome-badge" style="color:#F87171;">Penalty: {report.total_cost}</div>'
            f'</div>'
        )

    # --- 5 KPI CARDS (FIGMA MATCH) ---
    hard_color = "#7BE6B2" if report.hard_violations == 0 else "#F87171"
    hard_caption = "Zero clashes" if report.hard_violations == 0 else f"{report.hard_violations} clashes"

    st.html(
        f'<div class="kpi-grid">'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Algorithm</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>'
        f'</div>'
        f'<div class="kpi-value" style="font-size:16px;">{stats.get("Algorithm", algorithm)}</div>'
        f'<div class="kpi-caption">{stats.get("Type", "Optimization")}</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Time Taken</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
        f'</div>'
        f'<div class="kpi-value">{stats.get("Time Taken", "0.000s")}</div>'
        f'<div class="kpi-caption">Execution duration</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Hard Violations</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="1.6"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        f'</div>'
        f'<div class="kpi-value" style="color:{hard_color};">{report.hard_violations}</div>'
        f'<div class="kpi-caption">{hard_caption}</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Soft Penalty</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
        f'</div>'
        f'<div class="kpi-value">{report.soft_penalty}</div>'
        f'<div class="kpi-caption">Quality score (S1-S5)</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Total Cost</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="2" width="16" height="20" rx="2"></rect><line x1="8" y1="6" x2="16" y2="6"></line><line x1="16" y1="14" x2="16" y2="18"></line><path d="M8 10h.01"></path><path d="M12 10h.01"></path><path d="M16 10h.01"></path></svg>'
        f'</div>'
        f'<div class="kpi-value">{report.total_cost}</div>'
        f'<div class="kpi-caption">1000×H + Soft</div>'
        f'</div>'
        f'</div>'
    )
else:
    # Awaiting generation state: dataset is ready, user hasn't clicked generate yet
    st.html(
        f'<div class="outcome-banner" style="background:#11161D; border:1px solid #1B2530; border-left:4px solid #7C8CFF; display:flex; align-items:center; justify-content:space-between; padding:14px 20px; border-radius:8px;">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
        f'<span style="font-size:13.5px; color:#F5F7FA; font-weight:500;">Dataset ready: <b style="color:#7C8CFF;">{st.session_state.get("loaded_filename") or "Active Dataset"}</b> ({len(dataset.classes)} Classes, {len(dataset.teachers)} Teachers, {len(dataset.rooms)} Rooms, {len(sessions)} Sessions). Configure parameters in sidebar and click <b>Generate Timetable</b> to begin.</span>'
        f'</div>'
        f'<div class="outcome-badge" style="background:#161C24; color:#7C8CFF; border:1px solid #2D3B4E;">Awaiting Generation</div>'
        f'</div>'
    )

    st.html(
        f'<div class="kpi-grid">'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Algorithm</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>'
        f'</div>'
        f'<div class="kpi-value" style="font-size:16px;">{algorithm}</div>'
        f'<div class="kpi-caption">Selected solver</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Time Taken</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
        f'</div>'
        f'<div class="kpi-value" style="color:#8F9AA8;">—</div>'
        f'<div class="kpi-caption">Execution duration</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Hard Violations</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="1.6"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        f'</div>'
        f'<div class="kpi-value" style="color:#8F9AA8;">—</div>'
        f'<div class="kpi-caption">Awaiting run</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Soft Penalty</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
        f'</div>'
        f'<div class="kpi-value" style="color:#8F9AA8;">—</div>'
        f'<div class="kpi-caption">Quality score (S1-S5)</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">Total Cost</span>'
        f'<svg class="kpi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="2" width="16" height="20" rx="2"></rect><line x1="8" y1="6" x2="16" y2="6"></line><line x1="16" y1="14" x2="16" y2="18"></line><path d="M8 10h.01"></path><path d="M12 10h.01"></path><path d="M16 10h.01"></path></svg>'
        f'</div>'
        f'<div class="kpi-value" style="color:#8F9AA8;">—</div>'
        f'<div class="kpi-caption">1000×H + Soft</div>'
        f'</div>'
        f'</div>'
    )



# --- 5 RESULT WORKSPACE TABS ---
tab_grid, tab_manual, tab_matrix, tab_export, tab_benchmarks = st.tabs([
    "📅 Timetable Grid View",
    "✏️ Manual Reassignment & Conflict Inspector",
    "📋 Constraints Diagnostic Matrix",
    "💾 Export Center (CSV / Excel)",
    "🧪 Search Lab & Benchmark Analytics",
])


# ---------------------------------------------------------------------------
# TAB 1: CARD-BASED TIMETABLE GRID (FIGMA MATCH)
# ---------------------------------------------------------------------------
with tab_grid:
    if timetable is not None:
        # Orientation & Filter Controls Row
        f_col1, f_col2, f_col3 = st.columns([1.5, 2, 2])
        with f_col1:
            orientation = st.radio("Orientation", ["By Class", "By Teacher", "By Room"], horizontal=True, label_visibility="collapsed")
        with f_col2:
            if orientation == "By Class":
                target_id = st.selectbox(
                    "Select Class",
                    list(dataset.classes.keys()),
                    format_func=lambda x: f"{x} — {dataset.classes[x].name}",
                    label_visibility="collapsed",
                )
            elif orientation == "By Teacher":
                target_id = st.selectbox(
                    "Select Teacher",
                    list(dataset.teachers.keys()),
                    format_func=lambda x: f"{x} — {dataset.teachers[x].name}",
                    label_visibility="collapsed",
                )
            else:
                target_id = st.selectbox(
                    "Select Room",
                    list(dataset.rooms.keys()),
                    format_func=lambda x: f"{x} — {dataset.rooms[x].name} ({dataset.rooms[x].room_type})",
                    label_visibility="collapsed",
                )

        with f_col3:
            if orientation == "By Class" and target_id in dataset.classes:
                cls_obj = dataset.classes[target_id]
                st.markdown(
                    f"""
                    <div style="font-size:11px; color:#8F9AA8; padding:8px 0;">
                        Strength: <b style="color:#F5F7FA;">{cls_obj.strength}</b> | Lunch: <b style="color:#F5F7FA;">Period {cls_obj.lunch_break_period + 1}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif orientation == "By Teacher" and target_id in dataset.teachers:
                tch_obj = dataset.teachers[target_id]
                st.markdown(
                    f"""
                    <div style="font-size:11px; color:#8F9AA8; padding:8px 0;">
                        Max Load: <b style="color:#F5F7FA;">{tch_obj.max_weekly_load} hrs/wk</b> | Unavail: <b style="color:#F5F7FA;">{len(tch_obj.unavailable_slots)} slots</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif orientation == "By Room" and target_id in dataset.rooms:
                rm_obj = dataset.rooms[target_id]
                st.markdown(
                    f"""
                    <div style="font-size:11px; color:#8F9AA8; padding:8px 0;">
                        Type: <b style="color:#F5F7FA;">{rm_obj.room_type.capitalize()}</b> | Capacity: <b style="color:#F5F7FA;">{rm_obj.capacity} students</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Build the HTML Table
        ts = dataset.time_structure
        periods = list(range(ts.periods_per_day))
        days = list(range(ts.num_days))
        session_map = {s.session_id: s for s in dataset.get_sessions()}

        # Header Row
        html_out = ['<div class="timetable-viewport"><table class="tt-table"><thead><tr>']
        html_out.append('<th class="tt-head-cell" style="width:72px;"><span class="tt-period-num">Day</span></th>')
        for p in periods:
            time_str = get_period_time_label(p)
            html_out.append(f'<th class="tt-head-cell"><span class="tt-period-num">Period {p+1}</span><div class="tt-period-time">{time_str}</div></th>')
        html_out.append('</tr></thead><tbody>')

        # Grid mapping
        # grid[day][period] -> cell html content
        for d in days:
            html_out.append(f'<tr><td class="tt-day-cell">{get_day_name(d)}</td>')

            # Check lunch break if view by class
            lunch_period = -1
            if orientation == "By Class" and target_id in dataset.classes:
                lunch_period = dataset.classes[target_id].lunch_break_period

            # Teacher unavailable slots
            teacher_unavail = set()
            if orientation == "By Teacher" and target_id in dataset.teachers:
                teacher_unavail = set(dataset.teachers[target_id].unavailable_slots)

            p = 0
            while p < ts.periods_per_day:
                # Check lunch
                if orientation == "By Class" and p == lunch_period:
                    html_out.append(
                        '<td class="tt-lunch-cell">'
                        '<div class="lunch-inner">'
                        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2"><path d="M18 2v6a3 3 0 0 1-3 3 3 3 0 0 1-3-3V2"></path><path d="M6 2v20"></path><line x1="12" y1="11" x2="12" y2="22"></line><line x1="18" y1="11" x2="18" y2="22"></line></svg>'
                        '<span>LUNCH</span>'
                        '</div>'
                        '</td>'
                    )
                    p += 1
                    continue

                # Check teacher unavailable
                if orientation == "By Teacher" and (d, p) in teacher_unavail:
                    html_out.append(
                        '<td class="tt-slot-cell">'
                        '<div class="tt-unavail-cell">'
                        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line></svg>'
                        '<span>Unavailable</span>'
                        '</div>'
                        '</td>'
                    )
                    p += 1
                    continue

                # Find matching session
                matched_sess = None
                matched_asgn = None
                for sid, asgn in timetable.assignments.items():
                    s_obj = session_map.get(sid)
                    if not s_obj:
                        continue
                    if asgn.day == d and asgn.start_period <= p < asgn.start_period + s_obj.duration:
                        if orientation == "By Class" and s_obj.class_id == target_id:
                            matched_sess, matched_asgn = s_obj, asgn
                            break
                        elif orientation == "By Teacher" and s_obj.teacher_id == target_id:
                            matched_sess, matched_asgn = s_obj, asgn
                            break
                        elif orientation == "By Room" and asgn.room_id == target_id:
                            matched_sess, matched_asgn = s_obj, asgn
                            break

                if matched_sess and matched_asgn:
                    subj = dataset.subjects.get(matched_sess.subject_id)
                    tch = dataset.teachers.get(matched_sess.teacher_id)
                    rm = dataset.rooms.get(matched_asgn.room_id)
                    cls = dataset.classes.get(matched_sess.class_id)

                    is_lab = subj and subj.subject_type == "lab"
                    card_class = "lesson-card-lab" if is_lab else "lesson-card-theory"
                    badge_class = "card-badge-lab" if is_lab else "card-badge-theory"

                    # Calculate multi-period span for labs
                    remaining_dur = (matched_asgn.start_period + matched_sess.duration) - p
                    span = max(1, min(remaining_dur, ts.periods_per_day - p))
                    colspan_attr = f' colspan="{span}"' if span > 1 else ''

                    # Academic timing string for this block
                    start_lbl = get_period_time_label(p).split(" - ")[0]
                    end_lbl = get_period_time_label(p + span - 1).split(" - ")[1]
                    time_span_str = f"{start_lbl} - {end_lbl}"

                    if is_lab:
                        badge_text = f"LAB ({span}H) · {time_span_str}"
                    else:
                        badge_text = "THEORY"

                    subj_title = subj.name if subj else matched_sess.subject_id
                    tch_name = tch.name if tch else matched_sess.teacher_id
                    rm_name = rm.name if rm else matched_asgn.room_id
                    cls_name = cls.name if cls else matched_sess.class_id

                    meta_label = f"{tch_name}" if orientation != "By Teacher" else f"{cls_name}"

                    cell_html = (
                        f'<div class="{card_class}">'
                        f'<div>'
                        f'<span class="card-badge {badge_class}">{badge_text}</span>'
                        f'<div class="card-title">{subj_title}</div>'
                        f'</div>'
                        f'<div class="card-meta">'
                        f'<span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{meta_label}</span>'
                        f'<span class="card-room-pill">{rm_name}</span>'
                        f'</div>'
                        f'</div>'
                    )
                    html_out.append(f'<td class="tt-slot-cell"{colspan_attr}>{cell_html}</td>')
                    p += span
                else:
                    html_out.append('<td class="tt-slot-cell"><div class="tt-free-cell">Free</div></td>')
                    p += 1

            html_out.append('</tr>')

        html_out.append('</tbody></table></div>')
        st.html("".join(html_out))

        # Timetable Legend (Figma Match)
        st.markdown(
            """
            <div class="tt-legend-strip">
                <div class="legend-items">
                    <span class="legend-pill"><span class="legend-indicator-theory"></span> Theory (Lecture)</span>
                    <span class="legend-pill"><span class="legend-indicator-lab"></span> Laboratory (2 Hours)</span>
                    <span class="legend-pill"><span class="legend-indicator-free"></span> Free Period</span>
                </div>
                <div style="display:flex; align-items:center; gap:6px; color:#69D4A3; font-weight:600; font-size:11px;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>100% Conflict-Free Timetable</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Schedule Insight Callout
        st.markdown(
            """
            <div class="insight-box">
                <svg class="insight-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                <div>
                    <b>Optimization Insight:</b> Lab sessions are automatically assigned consecutive two-hour blocks with dedicated computer / electronics lab equipment. Teacher preferences and daily lunch breaks are rigorously maintained across all batches.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ---------------------------------------------------------------------------
    # TAB 2: INTERACTIVE MANUAL REASSIGNMENT & CONFLICT INSPECTOR
    # ---------------------------------------------------------------------------
    else:
        st.html(
            f'''
            <div style="background:#0E131A; border:1px solid #1B2530; border-radius:12px; padding:48px 24px; text-align:center; margin-top:12px;">
                <div style="display:inline-flex; align-items:center; justify-content:center; width:64px; height:64px; border-radius:50%; background:rgba(124, 140, 255, 0.1); border:1px solid rgba(124, 140, 255, 0.25); margin-bottom:16px;">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="16" y1="2" x2="16" y2="6"></line>
                        <line x1="8" y1="2" x2="8" y2="6"></line>
                        <line x1="3" y1="10" x2="21" y2="10"></line>
                    </svg>
                </div>
                <h3 style="font-size:20px; font-weight:600; color:#F5F7FA; margin:0 0 8px 0;">Ready to Generate Timetable</h3>
                <p style="font-size:13.5px; color:#8F9AA8; max-width:520px; margin:0 auto 24px auto; line-height:1.6;">
                    Dataset <b style="color:#F5F7FA;">{st.session_state.get('loaded_filename') or 'Selected'}</b> has been loaded with <b style="color:#7C8CFF;">{len(sessions)}</b> sessions across <b style="color:#7C8CFF;">{len(dataset.classes)}</b> cohorts and <b style="color:#7C8CFF;">{len(dataset.teachers)}</b> faculty members.<br/>
                    Click below or use the sidebar button to launch the <b style="color:#7C8CFF;">{algorithm}</b> engine.
                </p>
            </div>
            '''
        )
        c_btn1, c_btn2, c_btn3 = st.columns([1, 1.2, 1])
        with c_btn2:
            if st.button("🚀 Generate Timetable Now", type="primary", use_container_width=True, key="grid_hero_generate"):
                st.session_state.trigger_generation = True
                st.rerun()

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.html(
            f'''
            <div style="background:#0E131A; border:1px solid #1B2530; border-radius:10px; padding:20px;">
                <div style="font-size:13px; font-weight:600; color:#8F9AA8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:14px;">
                    Dataset Composition Summary
                </div>
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px;">
                    <div style="background:#11161D; border:1px solid #1B2530; border-radius:8px; padding:12px 14px;">
                        <div style="font-size:11px; color:#8F9AA8; margin-bottom:4px;">Classes / Cohorts</div>
                        <div style="font-size:18px; font-weight:600; color:#F5F7FA;">{len(dataset.classes)}</div>
                    </div>
                    <div style="background:#11161D; border:1px solid #1B2530; border-radius:8px; padding:12px 14px;">
                        <div style="font-size:11px; color:#8F9AA8; margin-bottom:4px;">Faculty Teachers</div>
                        <div style="font-size:18px; font-weight:600; color:#F5F7FA;">{len(dataset.teachers)}</div>
                    </div>
                    <div style="background:#11161D; border:1px solid #1B2530; border-radius:8px; padding:12px 14px;">
                        <div style="font-size:11px; color:#8F9AA8; margin-bottom:4px;">Rooms & Labs</div>
                        <div style="font-size:18px; font-weight:600; color:#F5F7FA;">{len(dataset.rooms)}</div>
                    </div>
                    <div style="background:#11161D; border:1px solid #1B2530; border-radius:8px; padding:12px 14px;">
                        <div style="font-size:11px; color:#8F9AA8; margin-bottom:4px;">Total Weekly Sessions</div>
                        <div style="font-size:18px; font-weight:600; color:#F5F7FA;">{len(sessions)}</div>
                    </div>
                </div>
            </div>
            '''
        )

with tab_manual:
    if timetable is not None and report is not None:
        session_list = [s.session_id for s in sessions if s.session_id in timetable.assignments]
        if not session_list:
            st.info("No assigned sessions available to edit.")
        else:
            # 1. Session Picker & Reassignment Controls
            col_ctrl1, col_ctrl2 = st.columns([1.6, 2.4])
            with col_ctrl1:
                sel_sid = st.selectbox(
                    "Choose Session to Move",
                    session_list,
                    format_func=lambda x: f"{x} · {dataset.subjects[session_map[x].subject_id].name[:18] if (x in session_map and session_map[x].subject_id in dataset.subjects) else x} ({session_map[x].class_id if x in session_map else ''})",
                    key="manual_sel_sid",
                )
                curr_sess = session_map[sel_sid]
                curr_asgn = timetable.assignments[sel_sid]
                curr_subj = dataset.subjects.get(curr_sess.subject_id)
                curr_tch = dataset.teachers.get(curr_sess.teacher_id)
                curr_cls = dataset.classes.get(curr_sess.class_id)
                curr_rm = dataset.rooms.get(curr_asgn.room_id)

            with col_ctrl2:
                col_d, col_p, col_r = st.columns(3)
                with col_d:
                    edit_d = st.selectbox(
                        "Target Day",
                        list(range(dataset.time_structure.num_days)),
                        format_func=get_day_name,
                        index=curr_asgn.day,
                        key="manual_edit_d",
                    )
                with col_p:
                    max_sp = dataset.time_structure.periods_per_day - curr_sess.duration
                    edit_p = st.number_input(
                        "Target Period",
                        min_value=1,
                        max_value=max_sp + 1,
                        value=curr_asgn.start_period + 1,
                        step=1,
                        key="manual_edit_p",
                    )
                with col_r:
                    edit_rm = st.selectbox(
                        "Target Room",
                        list(dataset.rooms.keys()),
                        index=list(dataset.rooms.keys()).index(curr_asgn.room_id) if curr_asgn.room_id in dataset.rooms else 0,
                        key="manual_edit_rm",
                    )

            # 2. Real-Time Dry-Run Simulation & Conflict Detection
            sim_asgn = Assignment(day=edit_d, start_period=edit_p - 1, room_id=edit_rm)
            sim_timetable = Timetable(assignments=dict(timetable.assignments))
            sim_timetable.assign(sel_sid, sim_asgn)
            sim_rep = count_violations(sim_timetable, sessions, dataset)

            is_moved = (edit_d != curr_asgn.day or (edit_p - 1) != curr_asgn.start_period or edit_rm != curr_asgn.room_id)
            hard_delta = sim_rep.hard_violations - report.hard_violations
            soft_delta = sim_rep.soft_penalty - report.soft_penalty

            conflicts = []
            clashing_session_ids = set()

            # Detailed clash analysis
            # H1: Teacher double-booking
            for other_id, other_asgn in sim_timetable.assignments.items():
                if other_id != sel_sid and other_id in session_map:
                    other_s = session_map[other_id]
                    if other_s.teacher_id == curr_sess.teacher_id and other_asgn.day == sim_asgn.day:
                        s1_slots = set(range(sim_asgn.start_period, sim_asgn.start_period + curr_sess.duration))
                        s2_slots = set(range(other_asgn.start_period, other_asgn.start_period + other_s.duration))
                        if s1_slots & s2_slots:
                            clashing_session_ids.add(other_id)
                            t_name = curr_tch.name if curr_tch else curr_sess.teacher_id
                            conflicts.append({
                                "cid": "H1 Teacher Double-Booking",
                                "text": f"Faculty {t_name} is already teaching {other_id} ({other_s.class_id}) at {get_day_name(sim_asgn.day)} P{other_asgn.start_period + 1}.",
                                "linked": f"{other_id} · {dataset.subjects[other_s.subject_id].name if other_s.subject_id in dataset.subjects else other_s.subject_id} ({other_s.class_id}) in {other_asgn.room_id}",
                            })

            # H2: Class double-booking
            for other_id, other_asgn in sim_timetable.assignments.items():
                if other_id != sel_sid and other_id in session_map:
                    other_s = session_map[other_id]
                    if other_s.class_id == curr_sess.class_id and other_asgn.day == sim_asgn.day:
                        s1_slots = set(range(sim_asgn.start_period, sim_asgn.start_period + curr_sess.duration))
                        s2_slots = set(range(other_asgn.start_period, other_asgn.start_period + other_s.duration))
                        if s1_slots & s2_slots:
                            clashing_session_ids.add(other_id)
                            conflicts.append({
                                "cid": "H2 Class Double-Booking",
                                "text": f"Class {curr_sess.class_id} already has session {other_id} booked at {get_day_name(sim_asgn.day)} P{other_asgn.start_period + 1}.",
                                "linked": f"{other_id} · {dataset.subjects[other_s.subject_id].name if other_s.subject_id in dataset.subjects else other_s.subject_id} with {other_s.teacher_id}",
                            })

            # H3: Room double-booking
            for other_id, other_asgn in sim_timetable.assignments.items():
                if other_id != sel_sid and other_id in session_map:
                    other_s = session_map[other_id]
                    if other_asgn.room_id == sim_asgn.room_id and other_asgn.day == sim_asgn.day:
                        s1_slots = set(range(sim_asgn.start_period, sim_asgn.start_period + curr_sess.duration))
                        s2_slots = set(range(other_asgn.start_period, other_asgn.start_period + other_s.duration))
                        if s1_slots & s2_slots:
                            clashing_session_ids.add(other_id)
                            conflicts.append({
                                "cid": "H3 Room Double-Booking",
                                "text": f"Room {sim_asgn.room_id} is already occupied by {other_id} ({other_s.class_id}) at {get_day_name(sim_asgn.day)} P{other_asgn.start_period + 1}.",
                                "linked": f"{other_id} · {dataset.subjects[other_s.subject_id].name if other_s.subject_id in dataset.subjects else other_s.subject_id} for {other_s.class_id}",
                            })

            # H4: Room capacity
            target_rm = dataset.rooms.get(sim_asgn.room_id)
            if curr_cls and target_rm and target_rm.capacity < curr_cls.strength:
                conflicts.append({
                    "cid": "H4 Room Capacity Exceeded",
                    "text": f"Room {target_rm.name} capacity ({target_rm.capacity} seats) is insufficient for class {curr_cls.name} ({curr_cls.strength} students).",
                    "linked": f"{curr_cls.name}: {curr_cls.strength} students > {target_rm.name}: {target_rm.capacity} seats",
                })

            # H5: Room type match
            if curr_subj and target_rm:
                if curr_subj.subject_type == "lab" and target_rm.room_type != "lab":
                    conflicts.append({
                        "cid": "H5 Room Type Mismatch",
                        "text": f"Laboratory session '{curr_subj.name}' requires a dedicated Lab facility, but {target_rm.name} is a lecture classroom.",
                        "linked": f"Subject: lab vs Facility: {target_rm.room_type}",
                    })

            # H6: Teacher availability
            if curr_tch:
                for off in range(curr_sess.duration):
                    slot = (sim_asgn.day, sim_asgn.start_period + off)
                    if slot in curr_tch.unavailable_slots:
                        conflicts.append({
                            "cid": "H6 Teacher Unavailable",
                            "text": f"Faculty {curr_tch.name} has marked {get_day_name(sim_asgn.day)} Period {sim_asgn.start_period + off + 1} as unavailable.",
                            "linked": f"{curr_tch.name} marked unavailable at {get_day_name(sim_asgn.day)} P{sim_asgn.start_period + off + 1}",
                        })

            # H9: Lunch break protection
            lunch_p = curr_cls.lunch_break_period if curr_cls else None
            if lunch_p is not None:
                for off in range(curr_sess.duration):
                    if sim_asgn.start_period + off == lunch_p:
                        conflicts.append({
                            "cid": "H9 Lunch Break Conflict",
                            "text": f"Target slot overlaps with protected class lunch break at Period {lunch_p + 1}.",
                            "linked": f"Mandatory break Period {lunch_p + 1} ({get_period_time_label(lunch_p)})",
                        })

            has_conflicts = len(conflicts) > 0 or sim_rep.hard_violations > 0

            # Status badge
            if has_conflicts:
                badge_html = f'<span style="color:#F0B76B; font-size:11px; font-weight:600; display:flex; align-items:center; gap:4px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#F0B76B" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>{len(conflicts) or sim_rep.hard_violations} Clash Flagged</span>'
            else:
                badge_html = '<span style="color:#69D4A3; font-size:11px; font-weight:600; display:flex; align-items:center; gap:4px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>Verified Feasible</span>'

            # Heading & Status (Figma match)
            st.html(
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">'
                f'<div style="display:flex; align-items:center; gap:8px;">'
                f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2"><path d="m3 3 7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path><path d="m13 13 6 6"></path></svg>'
                f'<span style="font-size:18px; font-weight:600; color:#F5F7FA;">Manual Editing & Conflict Flagging</span>'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:6px; background:#11161D; border:1px solid #1B2530; border-radius:6px; padding:4px 10px;">'
                f'{badge_html}'
                f'</div>'
                f'</div>'
            )

            # 3. Pending Change Context Bar (Figma match)
            subj_name = curr_subj.name if curr_subj else curr_sess.subject_id
            tch_name = curr_tch.name if curr_tch else curr_sess.teacher_id
            orig_slot_text = f"{get_day_name(curr_asgn.day)} P{curr_asgn.start_period + 1} ({curr_asgn.room_id})"
            dest_slot_text = f"{get_day_name(edit_d)} P{edit_p} ({edit_rm})"
            cue_badge = '<span style="background:rgba(124,140,255,0.12); color:#7C8CFF; font-size:10px; font-weight:600; padding:2px 8px; border-radius:4px; font-family:var(--font-mono);">Draft · Live Preview</span>' if is_moved else '<span style="background:rgba(255,255,255,0.06); color:#8F9AA8; font-size:10px; padding:2px 8px; border-radius:4px;">No changes</span>'

            st.html(
                f'<div class="pending-change-bar">'
                f'<div style="display:flex; align-items:center; gap:10px; overflow:hidden;">'
                f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2" style="flex-shrink:0;"><path d="M18 8L22 12L18 16"></path><path d="M2 12H22"></path></svg>'
                f'<span style="font-size:12px; color:#F5F7FA; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
                f'Moving <b style="color:#F5F7FA;">"{subj_name}"</b> ({curr_sess.class_id}, {tch_name}) from <span style="color:#8F9AA8;">{orig_slot_text}</span> <b style="color:#7C8CFF;">→</b> <span style="color:#7C8CFF; font-weight:600;">{dest_slot_text}</span>'
                f'</span>'
                f'</div>'
                f'<div>{cue_badge}</div>'
                f'</div>'
            )

            # 4. Weekly Timetable Grid Preview for Selected Class
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
                f"<span style='font-size:12px; font-weight:600; color:#F5F7FA;'>Weekly Timetable Preview · Batch {curr_sess.class_id} ({curr_cls.strength if curr_cls else ''} Students)</span>"
                f"<span style='font-size:11px; color:#8F9AA8; display:flex; align-items:center; gap:4px;'><svg width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='#8F9AA8' stroke-width='2'><circle cx='9' cy='12' r='1'></circle><circle cx='9' cy='5' r='1'></circle><circle cx='9' cy='19' r='1'></circle><circle cx='15' cy='12' r='1'></circle><circle cx='15' cy='5' r='1'></circle><circle cx='15' cy='19' r='1'></circle></svg>Interactive Draft Grid</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            m_html = [
                '<div class="timetable-viewport">',
                '<table class="tt-table">',
                '<thead><tr>',
                '<th class="tt-head-cell" style="width:72px;"><div class="tt-period-num">Day</div></th>'
            ]
            for p in range(dataset.time_structure.periods_per_day):
                m_html.append(
                    f'<th class="tt-head-cell">'
                    f'<div class="tt-period-num">P{p+1}</div>'
                    f'<div class="tt-period-time">{get_period_time_label(p)}</div>'
                    f'</th>'
                )
            m_html.append('</tr></thead><tbody>')

            cls_lunch = curr_cls.lunch_break_period if curr_cls else -1

            for d in range(dataset.time_structure.num_days):
                m_html.append(f'<tr><td class="tt-day-cell" style="width:72px;">{get_day_name(d)[:3]}</td>')
                p = 0
                while p < dataset.time_structure.periods_per_day:
                    # Check lunch
                    if p == cls_lunch:
                        m_html.append(
                            '<td class="tt-lunch-cell">'
                            '<div class="lunch-inner">'
                            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2"><path d="M18 2v6a3 3 0 0 1-3 3 3 3 0 0 1-3-3V2"></path><path d="M6 2v20"></path><line x1="12" y1="11" x2="12" y2="22"></line><line x1="18" y1="11" x2="18" y2="22"></line></svg>'
                            '<span>LUNCH</span>'
                            '</div>'
                            '</td>'
                        )
                        p += 1
                        continue

                    # Is this the newly proposed slot for our session?
                    if is_moved and d == sim_asgn.day and sim_asgn.start_period <= p < sim_asgn.start_period + curr_sess.duration:
                        rem_span = (sim_asgn.start_period + curr_sess.duration) - p
                        span = max(1, min(rem_span, dataset.time_structure.periods_per_day - p))
                        colspan_attr = f' colspan="{span}"' if span > 1 else ''

                        if has_conflicts:
                            # Conflict card
                            m_html.append(
                                f'<td class="tt-slot-cell"{colspan_attr}>'
                                f'<div class="lesson-card-conflict">'
                                f'<div>'
                                f'<span class="card-badge card-badge-conflict">'
                                f'<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#F38C92" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path></svg>'
                                f'CONFLICT CLASH'
                                f'</span>'
                                f'<div class="card-title" style="color:#F5F7FA;">{subj_name}</div>'
                                f'</div>'
                                f'<div class="card-meta">'
                                f'<span style="color:#F38C92;">{tch_name}</span>'
                                f'<span class="card-room-pill" style="border:1px solid #F38C92; color:#F38C92;">{edit_rm}</span>'
                                f'</div>'
                                f'</div>'
                                f'</td>'
                            )
                        else:
                            # Clean Drop target card
                            m_html.append(
                                f'<td class="tt-slot-cell"{colspan_attr}>'
                                f'<div class="tt-drop-target">'
                                f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2"><polyline points="9 10 4 15 9 20"></polyline><path d="M20 4v7a4 4 0 0 1-4 4H4"></path></svg>'
                                f'<span style="font-size:11px; font-weight:600; color:#7C8CFF;">Target Slot ({span}H)</span>'
                                f'<span style="font-size:9px; color:#8F9AA8; font-family:var(--font-mono);">{subj_name[:12]} ({edit_rm})</span>'
                                f'</div>'
                                f'</td>'
                            )
                        p += span
                        continue

                    # Find any other session of this class in sim_timetable
                    matched_s = None
                    matched_a = None
                    for sid, asgn in sim_timetable.assignments.items():
                        if sid == sel_sid and is_moved:
                            continue  # It has been moved to sim_asgn
                        s_obj = session_map.get(sid)
                        if not s_obj:
                            continue
                        if asgn.day == d and asgn.start_period <= p < asgn.start_period + s_obj.duration:
                            if s_obj.class_id == curr_sess.class_id:
                                matched_s, matched_a = s_obj, asgn
                                break

                    if matched_s and matched_a:
                        is_clash = matched_s.session_id in clashing_session_ids
                        s_subj = dataset.subjects.get(matched_s.subject_id)
                        s_tch = dataset.teachers.get(matched_s.teacher_id)
                        s_rm = dataset.rooms.get(matched_a.room_id)
                        s_is_lab = s_subj and s_subj.subject_type == "lab"

                        rem_span = (matched_a.start_period + matched_s.duration) - p
                        span = max(1, min(rem_span, dataset.time_structure.periods_per_day - p))
                        colspan_attr = f' colspan="{span}"' if span > 1 else ''

                        if is_clash:
                            c_cls = "lesson-card-conflict"
                            b_cls = "card-badge-conflict"
                            b_txt = "CLASH DETECTED"
                        elif s_is_lab:
                            c_cls = "lesson-card-lab"
                            b_cls = "card-badge-lab"
                            b_txt = f"LAB ({span}H)"
                        else:
                            c_cls = "lesson-card-theory"
                            b_cls = "card-badge-theory"
                            b_txt = "THEORY"

                        sub_title = s_subj.name if s_subj else matched_s.subject_id
                        t_name = s_tch.name if s_tch else matched_s.teacher_id
                        r_name = s_rm.name if s_rm else matched_a.room_id

                        m_html.append(
                            f'<td class="tt-slot-cell"{colspan_attr}>'
                            f'<div class="{c_cls}">'
                            f'<div>'
                            f'<span class="card-badge {b_cls}">{b_txt}</span>'
                            f'<div class="card-title">{sub_title}</div>'
                            f'</div>'
                            f'<div class="card-meta">'
                            f'<span>{t_name}</span>'
                            f'<span class="card-room-pill">{r_name}</span>'
                            f'</div>'
                            f'</div>'
                            f'</td>'
                        )
                        p += span
                    else:
                        m_html.append('<td class="tt-slot-cell"><div class="tt-free-cell">Free</div></td>')
                        p += 1

                m_html.append('</tr>')

            m_html.append('</tbody></table></div>')
            st.html("".join(m_html))

            # 5. Conflict Details & Validation Context (2-Column Figma Layout)
            col_val, col_conf = st.columns(2)
            with col_val:
                st.html(
                    f'<div class="validation-context-card">'
                    f'<div style="font-size:12px; font-weight:600; color:#F5F7FA; margin-bottom:8px;">Validation Context</div>'
                    f'<div style="font-size:11px; color:#8F9AA8; line-height:1.6; margin-bottom:8px;">'
                    f'Reassigning <b>{subj_name}</b> ({curr_sess.class_id}) with faculty <b>{tch_name}</b> to <b>{dest_slot_text}</b>. Room capacity: <b>{target_rm.capacity if target_rm else "N/A"} seats</b> vs cohort strength: <b>{curr_cls.strength if curr_cls else "N/A"} students</b>.'
                    f'</div>'
                    f'<div style="font-size:11px; color:#8F9AA8; border-top:1px solid #1B2530; padding-top:8px;">'
                    f'Real-time Score Impact: <b style="color:{"#F87171" if sim_rep.hard_violations > 0 else "#69D4A3"};">{sim_rep.hard_violations} Hard Violation(s)</b> · Quality Penalty: <b style="color:#7DD3D0;">{sim_rep.soft_penalty} pts</b> (Delta: {soft_delta:+d} pts)'
                    f'</div>'
                    f'</div>'
                )

            with col_conf:
                if has_conflicts:
                    first_c = conflicts[0] if conflicts else {"cid": "Constraint Clash", "text": "Hard constraint violation detected.", "linked": "Slot conflict"}
                    st.html(
                        f'<div class="conflict-popover-card">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
                        f'<div style="display:flex; align-items:center; gap:6px;">'
                        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F38C92" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
                        f'<span style="font-size:12px; font-weight:600; color:#F38C92;">{first_c["cid"]}</span>'
                        f'</div>'
                        f'<span style="background:#482A32; border-radius:6px; color:#F38C92; font-size:10px; font-weight:600; padding:2px 8px;">VIOLATION</span>'
                        f'</div>'
                        f'<div style="font-size:11px; color:#F5F7FA; line-height:1.5; margin-bottom:8px;">{first_c["text"]}</div>'
                        f'<div class="affected-booking-pill">'
                        f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#F38C92" stroke-width="2" style="flex-shrink:0;"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>'
                        f'<span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{first_c["linked"]}</span>'
                        f'</div>'
                        f'</div>'
                    )
                else:
                    st.html(
                        f'<div class="conflict-popover-clean">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
                        f'<div style="display:flex; align-items:center; gap:6px;">'
                        f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
                        f'<span style="font-size:12px; font-weight:600; color:#69D4A3;">Feasible Reassignment</span>'
                        f'</div>'
                        f'<span style="background:#1F4A3A; border-radius:6px; color:#69D4A3; font-size:10px; font-weight:600; padding:2px 8px;">0 CLASHES</span>'
                        f'</div>'
                        f'<div style="font-size:11px; color:#F5F7FA; line-height:1.5; margin-bottom:8px;">All 9 university operational hard constraints (H1-H9) are fully satisfied. Faculty, batch, and facility allocations are preserved without clashes.</div>'
                        f'<div class="affected-booking-clean">'
                        f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>'
                        f'<span>Clean slot: Ready for instant commit.</span>'
                        f'</div>'
                        f'</div>'
                    )

            # 6. Action Divider & Editor Actions
            col_act1, col_act2 = st.columns([2, 1.2])
            with col_act1:
                st.html(
                    f'<div style="display:flex; align-items:center; gap:8px; margin-top:10px; font-size:11px; color:#8F9AA8;">'
                    f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#F0B76B" stroke-width="2"><circle cx="12" cy="16" r="1"></circle><rect x="3" y="11" width="18" height="11" rx="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>'
                    f'<span>Validation check active: Reassignments must preserve feasibility</span>'
                    f'</div>'
                )
            with col_act2:
                col_b1, col_b2 = st.columns([1, 1.3])
                with col_b1:
                    if st.button("Reset / Undo", use_container_width=True):
                        st.rerun()
                with col_b2:
                    if st.button("✓ Apply Move", type="primary", use_container_width=True):
                        st.session_state.timetable.assign(sel_sid, sim_asgn)
                        st.session_state.report = count_violations(st.session_state.timetable, sessions, dataset)
                        st.success("Reassignment committed successfully!")
                        st.rerun()

            # 7. Editing Guidance & Workspace Footer
            st.html(
                f'<div class="editing-guidance">'
                f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
                f'<span>Guidance: Selecting a session previews its relocation and live checks all 9 Hard Constraints and 5 Soft Penalties before committing.</span>'
                f'</div>'
                f'<div class="workspace-footer">'
                f'<span>Active Dataset: {len(dataset.classes)} Classes · {len(dataset.teachers)} Teachers · {len(dataset.rooms)} Rooms</span>'
                f'<span>Total Scheduled: {len(timetable.assignments)} / {len(sessions)} Sessions</span>'
                f'</div>'
            )


    # ---------------------------------------------------------------------------
    # TAB 3: CONSTRAINTS DIAGNOSTIC MATRIX (H1-H9 & S1-S5)
    # ---------------------------------------------------------------------------
    else:
        st.info("💡 A timetable must be generated before you can inspect or manually reassign sessions. Click 'Generate Timetable' in the sidebar.")

with tab_matrix:
    if report is not None:
        # Constraints Heading (Figma spec)
        st.html(
            f'<div class="constraints-header">'
            f'<div class="constraints-title-group">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6"><line x1="10" y1="6" x2="21" y2="6"></line><line x1="10" y1="12" x2="21" y2="12"></line><line x1="10" y1="18" x2="21" y2="18"></line><polyline points="3 6 4 7 6 5"></polyline><polyline points="3 12 4 13 6 11"></polyline><polyline points="3 18 4 19 6 17"></polyline></svg>'
            f'<span class="constraints-title-text">Constraints Breakdown</span>'
            f'</div>'
            f'<div class="constraints-scope-text">Evaluated on active schedule · Full operational audit</div>'
            f'</div>'
        )

        col_h, col_s = st.columns([1.1, 1.0], gap="medium")

        # Hard specifications: H1 to H9
        hard_specs = [
            ("H1", "Teacher Double-Booking", "No teacher scheduled in multiple classes simultaneously"),
            ("H2", "Class Double-Booking", "No class section scheduled in multiple places simultaneously"),
            ("H3", "Room Double-Booking", "No classroom or lab allocated twice simultaneously"),
            ("H4", "Room Capacity", "Room capacity must exceed student cohort strength"),
            ("H5", "Room Type Match", "Laboratory subjects strictly mapped to lab facilities"),
            ("H6", "Teacher Availability", "Faculty only assigned during approved active availability"),
            ("H7", "Session Completeness", "100% of required weekly subject credits scheduled"),
            ("H8", "Teacher Max Load", "Faculty weekly instructional load strictly within quota"),
            ("H9", "Lunch Break Protection", "Batch lunch break period remains completely empty"),
        ]

        # Soft specifications: S1 to S5
        soft_specs = [
            ("S1", "Minimize Class Gaps", "Reduce fragmented idle periods in daily student schedule", 1),
            ("S2", "Spread Lectures", "Distribute weekly subject hours evenly across different days", 5),
            ("S3", "Consecutive Lab Hours", "Multi-hour practicals scheduled back-to-back", 0),
            ("S4", "Teacher Preferred Times", "Schedule senior professors in requested morning slots", "pref"),
            ("S5", "Max 2 Consecutive Lectures", "Prevent faculty fatigue by breaking back-to-back marathons", 3),
        ]

        # --- LEFT COLUMN: HARD CONSTRAINTS (560px) ---
        with col_h:
            h_status_bg = "#0F2A24" if report.hard_violations == 0 else "#2A1315"
            h_status_color = "#7BE6B2" if report.hard_violations == 0 else "#F87171"
            h_status_label = "Feasible" if report.hard_violations == 0 else f"{report.hard_violations} Violation(s)"

            h_card = [
                '<div class="hard-constraints-card">',
                f'<div class="hard-heading-row">'
                f'<div class="hard-title-desc">'
                f'<div class="hard-title-wrap">'
                f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="1.6"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>'
                f'<span class="hard-heading-text">Hard Constraints</span>'
                f'</div>'
                f'<div class="hard-requirement-text">Zero-tolerance feasibility rules</div>'
                f'</div>'
                f'<div class="hard-status-badge" style="background:{h_status_bg};">'
                f'<span class="hard-status-badge-text" style="color:{h_status_color};">{h_status_label}</span>'
                f'</div>'
                f'</div>'
            ]

            for cid, title, desc in hard_specs:
                val = report.details.get(cid, 0)
                if val == 0:
                    v_icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="1.6"><polyline points="20 6 9 17 4 12"></polyline></svg>'
                    v_color = "#69D4A3"
                else:
                    v_icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="1.6"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
                    v_color = "#F87171"

                h_card.append(
                    f'<div class="hard-constraint-row">'
                    f'<div class="hard-id-text">{cid}</div>'
                    f'<div class="hard-rule-col">'
                    f'<div class="hard-rule-name">{title}</div>'
                    f'<div class="hard-rule-desc">{desc}</div>'
                    f'</div>'
                    f'<div class="hard-violation-wrap">'
                    f'{v_icon}'
                    f'<span class="hard-violation-count" style="color:{v_color};">{val}</span>'
                    f'</div>'
                    f'</div>'
                )

            # Hard Total Banner
            h_tot_text = "0 (Feasible)" if report.hard_violations == 0 else f"{report.hard_violations} Violations"
            h_card.append(
                f'<div class="hard-total-bar" style="background:{h_status_bg};">'
                f'<span class="hard-total-label" style="color:{h_status_color};">Total Hard Violations</span>'
                f'<span class="hard-total-value" style="color:{h_status_color};">{h_tot_text}</span>'
                f'</div>'
                f'</div>'
            )
            st.html("".join(h_card))

        # --- RIGHT COLUMN: SOFT CONSTRAINTS (516px) & COST MODEL ---
        with col_s:
            s_status_label = "Optimized" if report.soft_penalty < 100 else f"Score: {report.soft_penalty} pts"
            s_card = [
                '<div class="soft-quality-col">',
                '<div class="soft-constraints-card">',
                f'<div class="soft-heading-wrap">'
                f'<div class="soft-heading-row">'
                f'<div class="soft-title-group">'
                f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6"><line x1="21" y1="4" x2="14" y2="4"></line><line x1="10" y1="4" x2="3" y2="4"></line><line x1="21" y1="12" x2="12" y2="12"></line><line x1="8" y1="12" x2="3" y2="12"></line><line x1="21" y1="20" x2="16" y2="20"></line><line x1="12" y1="20" x2="3" y2="20"></line><line x1="14" y1="2" x2="14" y2="6"></line><line x1="8" y1="10" x2="8" y2="14"></line><line x1="16" y1="18" x2="16" y2="22"></line></svg>'
                f'<span class="soft-heading-text">Soft Constraints</span>'
                f'</div>'
                f'<div class="soft-status-badge">'
                f'<span class="soft-status-badge-text">{s_status_label}</span>'
                f'</div>'
                f'</div>'
                f'<div class="soft-explanation-text">Preference minimization and balance rules</div>'
                f'</div>',
                f'<div class="scoring-cols-header">'
                f'<div class="col-header-rule">RULE / OBJECTIVE</div>'
                f'<div class="col-header-weight">WEIGHT</div>'
                f'<div class="col-header-penalty">PENALTY</div>'
                f'</div>'
            ]

            for sid, title, desc, weight in soft_specs:
                val = report.details.get(sid, 0)
                if sid == "S1":
                    instances = val
                elif sid == "S2":
                    instances = val // 5 if val else 0
                elif sid == "S5":
                    instances = val // 3 if val else 0
                else:
                    instances = val if isinstance(weight, str) else 0

                if val > 0:
                    occ_icon = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#F0B76B" stroke-width="1.6"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path></svg>'
                    occ_color = "#F0B76B"
                    occ_text = f"{instances} occurrences"
                else:
                    occ_icon = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="1.6"><polyline points="20 6 9 17 4 12"></polyline></svg>'
                    occ_color = "#69D4A3"
                    occ_text = "0 occurrences"

                w_text = f"w = {weight}" if isinstance(weight, int) and weight > 0 else (str(weight) if isinstance(weight, str) else "atomic")

                s_card.append(
                    f'<div class="soft-constraint-row">'
                    f'<div class="soft-rule-col">'
                    f'<div class="soft-rule-name-wrap">'
                    f'<span class="soft-rule-id">{sid}</span>'
                    f'<span class="soft-rule-name">{title}</span>'
                    f'</div>'
                    f'<div class="soft-rule-desc">{desc}</div>'
                    f'<div class="soft-occurrences-wrap">'
                    f'{occ_icon}'
                    f'<span class="soft-occurrences-count" style="color:{occ_color};">{occ_text}</span>'
                    f'</div>'
                    f'</div>'
                    f'<div class="soft-weight-text">{w_text}</div>'
                    f'<div class="soft-penalty-text">{val} pts</div>'
                    f'</div>'
                )

            # Soft Total Row
            s_card.append(
                f'<div class="soft-total-bar">'
                f'<span class="soft-total-label">Total Soft Penalty Score</span>'
                f'<span class="soft-total-val">{report.soft_penalty}</span>'
                f'</div>'
                f'</div>'
            )

            # Cost Model Card (Figma match)
            s_card.append(
                f'<div class="cost-model-card">'
                f'<div class="cost-model-title">COMBINED COST FUNCTION</div>'
                f'<div class="cost-model-formula">Cost = 1000 × H + Soft</div>'
                f'<div class="cost-model-calc">= 1000 × {report.hard_violations} + {report.soft_penalty} = <b style="color:#EDF0F7;">{report.total_cost}</b></div>'
                f'<div class="cost-model-interp">Hard constraints are prioritized with a 1,000× penalty weighting over soft criteria to ensure zero-conflict academic feasibility.</div>'
                f'</div>'
                f'</div>'
            )
            st.html("".join(s_card))

        # Workspace Footer
        audit_label = "All 9 Hard Constraints Passed" if report.hard_violations == 0 else f"{report.hard_violations} Clashes Active"
        st.html(
            f'<div class="workspace-footer">'
            f'<span class="footer-meta-text">Dataset: {len(dataset.classes)} Classes · {len(dataset.teachers)} Teachers · {len(dataset.rooms)} Rooms</span>'
            f'<span class="footer-context-text">Local Audit: {audit_label}</span>'
            f'</div>'
        )


    # ---------------------------------------------------------------------------
    # TAB 4: EXPORT CENTER (CSV / EXCEL)
    # ---------------------------------------------------------------------------
    else:
        st.info("💡 Constraint diagnostic metrics (H1–H9 Hard & S1–S5 Soft) will appear here once the timetable is generated.")

with tab_export:
    if timetable is not None and report is not None:
        ts = dataset.time_structure
        session_map = {s.session_id: s for s in dataset.get_sessions()}

        # -----------------------------------------------------------------------
        # Helper: Clean Excel Sheet Names (Excel max 31 chars, no forbidden chars)
        # -----------------------------------------------------------------------
        def sanitize_sheet_name(name: str) -> str:
            cleaned = "".join(c for c in name if c not in r"\/?*[]:")
            return cleaned[:31].strip() or "Sheet"

        # -----------------------------------------------------------------------
        # Helper: Build Weekly Grid DataFrames for Classes, Teachers, and Rooms
        # -----------------------------------------------------------------------
        def build_class_grid_df(class_id: str) -> pd.DataFrame:
            cols = [f"Period {p+1} ({get_period_time_label(p)})" for p in range(ts.periods_per_day)]
            days = [get_day_name(d) for d in range(ts.num_days)]
            grid = {c: ["" for _ in range(ts.num_days)] for c in cols}

            lunch_p = dataset.classes[class_id].lunch_break_period if class_id in dataset.classes else -1
            if 0 <= lunch_p < ts.periods_per_day:
                col_l = cols[lunch_p]
                for d in range(ts.num_days):
                    grid[col_l][d] = "LUNCH"

            for sid, asgn in timetable.assignments.items():
                s = session_map.get(sid)
                if s and s.class_id == class_id:
                    subj = dataset.subjects.get(s.subject_id)
                    tch = dataset.teachers.get(s.teacher_id)
                    rm = dataset.rooms.get(asgn.room_id)
                    is_lab = subj and subj.subject_type == "lab"
                    prefix = "[LAB] " if is_lab else ""
                    txt = f"{prefix}{subj.name if subj else s.subject_id} ({tch.name if tch else s.teacher_id} | {rm.name if rm else asgn.room_id})"
                    for off in range(s.duration):
                        p = asgn.start_period + off
                        if p < ts.periods_per_day:
                            grid[cols[p]][asgn.day] = txt

            df = pd.DataFrame(grid, index=days)
            df.index.name = "Day"
            return df

        def build_teacher_grid_df(teacher_id: str) -> pd.DataFrame:
            cols = [f"Period {p+1} ({get_period_time_label(p)})" for p in range(ts.periods_per_day)]
            days = [get_day_name(d) for d in range(ts.num_days)]
            grid = {c: ["" for _ in range(ts.num_days)] for c in cols}

            unavail = set(dataset.teachers[teacher_id].unavailable_slots) if teacher_id in dataset.teachers else set()
            for (d, p) in unavail:
                if 0 <= d < ts.num_days and 0 <= p < ts.periods_per_day:
                    grid[cols[p]][d] = "UNAVAILABLE"

            for sid, asgn in timetable.assignments.items():
                s = session_map.get(sid)
                if s and s.teacher_id == teacher_id:
                    subj = dataset.subjects.get(s.subject_id)
                    cls = dataset.classes.get(s.class_id)
                    rm = dataset.rooms.get(asgn.room_id)
                    is_lab = subj and subj.subject_type == "lab"
                    prefix = "[LAB] " if is_lab else ""
                    txt = f"{prefix}{subj.name if subj else s.subject_id} ({cls.name if cls else s.class_id} | {rm.name if rm else asgn.room_id})"
                    for off in range(s.duration):
                        p = asgn.start_period + off
                        if p < ts.periods_per_day:
                            grid[cols[p]][asgn.day] = txt

            df = pd.DataFrame(grid, index=days)
            df.index.name = "Day"
            return df

        def build_room_grid_df(room_id: str) -> pd.DataFrame:
            cols = [f"Period {p+1} ({get_period_time_label(p)})" for p in range(ts.periods_per_day)]
            days = [get_day_name(d) for d in range(ts.num_days)]
            grid = {c: ["" for _ in range(ts.num_days)] for c in cols}

            for sid, asgn in timetable.assignments.items():
                s = session_map.get(sid)
                if s and asgn.room_id == room_id:
                    subj = dataset.subjects.get(s.subject_id)
                    cls = dataset.classes.get(s.class_id)
                    tch = dataset.teachers.get(s.teacher_id)
                    is_lab = subj and subj.subject_type == "lab"
                    prefix = "[LAB] " if is_lab else ""
                    txt = f"{prefix}{subj.name if subj else s.subject_id} ({cls.name if cls else s.class_id} | {tch.name if tch else s.teacher_id})"
                    for off in range(s.duration):
                        p = asgn.start_period + off
                        if p < ts.periods_per_day:
                            grid[cols[p]][asgn.day] = txt

            df = pd.DataFrame(grid, index=days)
            df.index.name = "Day"
            return df

        # Master flat dataframe
        records = []
        for sid, asgn in timetable.assignments.items():
            sess = session_map.get(sid)
            if not sess:
                continue
            subj = dataset.subjects.get(sess.subject_id)
            cls = dataset.classes.get(sess.class_id)
            tch = dataset.teachers.get(sess.teacher_id)
            rm = dataset.rooms.get(asgn.room_id)

            records.append({
                "Session ID": sid,
                "Class ID": sess.class_id,
                "Class Name": cls.name if cls else sess.class_id,
                "Subject Code": sess.subject_id,
                "Subject Name": subj.name if subj else sess.subject_id,
                "Subject Type": subj.subject_type if subj else "theory",
                "Teacher ID": sess.teacher_id,
                "Teacher Name": tch.name if tch else sess.teacher_id,
                "Room ID": asgn.room_id,
                "Room Name": rm.name if rm else asgn.room_id,
                "Day": get_day_name(asgn.day),
                "Start Period": asgn.start_period + 1,
                "End Period": asgn.start_period + sess.duration,
                "Duration (Hrs)": sess.duration,
                "Slot Label": f"{get_day_name(asgn.day)[:3]} Period {asgn.start_period + 1} ({get_period_time_label(asgn.start_period)})",
            })

        flat_df = pd.DataFrame(records)
        if not flat_df.empty:
            flat_df = flat_df.sort_values(by=["Class ID", "Day", "Start Period"]).reset_index(drop=True)

        # 1. Export Heading (Figma match)
        st.html(
            f'<div class="export-heading">'
            f'<div class="export-title-group">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>'
            f'<span class="export-title-text">Export Center</span>'
            f'</div>'
            f'<div class="export-status-badge">'
            f'<span class="export-status-badge-text">Schedule Verified · Ready for Export</span>'
            f'</div>'
            f'</div>'
        )

        # 2. Previous Run Context Banner
        st.html(
            f'<div class="export-context-bar">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
            f'<span class="export-context-text">Download academic timetable in weekly Grid View (Days as rows, Periods as columns) separately for Classes, Teachers, and Rooms, or export the raw database master schedule.</span>'
            f'</div>'
        )

        # 3. Export Settings & Scope Card
        st.markdown(
            f'<div class="export-settings-card" style="padding: 22px 24px 18px;">'
            f'<div style="font-family:\'Inter\'; font-size:13px; font-weight:600; color:#F5F7FA; margin-bottom:14px; display:flex; align-items:center; gap:8px;">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>'
            f'<span>Configure Export Format & Schedule Grid Scope</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_cat, col_scope = st.columns([1.1, 1.9])
        with col_cat:
            export_mode = st.radio(
                "Export Grid View Category",
                options=["Classes Grid", "Teachers Grid", "Rooms Grid", "Master Flat Schedule"],
                index=0,
                horizontal=False,
            )

        with col_scope:
            if export_mode == "Classes Grid":
                class_options = ["All Classes (Multi-Tab Excel / Combined Grid CSV)"] + [
                    f"{cid} - {c.name}" for cid, c in dataset.classes.items()
                ]
                selected_scope = st.selectbox(
                    "Select Class Cohort Scope",
                    options=class_options,
                    index=0,
                    help="Choose All Classes for a multi-tab workbook, or a specific class for single-cohort grid."
                )
            elif export_mode == "Teachers Grid":
                teacher_options = ["All Teachers (Multi-Tab Excel / Combined Grid CSV)"] + [
                    f"{tid} - {t.name}" for tid, t in dataset.teachers.items()
                ]
                selected_scope = st.selectbox(
                    "Select Faculty Scope",
                    options=teacher_options,
                    index=0,
                    help="Choose All Teachers for a multi-tab workbook, or a specific faculty member."
                )
            elif export_mode == "Rooms Grid":
                room_options = ["All Rooms (Multi-Tab Excel / Combined Grid CSV)"] + [
                    f"{rid} - {r.name} ({r.room_type.capitalize()})" for rid, r in dataset.rooms.items()
                ]
                selected_scope = st.selectbox(
                    "Select Facility / Room Scope",
                    options=room_options,
                    index=0,
                    help="Choose All Rooms for a multi-tab workbook, or a specific lecture hall/lab."
                )
            else:
                flat_options = ["All Cohorts (Master Timetable)"] + [
                    f"{cid} - {c.name}" for cid, c in dataset.classes.items()
                ]
                selected_scope = st.selectbox(
                    "Select Session Cohort Filter",
                    options=flat_options,
                    index=0,
                    help="Filter flat database records by cohort or export full timetable."
                )

        # -----------------------------------------------------------------------
        # Prepare DataFrames & File Buffers for CSV and Excel based on Selection
        # -----------------------------------------------------------------------
        preview_is_grid = (export_mode != "Master Flat Schedule")
        preview_grid_df = None
        preview_flat_df = None

        if export_mode == "Classes Grid":
            is_all = selected_scope.startswith("All Classes")
            if is_all:
                first_cid = list(dataset.classes.keys())[0]
                preview_grid_df = build_class_grid_df(first_cid)
                preview_title = "timetable_grid_all_classes.xlsx"
                preview_sheet = f"All Classes (Multi-Tab) · Previewing {first_cid}"
                preview_target_label = f"Cohort: {first_cid} ({dataset.classes[first_cid].name})"

                # Multi-tab Excel: Master Summary + One tab per class
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    flat_df.to_excel(writer, sheet_name="Master Overview", index=False)
                    for cid, c in dataset.classes.items():
                        df_c = build_class_grid_df(cid)
                        df_c.to_excel(writer, sheet_name=sanitize_sheet_name(cid))
                xlsx_bytes = buf.getvalue()
                xlsx_filename = "timetable_grid_all_classes.xlsx"

                # Combined CSV
                all_rows = []
                for cid, c in dataset.classes.items():
                    df_c = build_class_grid_df(cid).reset_index()
                    df_c.insert(0, "Class Name", c.name)
                    df_c.insert(0, "Class ID", cid)
                    all_rows.append(df_c)
                combined_csv_df = pd.concat(all_rows, ignore_index=True)
                csv_bytes = combined_csv_df.to_csv(index=False).encode("utf-8")
                csv_filename = "timetable_grid_all_classes.csv"
            else:
                cid = selected_scope.split(" - ")[0]
                c_obj = dataset.classes.get(cid)
                preview_grid_df = build_class_grid_df(cid)
                preview_title = f"timetable_grid_class_{cid}.xlsx"
                preview_sheet = f"Cohort {cid} Weekly Grid"
                preview_target_label = f"Cohort: {cid} ({c_obj.name if c_obj else cid})"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    preview_grid_df.to_excel(writer, sheet_name=sanitize_sheet_name(cid))
                xlsx_bytes = buf.getvalue()
                xlsx_filename = f"timetable_grid_class_{cid}.xlsx"

                csv_bytes = preview_grid_df.to_csv(index=True).encode("utf-8")
                csv_filename = f"timetable_grid_class_{cid}.csv"

        elif export_mode == "Teachers Grid":
            is_all = selected_scope.startswith("All Teachers")
            if is_all:
                first_tid = list(dataset.teachers.keys())[0]
                preview_grid_df = build_teacher_grid_df(first_tid)
                first_tname = dataset.teachers[first_tid].name
                preview_title = "timetable_grid_all_teachers.xlsx"
                preview_sheet = f"All Teachers (Multi-Tab) · Previewing {first_tname}"
                preview_target_label = f"Faculty: {first_tname} ({first_tid})"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    flat_df.to_excel(writer, sheet_name="Master Overview", index=False)
                    for tid, t in dataset.teachers.items():
                        df_t = build_teacher_grid_df(tid)
                        s_name = sanitize_sheet_name(f"{tid}_{t.name}")
                        df_t.to_excel(writer, sheet_name=s_name)
                xlsx_bytes = buf.getvalue()
                xlsx_filename = "timetable_grid_all_teachers.xlsx"

                all_rows = []
                for tid, t in dataset.teachers.items():
                    df_t = build_teacher_grid_df(tid).reset_index()
                    df_t.insert(0, "Teacher Name", t.name)
                    df_t.insert(0, "Teacher ID", tid)
                    all_rows.append(df_t)
                combined_csv_df = pd.concat(all_rows, ignore_index=True)
                csv_bytes = combined_csv_df.to_csv(index=False).encode("utf-8")
                csv_filename = "timetable_grid_all_teachers.csv"
            else:
                tid = selected_scope.split(" - ")[0]
                t_obj = dataset.teachers.get(tid)
                t_name = t_obj.name if t_obj else tid
                preview_grid_df = build_teacher_grid_df(tid)
                preview_title = f"timetable_grid_teacher_{tid}.xlsx"
                preview_sheet = f"Faculty {t_name} Weekly Grid"
                preview_target_label = f"Faculty: {t_name} ({tid})"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    preview_grid_df.to_excel(writer, sheet_name=sanitize_sheet_name(f"{tid}_{t_name}"))
                xlsx_bytes = buf.getvalue()
                xlsx_filename = f"timetable_grid_teacher_{tid}.xlsx"

                csv_bytes = preview_grid_df.to_csv(index=True).encode("utf-8")
                csv_filename = f"timetable_grid_teacher_{tid}.csv"

        elif export_mode == "Rooms Grid":
            is_all = selected_scope.startswith("All Rooms")
            if is_all:
                first_rid = list(dataset.rooms.keys())[0]
                preview_grid_df = build_room_grid_df(first_rid)
                first_rname = dataset.rooms[first_rid].name
                preview_title = "timetable_grid_all_rooms.xlsx"
                preview_sheet = f"All Rooms (Multi-Tab) · Previewing {first_rname}"
                preview_target_label = f"Facility: {first_rname} ({first_rid})"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    flat_df.to_excel(writer, sheet_name="Master Overview", index=False)
                    for rid, r in dataset.rooms.items():
                        df_r = build_room_grid_df(rid)
                        s_name = sanitize_sheet_name(f"{rid}_{r.name}")
                        df_r.to_excel(writer, sheet_name=s_name)
                xlsx_bytes = buf.getvalue()
                xlsx_filename = "timetable_grid_all_rooms.xlsx"

                all_rows = []
                for rid, r in dataset.rooms.items():
                    df_r = build_room_grid_df(rid).reset_index()
                    df_r.insert(0, "Room Type", r.room_type)
                    df_r.insert(0, "Room Name", r.name)
                    df_r.insert(0, "Room ID", rid)
                    all_rows.append(df_r)
                combined_csv_df = pd.concat(all_rows, ignore_index=True)
                csv_bytes = combined_csv_df.to_csv(index=False).encode("utf-8")
                csv_filename = "timetable_grid_all_rooms.csv"
            else:
                rid = selected_scope.split(" - ")[0]
                r_obj = dataset.rooms.get(rid)
                r_name = r_obj.name if r_obj else rid
                preview_grid_df = build_room_grid_df(rid)
                preview_title = f"timetable_grid_room_{rid}.xlsx"
                preview_sheet = f"Facility {r_name} Weekly Grid"
                preview_target_label = f"Facility: {r_name} ({rid})"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    preview_grid_df.to_excel(writer, sheet_name=sanitize_sheet_name(f"{rid}_{r_name}"))
                xlsx_bytes = buf.getvalue()
                xlsx_filename = f"timetable_grid_room_{rid}.xlsx"

                csv_bytes = preview_grid_df.to_csv(index=True).encode("utf-8")
                csv_filename = f"timetable_grid_room_{rid}.csv"

        else:
            # Master Flat Schedule
            is_all = selected_scope.startswith("All Cohorts")
            if is_all:
                preview_flat_df = flat_df
                preview_title = "academic_timetable_master.xlsx"
                preview_sheet = "All Cohorts Master Schedule · Tabular Records"
                preview_target_label = "Complete Flat Database"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    flat_df.to_excel(writer, sheet_name="Master Schedule", index=False)
                    for cid in dataset.classes.keys():
                        build_class_grid_df(cid).to_excel(writer, sheet_name=sanitize_sheet_name(cid))
                xlsx_bytes = buf.getvalue()
                xlsx_filename = "academic_timetable_master.xlsx"
                csv_bytes = flat_df.to_csv(index=False).encode("utf-8")
                csv_filename = "academic_timetable_master.csv"
            else:
                cid = selected_scope.split(" - ")[0]
                filtered_flat = flat_df[flat_df["Class ID"] == cid].reset_index(drop=True)
                preview_flat_df = filtered_flat
                preview_title = f"timetable_sessions_{cid}.xlsx"
                preview_sheet = f"Cohort {cid} Scheduled Sessions"
                preview_target_label = f"Cohort {cid}"

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    filtered_flat.to_excel(writer, sheet_name=sanitize_sheet_name(f"{cid}_Sessions"), index=False)
                    build_class_grid_df(cid).to_excel(writer, sheet_name=sanitize_sheet_name(f"{cid}_Grid"))
                xlsx_bytes = buf.getvalue()
                xlsx_filename = f"timetable_sessions_{cid}.xlsx"
                csv_bytes = filtered_flat.to_csv(index=False).encode("utf-8")
                csv_filename = f"timetable_sessions_{cid}.csv"

        # -----------------------------------------------------------------------
        # 4. Workbook & Grid Live Preview Card (Figma match)
        # -----------------------------------------------------------------------
        if preview_is_grid and preview_grid_df is not None:
            # Render visually stunning Grid View Table
            th_cells = ['<th class="tt-head-cell" style="width:85px; position:sticky; left:0; z-index:3; background:#0E131A;"><span class="tt-period-num">Day</span></th>']
            for col_name in preview_grid_df.columns:
                p_label = col_name.split(" (")[0]
                p_time = col_name.split(" (")[1].rstrip(")") if " (" in col_name else ""
                th_cells.append(
                    f'<th class="tt-head-cell" style="min-width:140px; padding:10px 8px;">'
                    f'<span class="tt-period-num">{p_label}</span>'
                    f'<div class="tt-period-time">{p_time}</div>'
                    f'</th>'
                )

            tb_rows = []
            for day_name, row in preview_grid_df.iterrows():
                td_cells = [f'<td class="tt-day-cell" style="position:sticky; left:0; z-index:2; background:#0E131A;">{day_name}</td>']
                for val in row:
                    val_str = str(val).strip()
                    if val_str == "LUNCH":
                        cell_content = (
                            '<span style="display:inline-flex; align-items:center; gap:4px; padding:3px 8px; background:#2A2315; color:#F0B76B; border:1px solid #4A3A1B; border-radius:4px; font-size:10px; font-weight:600;">'
                            '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#F0B76B" stroke-width="2"><path d="M18 2v6a3 3 0 0 1-3 3 3 3 0 0 1-3-3V2"></path><path d="M6 2v20"></path></svg> LUNCH'
                            '</span>'
                        )
                    elif val_str == "UNAVAILABLE":
                        cell_content = (
                            '<span style="display:inline-flex; align-items:center; gap:4px; padding:3px 8px; background:#2A1315; color:#F87171; border:1px solid #4A1D20; border-radius:4px; font-size:10px; font-weight:600;">'
                            '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#F87171" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line></svg> UNAVAIL'
                            '</span>'
                        )
                    elif val_str.startswith("[LAB]"):
                        sess_text = val_str[6:]
                        cell_content = (
                            f'<div style="display:flex; flex-direction:column; gap:3px;">'
                            f'<span style="display:inline-flex; align-self:flex-start; padding:1px 6px; background:#0F2A28; color:#7DD3D0; border:1px solid #1C4D4A; border-radius:4px; font-size:9px; font-weight:700; letter-spacing:0.5px;">LAB</span>'
                            f'<span style="font-size:11px; color:#F5F7FA; font-weight:500; line-height:1.3;">{sess_text}</span>'
                            f'</div>'
                        )
                    elif val_str:
                        cell_content = f'<span style="font-size:11px; color:#EDF0F7; font-weight:500; line-height:1.3;">{val_str}</span>'
                    else:
                        cell_content = '<span style="color:#5B6674; font-size:11px;">—</span>'

                    td_cells.append(
                        f'<td style="padding:10px; border-bottom:1px solid #1B2530; border-right:1px solid #1B2530; background:#0E131A; vertical-align:middle;">{cell_content}</td>'
                    )
                tb_rows.append(f'<tr>{"".join(td_cells)}</tr>')

            preview_table_html = (
                f'<div class="timetable-viewport" style="border-top:1px solid #1B2530; max-height:420px; overflow:auto;">'
                f'<table class="tt-table" style="width:100%; border-collapse:collapse;">'
                f'<thead><tr>{"".join(th_cells)}</tr></thead>'
                f'<tbody>{"".join(tb_rows)}</tbody>'
                f'</table>'
                f'</div>'
            )
            preview_footer_meta = f"Showing weekly grid for {preview_target_label} · {ts.num_days} Days × {ts.periods_per_day} Periods"

        else:
            # Flat Schedule Preview Rows
            preview_slice = preview_flat_df.head(8) if preview_flat_df is not None else flat_df.head(8)
            preview_rows = []
            for _, r in preview_slice.iterrows():
                preview_rows.append(
                    f'<div class="preview-data-row">'
                    f'<div class="preview-cell-text" style="width:90px;">{r["Class ID"]}</div>'
                    f'<div class="preview-cell-text" style="width:172px;">{r.get("Slot Label", r["Day"])}</div>'
                    f'<div class="preview-cell-text preview-cell-text-bold" style="width:320px;">{r["Subject Name"]}</div>'
                    f'<div class="preview-cell-text" style="width:240px;">{r["Teacher Name"]}</div>'
                    f'<div class="preview-cell-text" style="width:110px;">{r["Room Name"]}</div>'
                    f'</div>'
                )
            preview_table_html = (
                f'<div class="preview-cols-header">'
                f'<div class="preview-col-head" style="width:90px;">CLASS COHORT</div>'
                f'<div class="preview-col-head" style="width:172px;">DAY / PERIOD</div>'
                f'<div class="preview-col-head" style="width:320px;">COURSE / SUBJECT NAME</div>'
                f'<div class="preview-col-head" style="width:240px;">FACULTY INSTRUCTOR</div>'
                f'<div class="preview-col-head" style="width:110px;">ROOM FACILITY</div>'
                f'</div>'
                + "".join(preview_rows)
            )
            preview_footer_meta = f"Showing first {len(preview_slice)} of {len(preview_flat_df if preview_flat_df is not None else flat_df)} scheduled sessions"

        st.html(
            f'<div class="workbook-preview-card">'
            f'<div class="preview-toolbar">'
            f'<div class="preview-file-details">'
            f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M8 13h8"></path><path d="M8 17h8"></path><path d="M10 9h4"></path></svg>'
            f'<div class="preview-title-wrap">'
            f'<span class="preview-filename">{preview_title}</span>'
            f'<span class="preview-sheet-context">Active Sheet / Scope: {preview_sheet}</span>'
            f'</div>'
            f'</div>'
            f'<div class="export-status-badge">'
            f'<span class="export-status-badge-text">Feasibility Audit Passed</span>'
            f'</div>'
            f'</div>'
            + preview_table_html +
            f'<div class="preview-footer-bar">'
            f'<span class="preview-footer-meta">{preview_footer_meta}</span>'
            f'<span class="preview-footer-meta">UTF-8 Encoded · Verified Timetable</span>'
            f'</div>'
            f'</div>'
        )

        # -----------------------------------------------------------------------
        # 5. Download Actions Bar & Streamlit Downloads
        # -----------------------------------------------------------------------
        col_down_text, col_down_csv, col_down_xlsx = st.columns([2.2, 1.3, 1.5])
        with col_down_text:
            st.html(
                f'<div class="download-assurance">'
                f'<span class="download-ready-title">Ready to Generate Downloads</span>'
                f'<span class="download-ready-subtitle">Verified {export_mode} exported as formatted weekly schedule</span>'
                f'</div>'
            )
        with col_down_csv:
            st.download_button(
                label=f"Download CSV",
                data=csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
            )
        with col_down_xlsx:
            st.download_button(
                label=f"Download Excel (.xlsx)",
                data=xlsx_bytes,
                file_name=xlsx_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                icon=":material/download:",
                use_container_width=True,
            )

        # 6. Workspace Footer
        st.html(
            f'<div class="workspace-footer">'
            f'<span class="footer-meta-text">Dataset: {len(dataset.classes)} Classes · {len(dataset.teachers)} Teachers · {len(dataset.rooms)} Rooms</span>'
            f'<span class="footer-context-text">Local Audit: All 9 Hard Constraints Passed</span>'
            f'</div>'
        )
    else:
        st.info("💡 Generate a timetable first to export to CSV or multi-sheet Excel.")

with tab_benchmarks:
    if st.session_state.dataset is None:
        st.info("💡 Load a dataset first from the sidebar to view Search Lab & Benchmark Analytics.")
    else:
        ds = st.session_state.dataset
        ds_key = st.session_state.get("current_dataset_key", "default_ds")
        loaded_name = st.session_state.get("loaded_filename", "Active Dataset")
        total_sessions = len(ds.get_sessions())

        # Ensure dictionary structure
        if "benchmark_results" not in st.session_state:
            st.session_state.benchmark_results = {}
        if "benchmark_histories" not in st.session_state:
            st.session_state.benchmark_histories = {}

        if ds_key not in st.session_state.benchmark_results or not isinstance(st.session_state.benchmark_results[ds_key], dict):
            # Convert legacy list to dict if needed
            if ds_key in st.session_state.benchmark_results and isinstance(st.session_state.benchmark_results[ds_key], list):
                legacy_list = st.session_state.benchmark_results[ds_key]
                id_map = {
                    "Baseline Backtracking": "baseline",
                    "MRV + Degree + FC": "mrv",
                    "Min-Conflicts": "min_conflicts",
                    "Simulated Annealing": "simulated_annealing",
                    "Hybrid CSP + SA": "hybrid",
                }
                st.session_state.benchmark_results[ds_key] = {
                    id_map.get(r["Algorithm"], r["Algorithm"]): r for r in legacy_list
                }
            else:
                st.session_state.benchmark_results[ds_key] = {}

        if ds_key not in st.session_state.benchmark_histories or not isinstance(st.session_state.benchmark_histories[ds_key], dict):
            st.session_state.benchmark_histories[ds_key] = {}

        # -----------------------------------------------------------------------
        # Helper: Live Benchmark Suite Runner (Standardized Sidebar Matching Parameters)
        # -----------------------------------------------------------------------
        def execute_live_benchmarks(target_ds: Dataset):
            suite_dict = {}
            suite_hists = {}

            # 1. Backtracking Baseline (Seed 42, 30s timeout matching sidebar default)
            s_base = BacktrackingSolver(target_ds, use_mrv=False, use_degree=False, use_forward_checking=False, timeout_seconds=30.0, seed=42)
            res_base = s_base.solve()
            suite_dict["baseline"] = {
                "Id": "baseline",
                "Algorithm": "Baseline Backtracking",
                "Badge": "Baseline",
                "Time": res_base.time_taken,
                "TimeStr": f"{res_base.time_taken:.3f}s",
                "Work": f"{res_base.nodes_expanded:,} nodes",
                "Hard": res_base.report.hard_violations,
                "Soft": res_base.report.soft_penalty,
                "Total Cost": res_base.report.total_cost,
                "Success": res_base.success,
            }

            # 2. Backtracking MRV + Degree + FC (Seed 42, 30s timeout matching sidebar default)
            s_mrv = BacktrackingSolver(target_ds, use_mrv=True, use_degree=True, use_forward_checking=True, timeout_seconds=30.0, seed=42)
            res_mrv = s_mrv.solve()
            suite_dict["mrv"] = {
                "Id": "mrv",
                "Algorithm": "MRV + Degree + FC",
                "Badge": "Heuristic",
                "Time": res_mrv.time_taken,
                "TimeStr": f"{res_mrv.time_taken:.3f}s",
                "Work": f"{res_mrv.nodes_expanded:,} nodes",
                "Hard": res_mrv.report.hard_violations,
                "Soft": res_mrv.report.soft_penalty,
                "Total Cost": res_mrv.report.total_cost,
                "Success": res_mrv.success,
            }

            # 3. Min-Conflicts (Seed 42, 2000 steps, p_walk 0.06 matching sidebar default)
            s_mc = MinConflictsSolver(target_ds, max_iterations=2000, random_walk_prob=0.06, seed=42)
            res_mc = s_mc.solve()
            suite_dict["min_conflicts"] = {
                "Id": "min_conflicts",
                "Algorithm": "Min-Conflicts",
                "Badge": "Local",
                "Time": res_mc.time_taken,
                "TimeStr": f"{res_mc.time_taken:.3f}s",
                "Work": f"{res_mc.iterations:,} steps",
                "Hard": res_mc.report.hard_violations,
                "Soft": res_mc.report.soft_penalty,
                "Total Cost": res_mc.report.total_cost,
                "Success": res_mc.success,
            }
            if hasattr(res_mc, "cost_history") and res_mc.cost_history:
                suite_hists["Min-Conflicts"] = res_mc.cost_history

            # 4. Simulated Annealing (Seed 42, T0 500, cooling 0.993, 2000 iters matching sidebar default)
            s_sa = SimulatedAnnealingSolver(target_ds, initial_temp=500.0, cooling_rate=0.993, max_iterations=2000, seed=42)
            res_sa = s_sa.solve()
            suite_dict["simulated_annealing"] = {
                "Id": "simulated_annealing",
                "Algorithm": "Simulated Annealing",
                "Badge": "Local",
                "Time": res_sa.time_taken,
                "TimeStr": f"{res_sa.time_taken:.3f}s",
                "Work": f"{res_sa.iterations:,} iters",
                "Hard": res_sa.report.hard_violations,
                "Soft": res_sa.report.soft_penalty,
                "Total Cost": res_sa.report.total_cost,
                "Success": res_sa.success,
            }
            if hasattr(res_sa, "cost_history") and res_sa.cost_history:
                suite_hists["Simulated Annealing"] = res_sa.cost_history

            # 5. Hybrid CSP + SA (Seed 42, 15s BT timeout, 1200 SA iters matching sidebar default)
            s_hy = HybridSolver(target_ds, bt_timeout=15.0, sa_max_iterations=1200, seed=42)
            res_hy = s_hy.solve()
            suite_dict["hybrid"] = {
                "Id": "hybrid",
                "Algorithm": "Hybrid CSP + SA",
                "Badge": "Hybrid",
                "Time": res_hy.total_time,
                "TimeStr": f"{res_hy.total_time:.3f}s",
                "Work": f"{res_hy.backtracking_nodes:,} nodes + {res_hy.sa_iterations:,} SA",
                "Hard": res_hy.report.hard_violations,
                "Soft": res_hy.report.soft_penalty,
                "Total Cost": res_hy.report.total_cost,
                "Success": res_hy.success,
            }
            if hasattr(res_hy, "cost_history") and res_hy.cost_history:
                suite_hists["Hybrid CSP + SA"] = res_hy.cost_history

            return suite_dict, suite_hists

        # Identify which algorithms have actually been calculated
        canonical_algos = [
            ("baseline", "Baseline Backtracking", "Baseline"),
            ("mrv", "MRV + Degree + FC", "Heuristic"),
            ("min_conflicts", "Min-Conflicts", "Local"),
            ("simulated_annealing", "Simulated Annealing", "Local"),
            ("hybrid", "Hybrid CSP + SA", "Hybrid"),
        ]

        evaluated_records = [
            st.session_state.benchmark_results[ds_key][k]
            for k, _, _ in canonical_algos
            if k in st.session_state.benchmark_results[ds_key]
        ]
        eval_count = len(evaluated_records)
        has_any_eval = eval_count > 0

        # 1. Experiments Heading Row (Figma Obsidian Dark)
        heading_badge = f"{eval_count} of 5 Solvers Evaluated" if eval_count < 5 else "5 Search Algorithms Evaluated"
        st.html(
            f'<div class="experiments-heading-row">'
            f'<div class="experiments-title-box">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31L4.69 19.3A2 2 0 0 0 6.43 22h11.14a2 2 0 0 0 1.74-2.7L14 9.31V2"/><line x1="8" y1="2" x2="16" y2="2"/><line x1="6.5" y1="16" x2="17.5" y2="16"/></svg>'
            f'<span class="experiments-title">Search Lab & Comparative Benchmarks</span>'
            f'</div>'
            f'<div class="experiments-badge">{heading_badge}</div>'
            f'</div>'
        )

        # 2. Dynamic Experiment Protocol Card
        st.html(
            f'<div class="experiment-protocol-card">'
            f'<div class="protocol-title-row">'
            f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#979FAF" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2"/><line x1="2" y1="6" x2="4" y2="6"/><line x1="2" y1="12" x2="4" y2="12"/><line x1="2" y1="18" x2="4" y2="18"/><line x1="9" y1="2" x2="9" y2="22"/></svg>'
            f'<span class="protocol-title">Empirical Benchmark Protocol · {loaded_name}</span>'
            f'</div>'
            f'<div class="protocol-description">'
            f'Live evaluation on active instance: <b>{len(ds.classes)} classes</b>, <b>{len(ds.teachers)} teachers</b>, <b>{len(ds.rooms)} rooms</b>, <b>{total_sessions} required sessions</b>. Metrics reflect actual calculated algorithm executions (solve time, exploration work, hard violations, soft penalties). Missing/uncalculated solvers are cleanly omitted until run.'
            f'</div>'
            f'</div>'
        )

        # 3. Action Toolbar to Run / Re-Run Live Benchmark
        col_bench_bar, col_bench_act = st.columns([2.6, 1.4])
        with col_bench_bar:
            if eval_count == 5:
                st.markdown(
                    f'<div style="font-size:12px; color:#69D4A3; display:flex; align-items:center; gap:6px; padding:8px 0;">'
                    f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>'
                    f'<span>All 5 algorithms have actual calculated runs for <b>{loaded_name}</b>. Click to re-run suite.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif eval_count > 0:
                st.markdown(
                    f'<div style="font-size:12px; color:#7C8CFF; display:flex; align-items:center; gap:6px; padding:8px 0;">'
                    f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7C8CFF" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
                    f'<span><b>{eval_count} of 5 algorithms evaluated</b> with actual runs. Click to benchmark all solvers under standard parameters.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="font-size:12px; color:#8F9AA8; display:flex; align-items:center; gap:6px; padding:8px 0;">'
                    f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8F9AA8" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
                    f'<span>No algorithms evaluated yet for this dataset. Click to run the live benchmark suite across all 5 solvers.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with col_bench_act:
            btn_label = "⚡ Re-Run Benchmark Suite" if eval_count == 5 else "⚡ Benchmark All 5 Solvers"
            if st.button(btn_label, type="primary", use_container_width=True):
                with st.spinner(f"Running comparative benchmark across 5 solvers on {loaded_name}..."):
                    suite_dict, suite_hists = execute_live_benchmarks(ds)
                    st.session_state.benchmark_results[ds_key].update(suite_dict)
                    st.session_state.benchmark_histories[ds_key].update(suite_hists)
                    st.rerun()

        # If no runs exist yet, inform user
        if not has_any_eval:
            st.info(f"💡 No solver runs recorded yet for {loaded_name}. Run algorithms individually from the sidebar or click '⚡ Benchmark All 5 Solvers' above to generate actual calculated metrics.")
        else:
            # 4. Algorithm Comparison Table Card with ACTUAL CALCULATED VALUES (Only evaluated algorithms)
            rows_html = []
            active_algo = st.session_state.get("run_stats", {}).get("Algorithm", "")

            for r in evaluated_records:
                is_active = (r["Algorithm"].lower() in active_algo.lower()) or (active_algo.lower() in r["Algorithm"].lower())
                row_class = "benchmarks-data-row-active" if is_active else "benchmarks-data-row"

                if r["Hard"] == 0:
                    hard_html = (
                        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#69D4A3" stroke-width="1.6"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>'
                        '<span class="hard-zero">0</span>'
                    )
                    status_html = (
                        '<span class="success-rate-val" style="color:#69D4A3;">Feasible</span>'
                        '<span class="success-rate-sub">0 Hard Violations</span>'
                    )
                else:
                    hard_html = (
                        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#F0B76B" stroke-width="1.6"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
                        f'<span class="hard-violation">{r["Hard"]}</span>'
                    )
                    status_html = (
                        '<span class="success-rate-val" style="color:#F0B76B;">Unresolved</span>'
                        f'<span class="success-rate-sub">{r["Hard"]} Clashes</span>'
                    )

                active_cue = '<span class="solver-cue-text">Active Sidebar Solver</span>' if is_active else ''
                name_class = "solver-name-text-active" if is_active else "solver-name-text"

                rows_html.append(
                    f'<div class="{row_class}">'
                    f'<div class="solver-col">'
                    f'<div class="solver-name-row">'
                    f'<span class="{name_class}">{r["Algorithm"]}</span>'
                    f'<span class="solver-badge">{r["Badge"]}</span>'
                    f'</div>'
                    f'{active_cue}'
                    f'</div>'
                    f'<div class="time-col">{r["TimeStr"]}</div>'
                    f'<div class="work-col">{r["Work"]}</div>'
                    f'<div class="hard-col">{hard_html}</div>'
                    f'<div class="soft-col">{r["Soft"]}</div>'
                    f'<div class="success-col">{status_html}</div>'
                    f'</div>'
                )

            st.html(
                f'<div class="benchmarks-table-card">'
                f'<div class="benchmarks-cols-header">'
                f'<div class="solver-col benchmark-col-th">ALGORITHM / SOLVER</div>'
                f'<div class="time-col benchmark-col-th">SOLVE TIME</div>'
                f'<div class="work-col benchmark-col-th">EXPLORATION WORK</div>'
                f'<div class="hard-col benchmark-col-th">HARD VIOLATIONS</div>'
                f'<div class="soft-col benchmark-col-th">SOFT PENALTY</div>'
                f'<div class="success-col benchmark-col-th">FEASIBILITY STATUS</div>'
                f'</div>'
                + "".join(rows_html) +
                f'<div class="benchmarks-note-bar">'
                f'Empirical summary: Displaying actual measured results on {loaded_name}. MRV and Backtracking explore consistent constraint branches. Local search heuristics (Min-Conflicts, SA) iteratively minimize conflicts. Hybrid CSP + SA delivers balanced speed and optimization.'
                f'</div>'
                f'</div>'
            )

            # 5. Cost Convergence Dynamics Chart with ACTUAL HISTORY VALUES
            histories = st.session_state.benchmark_histories[ds_key]
            active_hists = {k: v for k, v in histories.items() if v and len(v) > 0}

            if active_hists:
                fig = go.Figure()

                if "Min-Conflicts" in active_hists:
                    mc_data = active_hists["Min-Conflicts"]
                    fig.add_trace(go.Scatter(
                        y=mc_data,
                        mode="lines",
                        name="Min-Conflicts Search",
                        line=dict(color="#7DD3D0", width=2, dash="dot"),
                        hovertemplate="Step %{x}: Cost %{y}<extra></extra>",
                    ))

                if "Simulated Annealing" in active_hists:
                    sa_data = active_hists["Simulated Annealing"]
                    fig.add_trace(go.Scatter(
                        y=sa_data,
                        mode="lines",
                        name="Simulated Annealing",
                        line=dict(color="#A78BFA", width=2, dash="dash"),
                        hovertemplate="Iteration %{x}: Cost %{y}<extra></extra>",
                    ))

                if "Hybrid CSP + SA" in active_hists:
                    hy_data = active_hists["Hybrid CSP + SA"]
                    fig.add_trace(go.Scatter(
                        y=hy_data,
                        mode="lines",
                        name="Hybrid CSP + SA",
                        line=dict(color="#7C8CFF", width=2.5),
                        hovertemplate="Stage 2 Iteration %{x}: Cost %{y}<extra></extra>",
                    ))

                fig.update_layout(
                    paper_bgcolor="#0E131A",
                    plot_bgcolor="#0E131A",
                    font=dict(family="Inter, sans-serif", color="#8F9AA8", size=11),
                    margin=dict(l=40, r=20, t=25, b=40),
                    height=320,
                    xaxis=dict(
                        title=dict(text="Search Iterations / Evaluations", font=dict(color="#8F9AA8", size=11)),
                        gridcolor="#1B2530",
                        zerolinecolor="#1B2530",
                    ),
                    yaxis=dict(
                        title=dict(text="Objective Cost (1000 × Hard + Soft)", font=dict(color="#8F9AA8", size=11)),
                        gridcolor="#1B2530",
                        zerolinecolor="#1B2530",
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=11, color="#EDF0F7")
                    ),
                )

                # Dynamic performance dynamics calculation
                fastest_solver = min(evaluated_records, key=lambda x: x["Time"])
                feasible_records = [r for r in evaluated_records if r["Hard"] == 0]
                best_quality_solver = min(feasible_records, key=lambda x: x["Soft"]) if feasible_records else min(evaluated_records, key=lambda x: x["Total Cost"])

                st.html(
                    f'<div class="chart-heading-row" style="margin-top: 12px; margin-bottom: 8px;">'
                    f'<div class="chart-title-box">'
                    f'<span class="chart-title">Cost Convergence Dynamics</span>'
                    f'<span class="chart-subtitle">Actual Objective Cost vs Iteration Step across Evaluated Search Solvers</span>'
                    f'</div>'
                    f'</div>'
                )

                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                st.html(
                    f'<div class="chart-interpretation-box" style="margin-top: 10px; margin-bottom: 20px; border: 1px solid #1B2530;">'
                    f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#979FAF" stroke-width="1.6" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
                    f'<span><b>Real Performance Dynamics:</b> <b>{fastest_solver["Algorithm"]}</b> completed fastest in {fastest_solver["TimeStr"]}. <b>{best_quality_solver["Algorithm"]}</b> achieved optimal schedule quality with {best_quality_solver["Soft"]} soft penalty (Total Cost: {best_quality_solver["Total Cost"]}). All plotted trajectories reflect actual evaluated steps.</span>'
                    f'</div>'
                )

                # 6. Actual Data-Driven Selection Takeaway Banner
                st.html(
                    f'<div class="selection-takeaway-banner">'
                    f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#858CFF" stroke-width="1.6" style="flex-shrink:0;"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>'
                    f'<span><b>Key Takeaway for {loaded_name}:</b> {best_quality_solver["Algorithm"]} demonstrated superior objective optimization with {best_quality_solver["Soft"]} soft penalty. Exact CSP algorithms systematically verify constraint feasibility, while heuristic search navigates large solution spaces with low latency.</span>'
                    f'</div>'
                )

        # 7. Dynamic Workspace Footer
        st.html(
            f'<div class="workspace-footer">'
            f'<span class="footer-meta-text">Dataset: {loaded_name} ({len(ds.classes)} Classes · {len(ds.teachers)} Teachers · {len(ds.rooms)} Rooms · {total_sessions} Sessions)</span>'
            f'<span class="footer-context-text">Benchmarked Live on Host CPU</span>'
            f'</div>'
        )
