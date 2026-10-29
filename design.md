# 🎨 Design Specification: AI Timetable Scheduler (Next-Gen Dark Mode)

## 1. Executive Summary & Design Philosophy
This document establishes the UI/UX design architecture for the **AI-Based Timetable Scheduler** college project. The aesthetic direction is **High-Performance Dark Minimalist**—taking direct inspiration from professional engineering dashboards like Linear, Raycast, and Vercel. 

### Core Pillars:
1. **Clarity Over Clutter**: Every pixel has a purpose. Subtle borders (`1px solid #1B2530`) replace loud shadows.
2. **Harmonious Color Hierarchy**: Deep obsidian backdrops (`#0B0F14` & `#0E131A`) paired with purposeful accent colors (Electric Indigo `#7C8CFF`, Mint Success `#69D4A3`, Cyan Teal `#7DD3D0`).
3. **Information Density & Legibility**: Academic timetables require dense data display without cognitive overload. Typography scale uses `Inter` with tight tracking and distinct weights.
4. **Fluid Responsiveness**: Full adaptation from desktop 1440px+ down to mobile devices with smooth horizontal scrolling for timetable grids.

---

## 2. Color System & Design Tokens

### 2.1 Surfaces & Backgrounds
| Token Name | Hex Code | HSL Equivalent | Role / Usage |
| :--- | :--- | :--- | :--- |
| `surface-ground` | `#0B0F14` | `hsl(214, 30%, 6%)` | Main canvas backdrop |
| `surface-sidebar` | `#0E131A` | `hsl(214, 30%, 8%)` | Configuration sidebar & panels |
| `surface-card` | `#11161D` | `hsl(214, 25%, 9%)` | Elevated KPI cards, tab active states |
| `surface-subtle` | `#161C24` | `hsl(214, 24%, 11%)` | Hover states, pill badges, inputs |
| `surface-border` | `#1B2530` | `hsl(213, 28%, 15%)` | Universal hairline divider / border |
| `surface-border-focus` | `#2D3B4E` | `hsl(215, 26%, 24%)` | Hovered or focused card borders |

### 2.2 Brand & Semantic Accents
| Token Name | Hex Code | Usage |
| :--- | :--- | :--- |
| `accent-primary` | `#7C8CFF` | Primary CTA, active tabs, sliders, Theory session badges |
| `accent-primary-hover` | `#96A3FF` | Primary button hover state |
| `accent-primary-glow` | `rgba(124, 140, 255, 0.15)` | Subtle ambient neon backlights |
| `accent-lab-bg` | `#12343A` | Multi-hour Lab cards background |
| `accent-lab-border` | `#1F4A3A` | Lab card borders |
| `accent-lab-text` | `#7DD3D0` | Lab tag badges & highlight text |
| `status-success-bg` | `#0F2A24` | Feasible schedule outcome banner |
| `status-success-border`| `#1F4A3A` | Feasibility banner outline |
| `status-success-text` | `#69D4A3` / `#7BE6B2` | 100% hard constraints satisfied |
| `status-warning-bg` | `#2A2210` | Infeasible / conflict warning banner |
| `status-warning-border`| `#4A3B18` | Warning outline |
| `status-warning-text` | `#FBBF24` | Conflict warning text |
| `status-danger-bg` | `#2A1315` | Critical violation tag |
| `status-danger-text` | `#F87171` | Hard clash alert |

### 2.3 Typography & Text Contrast
| Token Name | Hex Code | Purpose |
| :--- | :--- | :--- |
| `text-primary` | `#F5F7FA` | Primary titles, metric values, subject titles (high contrast) |
| `text-secondary` | `#8F9AA8` | Field labels, captions, teachers, room names |
| `text-muted` | `#5B6674` | Slider bounds, empty slot placeholders, copyright |
| `text-dim` | `#6F7A89` | Free periods, tertiary hints |

---

## 3. Typography Architecture
- **Font Family**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Monospace Family (for Metrics/IDs)**: `'JetBrains Mono', 'Fira Code', monospace`

### Scale Hierarchy:
- **Display Title**: 26px / 32px Line Height / Weight 600
- **Section Heading**: 18px / 24px / Weight 600
- **Subsection Title**: 16px / 20px / Weight 600
- **Body Bold / Metric**: 14px / 20px / Weight 600
- **Body Standard**: 12px / 16px / Weight 400
- **Small Label / Caption**: 11px / 14px / Weight 400-500
- **Micro Badge**: 9px-10px / 12px / Weight 500-600 (Uppercase, letter-spacing +0.5px)

---

## 4. UI Layout & Component Specifications

### 4.1 Global Dual-Pane Architecture
1. **Sidebar Navigation & Controls (`width: 300px`, sticky, scrollable)**:
   - Sticky top branding with `sliders-horizontal` icon and assignment scope.
   - Section 1: Problem Dataset (Dataset switcher, upload dropzone, loaded indicator, 2x2 stat matrix).
   - Section 2: AI Search Algorithm (Algorithm dropdown with strategy badge, algorithm-specific dynamic hyperparameter sliders, cooling rate bar, seed control).
   - Launch Engine CTA (`#7C8CFF` button with micro-bounce, glowing border, and sparkles icon).
   - Academic context footer (`graduation-cap` badge).

2. **Main Workspace Canvas (`flex: 1`, min-width: 0)**:
   - **Top Summary Header**: Title, dataset context, algorithm execution status badge.
   - **Outcome Banner**: Feasible / Infeasible responsive alert banner with check-circle / alert-triangle.
   - **KPI Metric Strip**: 5 glassmorphism cards (Algorithm, Time Taken, Hard Violations, Soft Penalty, Total Cost) with bespoke SVG icons.
   - **Segmented Results Tabs**:
     - `📅 Timetable Grid`: Multi-view schedule with Class/Teacher/Room orientation selector.
     - `✏️ Manual Editor`: Live conflict inspector & drag/select reassigner.
     - `📋 Constraints Breakdown`: Full H1–H9 and S1–S5 diagnostic audit tables.
     - `💾 Export Hub`: Clean CSV and multi-sheet Excel download center.
     - `🧪 Performance Analytics`: Benchmarks, convergence graphs, and search statistics.

### 4.2 Timetable Grid Anatomy
- **Grid Structure**:
  - Period Header row with explicit academic timings (e.g., `Period 1: 09:00 - 10:00`, `Period 2: 10:00 - 11:00`, `Lunch: 12:00 - 13:00`, etc.).
  - Day column (Monday–Friday) with sticky vertical axis.
- **Card Variations**:
  - **Theory Session**:
    - Container: `#11161D` with `#1B2530` border.
    - Badge: `#7C8CFF` pill with `THEORY`.
    - Subject: High contrast `#F5F7FA` 12px bold.
    - Meta: Teacher name + Room tag `#8F9AA8`.
  - **Lab Session (Multi-period Spanning)**:
    - Container: `#12343A` with `#1F4A3A` border.
    - Badge: `#7DD3D0` pill with `LAB (2H)`.
    - Subject + Teacher + Lab Room.
  - **Lunch Break**:
    - Centered cutlery icon `utensils`, subtle repeating diagonal hatch pattern `#1B2530`, clean label.
  - **Free Period**:
    - Minimalist dashed container with subtle `Free` indicator.

---

## 5. Responsiveness & Adaptive Breakpoints
1. **Desktop (> 1280px)**: Two-column layout with 300px sticky sidebar and expansive timetable canvas.
2. **Laptop / Tablet Landscape (992px – 1279px)**: Sidebar narrows or collapses smoothly; timetable table enables horizontal frictionless momentum scroll.
3. **Mobile (< 992px)**: Sidebar transforms into an off-canvas drawer or collapsible top panel; KPI cards wrap to 2-column or 3-column grid; timetable grid preserves readability with sticky day labels.

---

## 6. Micro-Interactions & Transitions
- Button Hover: `transform: translateY(-1px); box-shadow: 0 4px 20px rgba(124, 140, 255, 0.25);`
- Card Hover: subtle border shift from `#1B2530` to `#2D3B4E`.
- Tab Transition: `cubic-bezier(0.4, 0, 0.2, 1)` smooth color and indicator fade.
- Metric Counter: crisp rendering with zero layout jitter.
