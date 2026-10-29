# ⚡ UI Engineering & Timetabling Skills Guide (`skills.md`)

## 1. Domain Expertise: AI Timetabling Visualization
Timetable generation is an NP-hard Constraint Satisfaction Problem (CSP) involving high multidimensionality:
$$\text{Variables } X = \{S_1, S_2, \dots, S_n\} \quad (\text{Sessions})$$
$$\text{Domains } D = \{(\text{Day}, \text{Period}, \text{Room})\}$$

### Visualization Challenge:
A linear list of assignments is incomprehensible to university administrators, teachers, and students. A top-tier UI must project the 4D assignment space $(\text{Session}, \text{Day}, \text{Period}, \text{Room})$ into human-intuitive 2D coordinate planes:
1. **Class View**: Fixed Class $\rightarrow$ Grid of $\text{Day} \times \text{Period} \rightarrow$ Cell shows $(\text{Subject}, \text{Teacher}, \text{Room})$.
2. **Teacher View**: Fixed Teacher $\rightarrow$ Grid of $\text{Day} \times \text{Period} \rightarrow$ Shows teaching load, gaps, and unavailable slots.
3. **Room View**: Fixed Room $\rightarrow$ Grid of $\text{Day} \times \text{Period} \rightarrow$ Shows room occupancy, capacity utilization, and idle periods.

---

## 2. Advanced Streamlit CSS Architecture Skills

### 2.1 CSS Injection Technique
Streamlit widgets are rendered inside Shadow DOM-like structures or emotion-cache divs. To reliably style them without fragile selector breaks:
- Inject CSS at the very top using `st.markdown('<style>...</style>', unsafe_allow_html=True)`.
- Target stable data attributes like `[data-testid="stSidebar"]`, `[data-testid="stAppViewContainer"]`, `[data-testid="stMetric"]`, and `.stTabs`.
- Hide distracting Streamlit chrome (deploy button, footer, header decorations) to produce an authentic standalone SaaS software feel:
  ```css
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stToolbar"] { display: none; }
  ```

### 2.2 Dark Theme Override Engine
Ensure pure black and deep obsidian surfaces rather than the washed-out default gray:
- Background: `#0B0F14` (Main body), `#0E131A` (Sidebar & Cards).
- Hairline Borders: `1px solid #1B2530`.
- Custom Scrollbars: Sleek thin scrollbars (`::-webkit-scrollbar { width: 6px; height: 6px; }`).

### 2.3 HTML Table / Card Render Engine
Instead of plain text dataframes, render dynamic HTML tables with inline flex cards:
- Multi-period lab sessions: Rendered with distinct cyan backgrounds (`#12343A`) and badges.
- Theory sessions: Rendered with indigo accents (`#7C8CFF`).
- Lunch breaks: Diagonal CSS repeating-linear-gradient stripes (`repeating-linear-gradient(45deg, #0E131A, #0E131A 4px, #161D26 4px, #161D26 8px)`).
- Free periods: Clean dotted borders with muted labels.

---

## 3. UI/UX Minimalism & Ergonomic Principles

### 3.1 Visual Noise Reduction
- **Zero Heavy Drop Shadows**: Use border contrast (`#1B2530` against `#0E131A`) and subtle glow highlights rather than blurry black shadows.
- **Micro-Badges**: Small uppercase pill badges with 9px font size and `letter-spacing: 0.5px` categorize session types instantly.
- **Contextual KPIs**: Key performance indicators displayed in compact cards with SVG icons, showing both the absolute metric and its delta/significance.

### 3.2 High-Feedback Interactive States
- **Instant Feasibility Banner**: Green `#0F2A24` banner when hard violations equal 0; Warning banner with detailed bulleted violation breakdown when clashes exist.
- **Manual Move Conflict Detection**: Immediately executes `cost_function.count_violations` when any session is moved, providing instant visual feedback before finalizing changes.

---

## 4. Responsive Design Checklist
- [x] **Flexible Layout Grid**: Fluid flexbox container for workspace and sidebar.
- [x] **Horizontally Scrollable Timetable**: `overflow-x: auto` with sticky Day column on narrow screens.
- [x] **Mobile-Friendly KPI Wrap**: Metrics reorder into 2-column or 1-column layouts on mobile viewports.
- [x] **Accessible Contrast**: All text satisfies WCAG AA contrast (≥ 4.5:1 ratio against dark backdrops).
