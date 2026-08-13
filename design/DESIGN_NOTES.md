# AERIA Design Notes

Design direction for the DFW Air Quality Dashboard UI overhaul.

This document defines what is **locked** (decided, do not relitigate) and what is **open** (intentional flexibility for Claude Code during implementation). It exists so any future implementation session has a single source of truth for design intent.

The HTML and screenshots in this folder are visual references — not implementation targets. The information architecture and aesthetic direction are what carry forward; the specific Three.js code is being replaced with React Three Fiber in the production build.

---

## Product Identity

**Name:** AERIA
**Tagline:** DFW · PM₂.₅ Atlas
**One-line description:** A real-time, street-level air quality dashboard for the Dallas–Fort Worth metro.
**Emotional target:** "Wow, I can see my city's air." Architectural and atmospheric, not analytical.

---

## Locked — Information Architecture

These decisions are final. They were validated across multiple iteration rounds and represent the working product structure.

### Two primary views

1. **City overview** — top-down isometric 3D scene of the DFW metro with the bounding box outlined, grid cells, buildings, particles representing air quality, and clickable interactivity per cell.
2. **Street view** — first-person ground-level scene the user drops into when they click a cell. Reusable stylized scene; only the air quality state changes per cell.

### Top navigation tabs

- **City overview** (default)
- **Street view** (active when dropped into a cell)
- **Time machine** (future — historical playback)
- **Route lab** (future — cleanest-path optimizer)

### Top status bar (right-aligned)

- Live indicator with sensor count
- Average PM₂.₅ across the network
- Wind speed and direction
- "Updated N min ago" timestamp

### Persistent left panel

Always visible in both views. Collapsible header with the AQI category at a glance.

Sections, top to bottom:
1. **Air Quality** header — category label (Moderate, Unhealthy/Sens., etc.) with colored dot
2. **Reading** — large PM₂.₅ value, delta vs. 24h, attribution line ("EPA-corrected · IDW from N nearby sensors · High confidence")
3. **Who should take care** — checkbox list with health guidance for sensitive groups and general public, content driven by current AQI category
4. **Activity guidance** — checkbox list for outdoor exercise, windows, masks, also AQI-driven
5. **Cell breakdown** — small stat tiles: traffic adjustment, wind adjustment, highway distance, last updated time

The panel updates dynamically as the user clicks different cells.

### Top-right cell info card (city overview only)

When a cell is selected, this card appears in the top-right:
- Zip code and cell ID (e.g. "ZIP 752-78 / Cell 13·17 · Good")
- Lat/lon coordinates
- PM₂.₅ value and AQI number
- "Drop into street" button
- "Pin" button

### Find by zip search

Search box near the top of the city overview. Typing a zip pans the camera to that cell, selects it, and updates the panel and info card. This is the most important user-facing entry point — someone opens the dashboard, types their zip, and immediately sees their air quality.

### Find a route button

Lower-right of the city overview. Placeholder for now. Label: "Find a route · Cleanest path · Soon"

### Bottom timeline scrubber

Spans the full width below the 3D scene. "NOW" marker on the left, current time on the right. Future use: scrub historical data when Time Machine is implemented.

### Bottom breadcrumb footer

Small text at the very bottom showing current navigation path:
`AERIA.ATLAS > DFW > CITY OVERVIEW > CELL 13·17 · ZIP 752-78`
Plus build metadata on the right: bbox coordinates, IDW parameters, build version.

---

## Locked — Grid System

The grid is the foundation of the entire experience. These are non-negotiable.

- **30 × 30 = 900 cells** covering the bounding box (lat 32.55→33.08, lon -97.05→-96.46)
- Each cell represents roughly a 1-mile geographic square
- **Zip code is the primary cell identifier** — pulled from reverse geocoding, displayed on a subset of cells (one label per ~3×3 cluster) to avoid visual noise
- **City and neighborhood labels** float above the grid as a higher-altitude reference layer (Plano · Frisco, Downtown Dallas, Garland, etc.) — they do not define the grid structure
- **Cells are clickable.** Hover shows a tooltip with zip, PM₂.₅, AQI category, coverage. Click selects the cell and updates the panel + info card.
- **Cell borders stay neutral** — never color them by AQI. The particles do that work.

---

## Locked — AQI Communication Hierarchy

Three layers of signal in order of strength:

1. **Particles (primary)** — colored floating specs above the city. Color encodes AQI category, density encodes severity. The dominant visual signal at the city scale and the immersive signal in the street view.
2. **Cell floor tint (secondary)** — very subtle semi-transparent AQI tint on each cell's ground plane. Faint enough not to compete with the particles but present enough to scan at a glance.
3. **Tooltip and panel (precise)** — exact PM₂.₅ values appear only on interaction.

### AQI color scale

The standard EPA breakpoints, used only for particles:

| Category | Range (µg/m³) | Color |
|---|---|---|
| Good | 0 – 12 | Green |
| Moderate | 12 – 35 | Yellow |
| Unhealthy/Sensitive | 35 – 55 | Orange |
| Unhealthy | 55 – 150 | Red |
| Very Unhealthy | 150 – 250 | Purple |
| Hazardous | 250+ | Dark red |

---

## Locked — Aesthetic Direction

The look established by the AERIA mockup is the design north star.

- **Dark, atmospheric, cinematic.** Dusk-toned palette. Deep night blues bleeding into magenta haze at the horizon, with warm gold accents for live signals and selected states.
- **Grey-toned buildings** with warm light catch on tall faces — concrete reading against dusk sky, not metallic or chrome.
- **Gold (`#ffd166`) is the signature accent.** Used for the live indicator, selected cell ring, "Drop into street" button, "Find a route" button, and any active/highlighted state. Not green — gold reads as warmer and more atmospheric.
- **Magenta (`#c66dd6`) and teal (`#6fd0c5`) appear sparingly** as secondary highlights.
- **Typography is precise and editorial.**
  - Body: Inter Tight (sans-serif, 14px base, weight 400/500)
  - Numerics and metadata: JetBrains Mono (uppercase, letterspaced)
  - Display readings: Fraunces serif for the large PM₂.₅ number
- **Hairline borders** (`rgba(255,240,220,0.06)` to `0.11`) — never solid lines, always whisper-thin separators.
- **No emoji, no rainbow palettes, no decorative gradients on chrome.** Only the AQI particles carry color.

### Street view atmospheric reference

The first-person street view as built in the mockup is the high-water mark for atmosphere. Particles fill the air around the viewer, color and density driven by AQI. Buildings are stylized grey rectangles with subtle shading. The scene is reusable across all 900 cells — only the air quality state changes.

When implementing the production version, match or exceed this atmospheric quality. The mockup proves it's achievable.

---

## Open — Implementation Decisions

These are intentionally not specified. Use best judgment in the production R3F build.

- **Camera angle, FOV, and altitude** for the city overview (the mockup's values were too distant; the real implementation should bring the city closer and feel more architectural)
- **Building generation strategy** — procedural or asset-based, instanced or individual meshes, exact density per zone, height distributions
- **Particle system implementation** — instanced sprites, GPU shaders, or point clouds — whatever performs best for ~900 cells × N particles
- **Lighting setup** — directional sun, hemisphere, ambient, with shadow mapping or without — the mockup struggled with this; treat it as open
- **Camera controls** — orbit, pan, zoom limits — pick what feels good
- **Transition between city and street views** — animated fly-down, hard cut, fade, or split-screen — designer's choice
- **Mobile responsiveness** — out of scope for v1, optimize for desktop
- **Loading states and skeleton screens** — use judgment
- **Animation and motion design** — subtle ambient motion is welcomed but not required

---

## Reference Files in This Folder

| File | Purpose |
|---|---|
| `index.html` | Final mockup HTML — layout, panel structure, typography, chrome |
| `scene.js` | Final mockup Three.js code — building generation, particles, camera (reference only, will be reimplemented in R3F) |
| `screens/city-overview.png` | Latest city overview screenshot |
| `screens/street-view.png` | Latest street view screenshot — represents the aesthetic high-water mark |
| `screens/panel-states.png` | Left panel collapsed and expanded states |

---

## What Not to Do

- Do not retreat to the Streamlit/Folium aesthetic — that app stays running in parallel but the new UI is not a port
- Do not use AQI colors for chrome (borders, buttons, text). Particles only.
- Do not introduce a fourth view. Two views (city + street), two future tabs (time machine + route lab). That's the full surface.
- Do not abandon the 900-cell grid for fewer cells. The granularity is the product.
- Do not lose the zip code as the primary cell identifier in favor of city names.
- Do not redesign the panel structure. Sections and order are locked.

---

## Implementation Sequence

The production build is sequenced across roughly six Claude Code sessions:

1. FastAPI backend scaffold wrapping existing pipeline
2. Vite + React + R3F frontend scaffold talking to backend
3. City overview 3D scene — grid, buildings, particles, camera
4. Cell interactions — hover, click, zip search, info card
5. Left panel — health guidance, cell breakdown, navigation chrome
6. Street drop-in view — first-person scene with particles

The route optimizer lands cleanly into the finished UI as a later increment.

---

*Last updated: 2026-05-01*
