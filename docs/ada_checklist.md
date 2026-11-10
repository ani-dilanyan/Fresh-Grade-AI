# Accessibility (ADA / WCAG 2.1 AA) checklist — FreshGrade AI

Owner: Member 4. Use this table as the "ADA compliance checklist and evidence" appendix.
Re-check before the defense with the WAVE browser extension (wave.webaim.org) or axe DevTools,
and paste screenshots of the results next to each row.

| # | WCAG criterion | How FreshGrade meets it | Where | Evidence to collect |
|---|---|---|---|---|
| 1 | 1.1.1 Non-text content | Every image has descriptive alt text generated from the results (fruit, class, defect %) | `results.html` | Screenshot of WAVE "alt" icons |
| 2 | 1.3.1 Info & relationships | Semantic landmarks (`header`, `nav`, `main`, `footer`), one `h1` per page, `th scope` in tables, `fieldset/legend` in forms | `base.html`, all pages | WAVE structure view |
| 3 | 1.4.1 Use of colour | Grades shown as **text + icon + colour** (e.g. "✓ Class I"), never colour alone; overlay labels name the class in text | badges, overlay | Screenshot in greyscale mode |
| 4 | 1.4.3 Contrast (minimum) | Design tokens chosen for ≥ 4.5:1 text contrast (e.g. #18231d on #f6f5ef, white on #1f5c3f) | `styles.css` | WAVE contrast report: 0 errors |
| 5 | 1.4.10 Reflow | Responsive grid; works at 320 px width without horizontal scrolling (tables scroll inside their own container) | `styles.css` | Screenshot at 400 % zoom |
| 6 | 2.1.1 Keyboard | Every action reachable with Tab/Enter; **Space** captures the live inspection; no keyboard traps | all pages, `camera.js` | Video: full flow keyboard-only |
| 7 | 2.4.1 Bypass blocks | "Skip to main content" link as first focusable element | `base.html` | Screenshot of focused skip link |
| 8 | 2.4.3 / 2.4.7 Focus order & visible focus | Logical DOM order; 3 px blue focus outline on every control | `styles.css` | Screenshot |
| 9 | 2.4.4 Link purpose | Descriptive link texts; hidden context added to repeated "PDF" links | `history.html` | Screen-reader readout |
| 10 | 2.5.5 Target size | Buttons ≥ 44 px high | `styles.css` | Measurement in DevTools |
| 11 | 3.1.1 Language of page | `<html lang="en">` | `base.html` | Source view |
| 12 | 3.3.1 / 3.3.2 Error identification & labels | All inputs have `<label>`; upload/analysis errors shown as text in `role="alert"`/`role="status"` regions | `upload.html`, `live.html` | Screenshot of error message |
| 13 | 4.1.3 Status messages | Inspection results and live status announced via `aria-live="polite"` | `live.html`, `results.html` | NVDA / VoiceOver recording |
| 14 | 2.3.3 / reduced motion | `prefers-reduced-motion` disables transitions | `styles.css` | — |
| 15 | PDF accessibility | Reports use real text (not images of text), document title/author metadata, logical reading order | `reports/*.py` | Open PDF, select text |
| 16 | Works without JavaScript | Upload page posts a normal form (`/upload`) and redirects to results | `upload.html`, `main.py` | Test with JS disabled |

## Manual test script (5 minutes)
1. Unplug the mouse. Tab from the address bar: the skip link appears first.
2. Go to **Upload**, choose a sample image with the keyboard, submit with Enter.
3. On the results page, Tab through the report links; check the focus outline is always visible.
4. Turn on a screen reader (Windows: NVDA, free; macOS: VoiceOver with Cmd+F5) and confirm the result
   summary is read out automatically.
5. Switch the OS / browser to greyscale and confirm every grade can still be identified by its text.
