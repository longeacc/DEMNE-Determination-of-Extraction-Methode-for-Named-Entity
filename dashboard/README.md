# demne Dashboard

This folder contains the interactive NLP supervision GUI for the **demne** research project.
The Streamlit application lets you analyse level-2 (L2) metrics and act on the cascade routing (Rules → Transformers → LLM) according to scores computed by the pipeline or edited manually by the user.

---

## Table of Contents & Interfaces

1. **Metrics Dashboard** (1_Dashboard_Metriques.py)
   - Analytical view of the "Frugal Cascade" algorithm.
   - **Real-time Editing**: the metrics table lets users click and manually modify scores (Te, He, R, Yield, etc.). Charts (Radar, Scatter) and the model decision (Rules, LLM, etc.) are instantly recomputed.
   - Threshold sliders to compare energy-frugality modes.

2. **CLI Console** (2_Console_CLI.py)
   - Simulator and wrapper for the `main.py` terminal application.
   - Lets you launch `extract`, `batch`, and `metrics` commands directly and view terminal logs inside the web interface.

3. **REST Integration & Config** (3_REST_Integration.py)
   - Page dedicated to L2 JSON configuration export management.
   - Filter/exclude specific entities from the Dashboard without losing them.

4. **REST Server & Jupyter** (4_Notebook_REST.py)
   - Control panel for the clinical report processing API (DuraXell REST).
   - Embeds the Jupyter notebook (REST.ipynb) directly in an IFrame for interactive cross-testing on the same web server.

---

## Optimal Launch

1. **Activate the virtual environment** and ensure dependencies are up to date:

   ```bash
   pip install -r dashboard/requirements.txt
   ```

   *Tip:* Install Jupyter (`pip install jupyter`) if you plan to use the interactive Notebook IFrame.

2. **Start the application** — open a terminal in the `dashboard/` folder:

   ```bash
   cd dashboard
   streamlit run app.py
   ```

3. **Default port**: the interface will be served at `http://localhost:8501`.

---

## Editing Interactive Metrics (Demo / Playground)

On the **Metrics Dashboard L2** page, find the **Analytical Summary (Editable)** component.
Click any cell in the table (except the Entity and Decision columns):
- Change, for example, the `Yield` or `R` value of a biomarker (e.g. Ki67).
- Confirm the entry (press Enter) and you will immediately see the entity drop to LLM level or rise to Rule level. The charts below will reflect the update.

For the API and the Notebook, go to **REST Server & Jupyter**, press "Launch API" and/or "Launch Jupyter"; the IFrame will interactively display the environment.
