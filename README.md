# Low-carbon, redundancy-aware truss optimizer — web app

Interactive Streamlit application for the surrogate-assisted, redundancy-aware
multi-objective optimization framework. The user selects one of four benchmark
trusses (10-bar, 25-bar, 120-bar dome, 137-bar Burro Creek bridge), tunes the
design parameters, and runs the surrogate-assisted optimizer to obtain the
embodied-carbon-versus-redundancy Pareto front.

The full-evaluation NSGA-II used to validate the method in the paper is omitted
here; the app runs the validated surrogate-assisted optimizer alone so that
results return quickly.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud (free)

1. Create a public GitHub repository and push all the files in this folder
   (app.py, the `*.py` modules, the two logo PNGs, and requirements.txt).
2. Go to https://share.streamlit.io , sign in with GitHub, and choose
   "New app".
3. Select the repository, set the main file to `app.py`, and deploy.
4. The app will build from requirements.txt and become available at a public
   `*.streamlit.app` URL that can be cited in the paper.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit interface |
| `problems_all.py` | The four benchmark truss models (FEM + FORM) |
| `mo_optimize.py` | RBRO problem wrapper (carbon, redundancy, feasibility) |
| `surrogate_model.py` | Failure-aware two-stage surrogate |
| `surrogate_opt.py` | Sampling and non-dominated helpers |
| `surrogate_opt2.py` | Surrogate-assisted optimizer |
| `redundancy_index.py` | Redundancy objective |
| `carbon_redundancy.py` | Embodied-carbon model and factors |
| `logo_csie.png`, `logo_rostock.png` | Institutional logos |

## Notes

- The 137-bar bridge is the slowest case; keep the search budget at "Quick" or
  "Balanced" on the free cloud tier to stay within the request timeout.
- The carbon factors default to representative global values
  (BOF 2.30, EAF 0.70 kgCO2e/kg) and can be edited in the sidebar.
