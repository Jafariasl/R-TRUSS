"""
app.py — Surrogate-assisted low-carbon, redundancy-aware truss optimization
============================================================================
Streamlit web application for the framework presented in the paper. The user
selects one of four benchmark trusses, tunes the design parameters (load scale,
carbon factors, target reliability, search budget), and runs the surrogate-
assisted optimizer to obtain the carbon-vs-redundancy Pareto front. The full-
evaluation genetic algorithm is intentionally omitted here; it was used in the
paper only to validate the surrogate, so the online tool runs the validated
surrogate-assisted optimizer alone to keep the computation fast.

Run locally:
    pip install streamlit numpy scipy scikit-learn pandas plotly pillow
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import problems_all as PA
import mo_optimize as MO
import surrogate_opt2 as SO2
import carbon_redundancy as cr

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Low-Carbon Redundancy-Aware Truss Optimizer",
    page_icon="◭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design tokens and global styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=Spline+Sans+Mono:wght@400;500;600&display=swap');

    :root{
        --steel:#1b2a3a;          /* deep structural slate */
        --steel-2:#24384c;
        --ink:#0f1822;
        --paper:#f3f1ea;          /* drafting paper */
        --line:#c8cdd4;
        --carbon:#b4571f;         /* embodied-carbon accent (rust) */
        --green:#2f7d5b;          /* low-carbon / recycled accent */
        --muted:#6b7785;
    }

    html, body, [class*="css"]{
        font-family:'Archivo', system-ui, sans-serif;
    }
    .stApp{
        background:
            linear-gradient(180deg, #f6f4ee 0%, #eceae2 100%);
    }

    /* hide default chrome */
    #MainMenu, footer, header{visibility:hidden;}

    /* ---- masthead ---- */
    .masthead{
        background:linear-gradient(135deg, var(--steel) 0%, var(--ink) 100%);
        border-radius:14px;
        padding:26px 30px;
        margin-bottom:6px;
        color:var(--paper);
        box-shadow:0 10px 30px rgba(15,24,34,.22);
    }
    .masthead h1{
        font-size:1.72rem;
        font-weight:800;
        letter-spacing:-.02em;
        line-height:1.12;
        margin:0 0 6px 0;
        color:#fff;
    }
    .masthead .accent{color:#e9a06a;}
    .masthead p{
        font-size:.96rem;
        color:#aebccb;
        margin:0;
        max-width:62ch;
        line-height:1.5;
    }
    .eyebrow{
        font-family:'Spline Sans Mono', monospace;
        font-size:.72rem;
        letter-spacing:.22em;
        text-transform:uppercase;
        color:#7e93a8;
        margin-bottom:10px;
    }

    /* ---- objective chips ---- */
    .chips{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;}
    .chip{
        font-family:'Spline Sans Mono', monospace;
        font-size:.74rem;
        padding:6px 12px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,.16);
        color:#cdd8e3;
    }
    .chip b{color:#fff;font-weight:600;}
    .chip.carbon{border-color:rgba(180,87,31,.5);}
    .chip.green{border-color:rgba(47,125,91,.55);}

    /* ---- section labels ---- */
    .seclabel{
        font-family:'Spline Sans Mono', monospace;
        font-size:.74rem;
        letter-spacing:.16em;
        text-transform:uppercase;
        color:var(--muted);
        border-bottom:1px solid var(--line);
        padding-bottom:6px;
        margin:8px 0 14px 0;
    }

    /* ---- result metric cards ---- */
    .metricwrap{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 8px 0;}
    .metric{
        flex:1;min-width:150px;
        background:#fff;
        border:1px solid var(--line);
        border-left:4px solid var(--steel);
        border-radius:10px;
        padding:14px 16px;
    }
    .metric .k{
        font-family:'Spline Sans Mono', monospace;
        font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;
        color:var(--muted);margin-bottom:6px;
    }
    .metric .v{font-size:1.5rem;font-weight:700;color:var(--ink);line-height:1;}
    .metric .u{font-size:.78rem;color:var(--muted);font-weight:500;}
    .metric.carbon{border-left-color:var(--carbon);}
    .metric.green{border-left-color:var(--green);}

    /* sidebar */
    section[data-testid="stSidebar"]{
        background:#fff;border-right:1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}

    /* run button */
    .stButton>button{
        background:var(--steel);color:#fff;border:0;border-radius:9px;
        font-weight:600;font-size:.95rem;padding:.6rem 1rem;width:100%;
        transition:background .15s ease;
    }
    .stButton>button:hover{background:var(--carbon);color:#fff;}

    /* footer credit */
    .credit{
        margin-top:8px;text-align:center;
        font-size:.74rem;color:var(--muted);line-height:1.5;
    }
    .credit b{color:var(--ink);font-weight:600;}
    .journal{
        font-family:'Spline Sans Mono', monospace;
        font-size:.7rem;letter-spacing:.04em;color:var(--muted);
        text-align:center;margin-top:2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Benchmark metadata
# ----------------------------------------------------------------------
BENCH = {
    "10-bar planar truss": dict(
        key="10-bar", groups=10, mode="exact", k=None, units="imperial",
        note="Planar, displacement-controlled. Fast.", scale_loads=True),
    "25-bar spatial truss": dict(
        key="25-bar", groups=8, mode="exact", k=None, units="imperial",
        note="Spatial, stress and displacement limits. Fast.", scale_loads=False),
    "120-bar dome": dict(
        key="120-bar", groups=7, mode="approx", k=5, units="imperial",
        note="Spatial dome, larger model. Moderate runtime.", scale_loads=False),
    "137-bar Burro Creek bridge": dict(
        key="137-bar", groups=4, mode="approx", k=4, units="SI",
        note="Real bridge, 137 members. Slowest; keep the budget modest.",
        scale_loads=True),
}


# ----------------------------------------------------------------------
# Masthead
# ----------------------------------------------------------------------
def b64_logo(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


csie = b64_logo("logo_csie.png")
rostock = b64_logo("logo_rostock.png")

top_l, top_r = st.columns([3, 1.15])
with top_l:
    st.markdown(
        """
        <div class="masthead">
          <div class="eyebrow">Surrogate-assisted structural design · green transition</div>
          <h1>Low-carbon, <span class="accent">redundancy-aware</span> truss optimization</h1>
          <p>An interactive tool for the multi-objective design of truss structures,
          trading embodied carbon against structural robustness under a probabilistic
          reliability constraint. Select a structure, tune the parameters, and run the
          surrogate-assisted optimizer to obtain the Pareto front.</p>
          <div class="chips">
            <span class="chip carbon">objective&nbsp;1 · <b>minimize embodied carbon</b></span>
            <span class="chip green">objective&nbsp;2 · <b>maximize redundancy</b></span>
            <span class="chip">constraint · <b>R ≥ R<sub>target</sub></b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_r:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.image(f"data:image/png;base64,{csie}", use_container_width=True)
    st.markdown(
        """
        <div class="credit">
          <b>Dr. Jafar Jafari-Asl</b><br>
          <b>Prof. Dr. Panagiotis Spyridis</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(f"data:image/png;base64,{rostock}", use_container_width=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='seclabel'>1 · Structure</div>", unsafe_allow_html=True)
    bm_name = st.selectbox("Benchmark truss", list(BENCH.keys()), index=0,
                           label_visibility="collapsed")
    meta = BENCH[bm_name]
    st.caption(meta["note"])

    st.markdown("<div class='seclabel'>2 · Reliability</div>", unsafe_allow_html=True)
    target_R = st.slider("Target reliability  R_target", 0.90, 0.999, 0.99, 0.001,
                         format="%.3f")

    if meta["scale_loads"]:
        st.markdown("<div class='seclabel'>3 · Loading</div>", unsafe_allow_html=True)
        load_scale = st.slider("Load scale factor", 0.5, 1.5, 1.0, 0.05,
                               help="Multiplies the reference loads of the benchmark.")
    else:
        load_scale = 1.0

    st.markdown("<div class='seclabel'>4 · Embodied carbon factors</div>",
                unsafe_allow_html=True)
    st.caption("kgCO₂e per kg of steel (A1–A3)")
    cf_bof = st.number_input("BOF — basic oxygen furnace", 1.5, 3.0, 2.30, 0.05)
    cf_eaf = st.number_input("EAF — electric arc furnace (recycled)", 0.2, 1.5, 0.70, 0.05)
    mix_bof = st.slider("Mix: share of BOF steel", 0, 100, 60, 5,
                        help="The remainder is recycled EAF steel.")
    cf_mix = (mix_bof / 100) * cf_bof + (1 - mix_bof / 100) * cf_eaf
    route = st.radio("Production route for the front", ["MIX", "BOF", "EAF"],
                     horizontal=True, index=0)
    st.caption(f"Mix factor = {cf_mix:.2f} kgCO₂e/kg")

    st.markdown("<div class='seclabel'>5 · Search budget</div>", unsafe_allow_html=True)
    budget = st.select_slider("Effort", options=["Quick", "Balanced", "Thorough"],
                              value="Balanced")
    seed = st.number_input("Random seed", 0, 9999, 1, 1)

    run = st.button("Run optimization  →")

# budget -> surrogate parameters
BUDGET = {
    "Quick":    dict(n_init=16, n_iter=4,  batch=5),
    "Balanced": dict(n_init=24, n_iter=8,  batch=6),
    "Thorough": dict(n_init=30, n_iter=14, batch=6),
}


# ----------------------------------------------------------------------
# Optimizer wrapper
# ----------------------------------------------------------------------
def run_optimization(meta, target_R, load_scale, cf_bof, cf_eaf, cf_mix, route, bud, seed):
    # inject the chosen carbon factors
    cr.CARBON_FACTORS["BOF"] = cf_bof
    cr.CARBON_FACTORS["EAF"] = cf_eaf
    cr.CARBON_FACTORS["MIX"] = cf_mix

    prob = PA.ALL_PROBLEMS[meta["key"]]()
    if meta["scale_loads"] and load_scale != 1.0:
        prob.P_MEAN = prob.P_MEAN * load_scale

    rb = MO.RBROProblem(prob, route=route, target_R=target_R,
                        red_mode=meta["mode"], red_k=meta["k"])
    res = SO2.run_surrogate2(rb, n_init=bud["n_init"], n_iter=bud["n_iter"],
                             batch=bud["batch"], seed=int(seed))
    return res, rb


# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
if run:
    bud = BUDGET[budget]
    with st.spinner(f"Running the surrogate-assisted optimizer on the {meta['key']} "
                    f"truss · {budget.lower()} budget …"):
        t0 = time.time()
        res, rb = run_optimization(meta, target_R, load_scale, cf_bof, cf_eaf,
                                   cf_mix, route, bud, seed)
        elapsed = time.time() - t0

    F = res["F"]
    if len(F) == 0:
        st.error("No feasible non-dominated design was found. Try relaxing the "
                 "target reliability or increasing the load scale.")
    else:
        carbon = F[:, 0]
        rred = -F[:, 1]
        order = np.argsort(carbon)
        carbon, rred = carbon[order], rred[order]
        X = res["X"][order] if len(res["X"]) == len(F) else None

        # ---- metric cards ----
        st.markdown("<div class='seclabel'>Results</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metricwrap">
              <div class="metric"><div class="k">Pareto designs</div>
                   <div class="v">{len(F)}</div><div class="u">non-dominated</div></div>
              <div class="metric carbon"><div class="k">Carbon range</div>
                   <div class="v">{carbon.min():,.0f}<span class="u"> – {carbon.max():,.0f}</span></div>
                   <div class="u">kgCO₂e (A1–A3)</div></div>
              <div class="metric green"><div class="k">Redundancy range</div>
                   <div class="v">{rred.min():.2f}<span class="u"> – {rred.max():.2f}</span></div>
                   <div class="u">R_red index</div></div>
              <div class="metric"><div class="k">Exact FEM solves</div>
                   <div class="v">{rb.fem_calls:,}</div><div class="u">{elapsed:.0f}s · R²={res['gp_r2']:.2f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_plot, col_tab = st.columns([1.4, 1])

        with col_plot:
            st.markdown("<div class='seclabel'>Carbon vs redundancy Pareto front</div>",
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=carbon, y=rred, mode="lines+markers",
                line=dict(color="#1b2a3a", width=2),
                marker=dict(size=9, color=rred, colorscale="YlOrBr_r",
                            line=dict(width=1, color="#1b2a3a"),
                            showscale=False),
                hovertemplate="Carbon: %{x:,.0f} kgCO₂e<br>R_red: %{y:.3f}<extra></extra>",
                name="Pareto front",
            ))
            fig.update_layout(
                template="simple_white",
                height=440,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Embodied carbon  (kgCO₂e, A1–A3)",
                yaxis_title="Redundancy index  R_red",
                font=dict(family="Archivo, sans-serif", size=13, color="#0f1822"),
                plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each point is a non-dominated design. Moving up the front buys "
                       "redundancy at the cost of embodied carbon. Hover for values.")

        with col_tab:
            st.markdown("<div class='seclabel'>Design table</div>", unsafe_allow_html=True)
            tab = {"Carbon (kgCO₂e)": np.round(carbon, 0).astype(int),
                   "R_red": np.round(rred, 3)}
            if X is not None:
                for j in range(X.shape[1]):
                    tab[f"A{j+1}"] = np.round(X[:, j], 4)
            df = pd.DataFrame(tab)
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode(),
                               file_name=f"pareto_{meta['key']}.csv", mime="text/csv")

        st.info(f"The surrogate reached a leave-one-out R² of {res['gp_r2']:.2f} "
                f"using only {rb.fem_calls:,} exact finite-element solves. The cross "
                "sectional areas A1, A2, … are the design groups of the selected truss.",
                icon="◭")

else:
    # idle state — guidance, not decoration
    st.markdown("<div class='seclabel'>Getting started</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Pick a structure**\n\nChoose one of four trusses in the sidebar, "
                "from the fast 10-bar truss to the real 137-bar bridge.")
    c2.markdown("**2 · Set the parameters**\n\nAdjust the target reliability, the carbon "
                "factors of the steel routes, and the search budget.")
    c3.markdown("**3 · Run and read the front**\n\nThe optimizer returns the carbon-vs-"
                "redundancy Pareto front and the design table, ready to download.")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.caption("The full-evaluation genetic algorithm used to validate the method in the "
               "paper is omitted here; this tool runs the validated surrogate-assisted "
               "optimizer alone so that results return in seconds to minutes.")
