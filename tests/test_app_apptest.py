"""
tests/test_app_apptest.py - drive the real Streamlit app with Streamlit's own
AppTest harness.

Why this exists on top of tests/test_app.py
-------------------------------------------
The headless sweep in test_app.py calls the library functions directly, so it
never executes `app.main()`. Two real crashes got past it - a cache key passed
where a hyper-parameter dict belonged, and `model.privacy_note` read as a bare
attribute on the models that do not define it. Both were only visible once the
actual app script ran.

AppTest runs the genuine script inside a real ScriptRunner, so changing a widget
triggers a genuine re-run and any exception surfaces in `at.exception`. That is
the closest thing to an examiner clicking through the demo, and unlike a browser
it can assert on every generator rather than just the default one.

Widget indices, verified against the running app rather than assumed:
    selectbox[0] Dataset | selectbox[1] Generator | selectbox[2] class filter
    slider[0] n_patients | slider[1] diversity
    radio[0] training budget | number_input[0] sampling seed
    toggle[0] "Retrain live in the browser"  (off = published frozen cohort)
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from streamlit.testing.v1 import AppTest

APP = os.path.join(HERE, "app.py")
TIMEOUT = 900
# The Generator selectbox is keyed by its human-readable label, not the model
# key. Deriving the mapping from app.MODEL_LABEL instead of restating it means a
# rename in the UI cannot silently break this suite - and it did exactly that.
import app as _app  # noqa: E402

GEN_LABEL = {k: v for k, v in _app.MODEL_LABEL.items()}
CLASS_MATCH, CLASS_DISEASED, CLASS_HEALTHY = _app.CLASS_OPTIONS


def first_error(at: AppTest) -> str:
    if not at.exception:
        return ""
    return str(at.exception[0].value).splitlines()[0][:220]


def renders_headline(at: AppTest) -> bool:
    """The app's headline is an HTML hero band (ui.components.hero), not
    st.title, so AppTest does not surface it in `at.title`.

    Checking `at.title` here would pass vacuously on a blank page; instead we
    require the actual title text to be present in a rendered markdown block.
    That is the same guarantee - the script got as far as the header - and it
    survives the presentation layer being restyled.
    """
    for md in at.markdown:
        if "Synthetic Medical Data Generator" in str(getattr(md, "value", "")):
            return True
    return False


def run(label: str, mutate) -> bool:
    """Boot the app, apply `mutate`, re-run, and require a real rendered result.

    'No exception' is not enough - a silent early `return` would also pass that.
    So we additionally require the title, the four tabs, and at least one metric
    to be present after the re-run.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    err = first_error(at)
    if err:
        print(f"  FAIL {label} (initial run): {err}")
        return False
    try:
        mutate(at)
    except Exception as exc:                        # noqa: BLE001
        print(f"  FAIL {label} (widget setup): {type(exc).__name__}: {exc}")
        return False
    at.run()
    err = first_error(at)
    if err:
        print(f"  FAIL {label}: {err}")
        return False
    if not renders_headline(at):
        print(f"  FAIL {label}: headline hero did not render")
        return False
    if len(at.tabs) != 4:
        print(f"  FAIL {label}: expected 4 tabs, got {len(at.tabs)}")
        return False
    if not at.metric:
        print(f"  FAIL {label}: no metrics rendered (silent early return?)")
        return False
    # `main()` wraps the whole train+sample step in try/except and reports
    # failures through st.error, so a clean run must have zero st.error blocks.
    # This is the assertion that actually catches a broken model path.
    if at.error:
        print(f"  FAIL {label}: the app reported an error: "
              f"{[str(e.value)[:120] for e in at.error]}")
        return False
    # Streamlit 1.30's AppTest exposes no `download_button` accessor, so the
    # CSV export cannot be asserted here; it was verified in a real browser
    # (tests/test_app.py layer 3 and the manual Chrome pass), where the
    # "Download synthetic cohort as CSV" button is present in the DOM.
    # Instead assert the sidebar caption that documents the same control panel
    # rendered, which proves the UI got past widget construction.
    captions = " ".join(str(c.value) for c in at.caption)
    if "diversity" not in captions.lower():
        print(f"  FAIL {label}: the diversity caption did not render")
        return False
    print(f"  ok   {label:<54s} metrics={len(at.metric)}")
    return True


def pick(dataset=None, gen=None, cls=None, n=None, diversity=None,
         budget=None, seed=None, epsilon=None, live=None):
    """Build a widget-mutation closure for one scenario."""
    def apply(at: AppTest) -> None:
        if live is not None and at.toggle:
            at.toggle[0].set_value(live)
        if dataset is not None:
            at.selectbox[0].set_value(dataset)
        if gen is not None:
            at.selectbox[1].set_value(GEN_LABEL[gen])
        if epsilon is not None:
            # the epsilon select_slider only exists once DP-VAE is selected
            if at.select_slider:
                at.select_slider[0].set_value(epsilon)
        if cls is not None:
            at.selectbox[2].set_value(cls)
        if n is not None:
            at.slider[0].set_value(n)
        if diversity is not None:
            at.slider[1].set_value(diversity)
        if budget is not None:
            at.radio[0].set_value(budget)
        if seed is not None:
            at.number_input[0].set_value(seed)
    return apply


def main(argv=None) -> int:
    """Run the scenario groups named on the command line.

    `--groups frozen,extreme` finishes in a couple of minutes because frozen
    mode loads stored cohorts instead of fitting torch models. The default is
    every group, which is the real gate before calling the app verified.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", nargs="*",
                    default=["live", "fast", "dp", "frozen", "extreme"],
                    choices=["live", "fast", "dp", "frozen", "extreme", "all"],
                    help="which scenario groups to run (default: all)")
    groups = set(ap.parse_args(argv).groups)
    if "all" in groups:
        groups = {"live", "fast", "dp", "frozen", "extreme"}
    fails = 0

    if "live" in groups:
        print("[1/5] every generator on Cleveland, standard budget")
        for g in GEN_LABEL:
            fails += 0 if run(f"heart generator={g}",
                              pick(gen=g, budget="Standard")) else 1

    if "fast" in groups:
        print("[2/5] every generator on Pima, fast budget")
        for g in GEN_LABEL:
            fails += 0 if run(f"pima generator={g}",
                              pick(dataset="pima", gen=g, budget="Fast")) else 1

    if "dp" in groups:
        print("[3/5] the DP epsilon control at both extremes")
        for eps in (1.0, 30.0):
            fails += 0 if run(f"dp_vae epsilon={eps:g}",
                              pick(gen="dp_vae", epsilon=eps,
                                   budget="Fast")) else 1

    if "frozen" in groups:
        print("[4/5] frozen published cohorts - no training, both datasets")
        for g in GEN_LABEL:
            for d in ("heart", "pima"):
                fails += 0 if run(f"frozen {d} generator={g}",
                                  pick(dataset=d, gen=g, live=False)) else 1

    if "extreme" in groups:
        print("[5/5] extreme control combinations")
        cases = {
            "10 patients, diversity 0.0, diseased-only":
                pick(n=10, diversity=0.0, cls=CLASS_DISEASED, gen="vae"),
            "5000 patients, diversity 1.0, healthy-only":
                pick(n=5000, diversity=1.0, cls=CLASS_HEALTHY, gen="vae"),
            "seed 777, standard budget, max diversity":
                pick(seed=777, budget="Standard", diversity=1.0, gen="wgan_gp"),
            "5000 patients forced into a rare class":
                pick(n=5000, cls=CLASS_DISEASED, gen="naive"),
            "minimum patients, minimum diversity":
                pick(n=10, diversity=0.0, gen="ctgan"),
            "frozen, 5000 rows forced into the rare class":
                pick(live=False, n=5000, cls=CLASS_DISEASED, gen="naive"),
        }
        for label, mutate in cases.items():
            fails += 0 if run(label, mutate) else 1

    print()
    print("TOTAL FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
