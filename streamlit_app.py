from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

from final_movie_analysis.functions import do_analysis_specific


def _sample_data() -> pd.DataFrame:
    """Small placeholder dataset for rapid UI feedback."""
    return pd.read_csv("movie_data.csv")


def _run_with_capture(func) -> str:
    """Capture stdout from placeholder pipelines so Streamlit can display it."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func()
    return buffer.getvalue().strip()


def main() -> None:
    st.set_page_config(page_title="STAT 386 Final Project-Movie Analysis", layout="wide")
    st.title("STAT 386 Final Project-Movie Analysis")

    with st.sidebar:
        st.header("Controls")
        st.info("Note: Data cleaning doesn't include data gathering")
        show_cleaning = st.checkbox("Preview cleaning pipeline output")
        show_analysis = st.checkbox("Preview analysis pipeline output")

    try:
        df = _sample_data()
    except:
        st.info("movie_data.csv not uploaded. Widgets will not work")

    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)

    if show_cleaning:
        st.subheader("Cleaning Pipeline Output")
        st.subheader("Before")
        df2 = pd.read_csv("dirty_data.csv")
        df2
        st.subheader("After")
        df
        st.subheader("After being prepped for ML Analysis (Note: The Total Box Office Revenue Column is actually log(Total Box Office Revenue) due to skewness of data)")
        df3 = pd.read_csv("ml_movie_data.csv")
        df3

    if show_analysis:
        st.subheader("Analysis Pipeline Output")
        analysis_output = _run_with_capture(do_analysis_specific)
        st.code(analysis_output or "Error: No output")
        #st.caption("Swap this stub with charts, metrics, or model diagnostics from your project.")

    #st.info("Next steps: customize the sidebar controls, drop in Streamlit charts (st.bar_chart, st.map, etc.), and layer in explanations so stakeholders can self-serve results.")


if __name__ == "__main__":
    main()
    #Run using uv run streamlit run streamlit_app.py