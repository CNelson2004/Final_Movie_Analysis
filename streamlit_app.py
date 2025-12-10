from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

from final_movie_analysis.functions import describe_revenue, earnings_correlation, season_significance, rating_significance, production_method_significance, genre_significance, do_ml_analysis_numbers, revenue_findings, factors_findings, ml_analysis_findings
from streamlit_graphs import graph_revenue, graph_revenue_by_year, graph_revenue_and_profit, season_earnings, genre_earnings, production_method_earnings, ratings_earnings, ml_graph_once, ml_graph_before, ml_graph_after, answer_question


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
        show_cleaning = st.checkbox("Preview gathering and cleaning pipeline output")
        show_revenue_graphs = st.checkbox("Preview graphs related to revenue")
        show_revenue_statistics = st.checkbox("Preview statistical information of revenue")
        show_feature_graphs = st.checkbox("Preview graphs related to features")
        show_feature_correlation = st.checkbox("Preview correlation values between features")
        show_feature_significance = st.checkbox("Preview statistical significance relating to features")
        show_feature_impact_graphs = st.checkbox("Preview feature impact on Total Revenue through graphs")
        show_feature_impact_numbers = st.checkbox("Preview feature impact on Total Revenue through numbers")
        show_revenue_findings = st.checkbox('Preview our analysis on revenue information')
        show_feature_findings = st.checkbox('Preview our analysis on feature information')
        show_impact_findings = st.checkbox('Preview our analysis on impact information')
        show_question_answer = st.checkbox("Preview our research question and answer based on this analysis")

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


    if show_question_answer:
        st.subheader("Research Question and Answer")
        analysis_output = _run_with_capture(answer_question)
        st.code(analysis_output or "Error: No output")
        st.info("For reasoning behind this analysis please check the boxes: 'Preview our analysis on feature information' and 'Preview our analysis on impact information'")


    if show_revenue_graphs:
        st.subheader("Graphs related to Revenue")
        fig1,fig1_5 = graph_revenue()
        if fig1:
            st.pyplot(fig1)
            st.pyplot(fig1_5)
        else:
            st.write("Error: No output")
        fig2 = graph_revenue_by_year()
        if fig2:
            st.pyplot(fig2)
        else:
            st.write("Error: No output")
        fig3 = graph_revenue_and_profit()
        if fig3:
            st.pyplot(fig3)
        else:
            st.write("Error: No output")


    if show_revenue_statistics:
        st.subheader("Statistical Revenue Description")
        analysis_output = _run_with_capture(describe_revenue)
        st.code(analysis_output or "Error: No output")


    if show_feature_graphs:
        st.subheader("Graphs related to Features")
        fig1 = season_earnings()
        if fig1:
            st.pyplot(fig1)
        else:
            st.write("Error: No output")
        fig2 = ratings_earnings()
        if fig2:
            st.pyplot(fig2)
        else:
            st.write("Error: No output")
        fig3 = genre_earnings()
        if fig3:
            st.pyplot(fig3)
        else:
            st.write("Error: No output")
        fig4 = production_method_earnings()
        if fig4:
            st.pyplot(fig4)
        else:
            st.write("Error: No output")


    if show_feature_correlation:
        st.subheader("Feature Correlation")
        analysis_output = _run_with_capture(earnings_correlation)
        st.code(analysis_output or "Error: No output")


    if show_feature_significance:
        st.subheader("Feature Significance")
        analysis_output = _run_with_capture(season_significance)
        st.code(analysis_output or "Error: No output")
        analysis_output = _run_with_capture(rating_significance)
        st.code(analysis_output or "Error: No output")
        analysis_output = _run_with_capture(genre_significance)
        st.code(analysis_output or "Error: No output")
        analysis_output = _run_with_capture(production_method_significance)
        st.code(analysis_output or "Error: No output")


    if show_feature_impact_graphs:
        st.subheader("Graphs related to Impact of Features on Total Box Office Revenue")
        fig1 = ml_graph_before()
        if fig1:
            st.pyplot(fig1)
        else:
            st.write("Error: No output")
        fig2 = ml_graph_once()
        if fig2:
            st.pyplot(fig2)
        else:
            st.write("Error: No output")
        fig3 = ml_graph_after()
        if fig3:
            st.pyplot(fig3)
        else:
            st.write("Error: No output")


    if show_feature_impact_numbers:
        st.subheader("Numbers related to Impact of Features on Total Box Office Revenue")
        analysis_output = _run_with_capture(do_ml_analysis_numbers)
        st.code(analysis_output or "Error: No output")


    if show_revenue_findings:
        st.subheader("Revenue Interpretation and Findings")
        analysis_output = _run_with_capture(revenue_findings)
        st.code(analysis_output or "Error: No output")
        st.info("For reasoning behind these interpretations please check the boxes: 'Preview graphs related to revenue'")


    if show_feature_findings:
        st.subheader("Feature Interpretation and Findings")
        analysis_output = _run_with_capture(factors_findings)
        st.code(analysis_output or "Error: No output")
        st.info("For reasoning behind these interpretations please check the boxes: 'Preview graphs related to features' and 'Preview correlation values between features'")


    if show_impact_findings:
        st.subheader("Feature Impact Interpretation and Findings")
        analysis_output = _run_with_capture(ml_analysis_findings)
        st.code(analysis_output or "Error: No output")
        st.info("For reasoning behind these interpretations please check the boxes: 'Preview feature impact on Total Revenue through graphs'")


if __name__ == "__main__":
    main()
    #Run using uv run streamlit run streamlit_app.py