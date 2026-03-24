import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from src.RiskEngineWrapper import RiskEngineWrapper
from src.Utils import verbose_frequency_horizon_converter, verbose_lookback_converter, verbose_benchmark_ticker

######################################################################
#                           Session states                           #
######################################################################

if st.session_state.get("risk_engine_wrapper") is None:
    st.session_state.risk_engine_wrapper = RiskEngineWrapper()

######################################################################
#                             Callbacks                              #
######################################################################

def init_risk_engine_wrapper():
    st.session_state.risk_engine_wrapper.init_risk_engine(st.session_state.ptf_file, verbose_benchmark_ticker[st.session_state.benchmark_ticker])

def perform_computations():
    st.session_state.risk_engine_wrapper.update_positions(st.session_state.new_positions)

def update_lookback_size():
    st.session_state.risk_engine_wrapper.perform_computations(window_size=verbose_lookback_converter[st.session_state.lookback])

def updatable_var_cvar_change():
    st.session_state.risk_engine_wrapper.update_var_cvar_horizon(verbose_frequency_horizon_converter[st.session_state.horizon], st.session_state.confidence)

def update_benchmark():
    st.session_state.risk_engine_wrapper.update_benchmark(verbose_benchmark_ticker[st.session_state.benchmark_ticker])

#######################################################################
#                             Main app                                #
#######################################################################

st.set_page_config(page_title="PortStats", layout="wide")

css = '''
<style>
section.stMain > div.stMainBlockContainer {
    padding-bottom: 5px;
    padding-top: 0rem;
}
header.stAppHeader {
    min-height: 1rem;
    height: 1rem;
}
div.stVerticalBlock {
    calc(-1px + 0.5rem);
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

if st.session_state.risk_engine_wrapper.initial_computations_done == False:
    st.title("PortStats")

    st.selectbox(
        "Choose your benchmark index",
        verbose_benchmark_ticker.keys(), 
        key="benchmark_ticker",
        index=0
    )
    st.subheader("Upload your portfolio data")
    st.file_uploader("Choose a CSV file", key="ptf_file", type="csv", on_change=init_risk_engine_wrapper)        
        
else:
    # Initializing the data
    rew = st.session_state.risk_engine_wrapper

    # Layout of the page

    # Sidebar
    with st.sidebar:
        st.header("PortStats")
        sidebar_params = st.container(
            border=True,
            height="stretch",
            vertical_alignment="center"
        )
        with sidebar_params:
            st.selectbox(
                    "Lookback size:", 
                    verbose_lookback_converter.keys(), 
                    key="lookback",
                    index=1,
                    on_change=update_lookback_size, 
                )
            st.selectbox(
                "Choose your benchmark index",
                verbose_benchmark_ticker.keys(), 
                key="benchmark_ticker",
                index=0,
                on_change=update_benchmark()
            )

        sidebar_ptf = st.container(
            border=True,
            height="stretch",
            vertical_alignment="center"
        )
        with sidebar_ptf:
            st.write("Current portfolio")
            st.data_editor(rew.positions, key="new_positions", hide_index=True, num_rows="dynamic")
            st.button(label="Refresh", on_click=perform_computations)
    
    # Managing the tabs of the page
    overview_tab, risk_tab = st.tabs(["Overview", "Risk"])
    
    # Overview tab
    with overview_tab:
        left_col, right_col = st.columns([3, 1])
        
        # Left column
        with left_col:
            
            # Left column, upper cell -> Portfolio prices chart
            top_left_cell = st.container(
                border = True,
                height = "stretch",
                vertical_alignment="center"
            )
            with top_left_cell:
                fig_prices = px.line(rew.portfolio_values, x="Date", y=["Portfolio value ($)", "Benchmark value ($)"])
                fig_prices.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=None,
                        title="Portfolio evolution"
                    )
                fig_prices.update_yaxes(
                    title_text="Value ($)",
                    secondary_y=False,
                )
                st.plotly_chart(fig_prices, height=350, width="stretch")
            
            # Left column, lower cell -> weights, geographical and sectorial exposures pie graphs
            low_left_cell = st.container(
                border = True,
                height = "stretch",
                vertical_alignment="center"
            )

            with low_left_cell:
                lower_col_left, lower_col_center, lower_col_right = st.columns(3)

                # Weights pie
                with lower_col_left:
                    fig_weights = px.pie(rew.weights, values="Weights", names='Ticker', title="Weights")
                    fig_weights.update_traces(textposition='inside', textinfo='percent+label')
                    fig_weights.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=None
                    )
                    st.plotly_chart(fig_weights, height=180, width="stretch")

                # Sectorial exposure pie
                with lower_col_center:
                    fig_sectors = px.pie(rew.sectorial_exposure, values="Value", names='Sector', title="Sectorial exposure")
                    fig_sectors.update_traces(textposition='inside', textinfo='percent+label')
                    fig_sectors.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=None
                    )
                    st.plotly_chart(fig_sectors, height=180, width="stretch")

                # geographical exposure pie
                with lower_col_right:
                    fig_countries = px.pie(rew.geographical_exposure, values="Value", names='Country', title="Geographical exposure")
                    fig_countries.update_traces(textposition='inside', textinfo='percent+label')
                    fig_countries.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=None
                    )
                    st.plotly_chart(fig_countries, height=180, width="stretch")



        # Right column
        with right_col:

            # Right column, upper cell -> Performance and global risk metrics
            top_right_cell = st.container(
                border = True,
                height = "stretch",
                vertical_alignment="center"
            )
            with top_right_cell:
                st.write("Ptf. Overview")
                st.dataframe(rew.ptf_metrics, hide_index=True)

            # Right column, bottom cell -> VaR and cVaR for 1 day
            bottom_right_cell = st.container(
                border=True,
                height="stretch",
                vertical_alignment="center"
            )
            with bottom_right_cell:
                st.write("Risk metrics 1-Day")
                st.dataframe(rew.risk_metrics_overview)

    # Risk tab
    with risk_tab:
        left_col_risk, right_col_risk = st.columns([3, 1])

        with left_col_risk:

            # Drawdown and ptf. evolution graph
            chart_cell = st.container(
                    border = True,
                    height = "stretch",
                    vertical_alignment="center"
                )
            with chart_cell:

                fig = make_subplots(
                    specs=[[{"secondary_y": True}]],
                    x_title="Date",
                )
                fig.add_trace(
                    go.Scatter(
                        x=rew.portfolio_values["Date"],
                        y=rew.portfolio_values["Portfolio value ($)"],
                        name="Portfolio value ($)",
                        mode="lines",
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=rew.drawdowns["Date"],
                        y=rew.drawdowns["Drawdown (%)"],
                        name="Drawdown",
                        mode="lines",
                        line=dict(color= "rgba(255, 127, 14, 1)"),
                        fill="tozeroy",
                        fillcolor="rgba(255, 127, 14, 0.2)",
                        yaxis="y2",
                    ),
                    secondary_y=True
                )
                fig.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=None,
                        title="Portfolio evolution & Drawdowns"
                    )
                fig.update_yaxes(
                    title_text="Value ($)",
                    secondary_y=False,
                )
                fig.update_yaxes(
                    title_text="Drawdown (%)",
                    secondary_y=True,
                )
                st.plotly_chart(fig, height=350, width="stretch")
            
            # Risk contributions
            risk_contributions_cell = st.container(
                    border = True,
                    height = "stretch",
                    vertical_alignment="center"
                )
            with risk_contributions_cell:
                risk_contrib_fig = px.bar(
                    rew.risk_contribution, 
                    x="Value", 
                    y="Type", 
                    color="Ticker", 
                    orientation='h'
                )
                risk_contrib_fig.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=None,
                        title="Risk contributions"
                    )
                st.plotly_chart(risk_contrib_fig, height=180, width="stretch")

        with right_col_risk:

            detailed_risk_cell = st.container(
                    border = True,
                    height = "stretch",
                    vertical_alignment="center"
                )
            with detailed_risk_cell:

                st.write("General risk metrics")
                st.dataframe(rew.risk_metrics, hide_index=True)

                st.write("VaR/cVaR")

                select_columns = st.columns(2)
                select_columns[0].selectbox(
                    "Horizon:", 
                    verbose_frequency_horizon_converter.keys(), 
                    key="horizon",
                    index=0,
                    on_change=updatable_var_cvar_change, 
                )
                select_columns[1].selectbox(
                    "Confidence level:", 
                    [0.95, 0.975, 0.99], 
                    key="confidence",
                    index=0,
                    on_change=updatable_var_cvar_change, 
                )
                st.dataframe(rew.updatable_var_cvar)

                