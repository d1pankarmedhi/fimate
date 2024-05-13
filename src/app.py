import pandas as pd
import streamlit as st
import yfinance as yf

from backtester.sma import SMABacktester


def get_data(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def main():
    st.markdown("<h1 style='text-align: center;'>FiMate</h1>", unsafe_allow_html=True)

    ## state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "strategy" not in st.session_state:
        st.session_state.strategy = ""
    if "ticker" not in st.session_state:
        st.session_state.ticker = ""
    if "start_date" not in st.session_state:
        st.session_state.start_date = ""
    if "end_date" not in st.session_state:
        st.session_state.end_date = ""

    ## sidebar
    with st.sidebar:
        st.session_state.ticker = st.text_input(
            label="Retrieve data", placeholder="Enter Ticker..."
        )
        st.session_state.start_date = st.text_input(
            label="Start date", placeholder="2020-01-01"
        )
        st.session_state.end_date = st.text_input(
            label="End date", placeholder="2024-01-01"
        )

        fetch_btn = st.button(label="Fetch data")

        # show data
        if (
            fetch_btn
            and st.session_state.ticker
            and st.session_state.start_date
            and st.session_state.end_date
        ):
            st.session_state.data = get_data(
                st.session_state.ticker,
                st.session_state.start_date,
                st.session_state.end_date,
            )
        elif fetch_btn and st.session_state.ticker:
            st.session_state.data = get_data(
                ticker=st.session_state.ticker,
            )

    ## Tabs
    tabs = st.tabs(
        [
            "Analysis/Charting",
            "Backtest",
            "Forcast",
            "Sentiment Analysis",
        ]
    )

    ## data analysis and charting
    with tabs[0]:
        if isinstance(st.session_state.data, pd.DataFrame):
            st.write(st.session_state.data)

    ## backtesting
    with tabs[1]:
        strategies = ["NONE", "Simple Moving Average (SMA)", "Momentum Backtesting"]

        # parameters
        st.session_state.strategy = st.selectbox(
            label="Select strategy",
            options=strategies,
        )
        if st.session_state.strategy == strategies[1]:
            cols = st.columns(2)
            with cols[0]:
                # get sma1 and sma2 values
                SMA1 = st.number_input(
                    label="SMA1", value=None, placeholder="Type a number"
                )
            with cols[1]:
                SMA2 = st.number_input(
                    label="SMA2", value=None, placeholder="Type a number"
                )
            if (
                st.session_state.data is not None
                and SMA1 is not None
                and SMA2 is not None
            ):
                sma_backtester = SMABacktester(
                    ticker=st.session_state.ticker,
                    SMA1=int(SMA1),
                    SMA2=int(SMA2),
                    start=st.session_state.start_date,
                    end=st.session_state.end_date,
                )
                data = st.session_state.data
                sma_backtester.prepare_data(data)
                st.write(sma_backtester.run_strategy())

                # plots
                cols = st.columns(3)
                plots = [sma_backtester.plot_sma(), sma_backtester.plot_returns()]
                for idx, plot in enumerate(plots):
                    with cols[idx % len(cols)]:
                        st.pyplot(plot)

                if plots:
                    st.markdown("**Optimze SMA parameters**")
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("SMA1 range")
                        sma1_low = st.number_input(label="set low range for SMA1")
                        sma1_high = st.number_input(label="set high range for SMA1")
                    with cols[1]:
                        st.markdown("SMA2 range")
                        sma2_low = st.number_input(label="set low range for SMA2")
                        sma2_high = st.number_input(label="set high range for SMA2")
                    optimize_btn = st.button(label="Optimize")
                    if optimize_btn:
                        smas, perf = sma_backtester.optimize_parameters(
                            SMA1_range=(sma1_low, sma1_high, 2),
                            SMA2_range=(sma2_low, sma2_high, 2),
                        )
                        st.write(f"SMA1: {smas[0]}, SMA2: {smas[1]}")

                        sma_backtester.set_parameters(int(smas[0]), int(smas[1]))
                        st.pyplot(sma_backtester.plot_returns())

        elif st.session_state.strategy == strategies[2]:
            pass


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="fimate",
        initial_sidebar_state="collapsed",
        page_icon="🏛",
    )
    main()
