"""Derivatives Market Trader dashboard.

    streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(page_title="Derivatives Market Trader", layout="wide")

nav = st.navigation({
    "Markets": [
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/futures.py", title="EEX futures", url_path="futures"),
        st.Page("views/spot_prices.py", title="Day-ahead prices", url_path="spot-prices"),
    ],
    "Drivers": [
        st.Page("views/fundamentals.py", title="Fundamentals", url_path="fundamentals"),
        st.Page("views/weather.py", title="Weather", url_path="weather"),
    ],
    "Models": [
        st.Page("views/model.py", title="German base month", url_path="model"),
    ],
})
nav.run()
