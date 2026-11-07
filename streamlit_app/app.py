"""
Dashboard Prediksi Volatilitas Saham Energi LQ45
================================================
Tesis Magister: Ilda Nurida (6032251068), SIMT ITS, 2026.

Dua halaman:
  • 💼 Investor          — radar risiko H+1, simulator skenario berita, analisis historis, berita terbaru
  • 🎓 Hasil penelitian  — performa model, uji Diebold–Mariano, SHAP, ablasi sentimen

Semua data dan model dibaca LANGSUNG dari repositori GitHub:
    https://github.com/ilsy16-inv/sentiment-driven-volatility-prediction

Menjalankan secara lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st

from common import BRANCH, CSS, REPO

st.set_page_config(page_title="Volatilitas Saham Energi LQ45", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Sumber data")
    st.session_state["repo"] = st.text_input("Repositori GitHub", st.session_state.get("repo", REPO))
    st.session_state["branch"] = st.text_input("Branch", st.session_state.get("branch", BRANCH))
    st.caption(f"[Buka repositori ↗](https://github.com/{st.session_state['repo']}/tree/{st.session_state['branch']})")
    if st.button("🔄 Muat ulang data", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption("Data di-cache 1 jam. Klik tombol di atas setelah push hasil notebook terbaru.")
    st.divider()
    st.caption("Tesis Magister — Ilda Nurida (6032251068)\nSIMT ITS · 2026")

halaman = st.navigation(
    [st.Page("investor.py", title="Investor", icon="💼", default=True),
     st.Page("riset.py", title="Hasil penelitian", icon="🎓")],
    position="top",
)
halaman.run()
