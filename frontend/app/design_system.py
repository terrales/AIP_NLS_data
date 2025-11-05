from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

def init():
    base_dir = Path(__file__).resolve().parent.parent
    static_dir = base_dir / "static"

    css_path = static_dir / "styles" / "nls.css"
    js_path = static_dir / "scripts" / "nls.js"

    if css_path.exists():
        with open(css_path) as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at {css_path}")

    if js_path.exists():
        with open(js_path) as f:
            js_content = f.read()
        components.html(f"<script>{js_content}</script>", height=0)
    else:
        st.warning(f"JS file not found at {js_path}")
