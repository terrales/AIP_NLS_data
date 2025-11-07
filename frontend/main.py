import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
import matplotlib.pyplot as plt
from app import design_system as ds


# Home Page
def home():
    st.title("Home Page")
    st.write("Welcome to the home page!")

# Page to display sample data
def data_page():
    st.title("Sample Data")
    data = {
        "Column 1": [1, 2, 3, 4],
        "Column 2": [10, 20, 30, 40],
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

# Page to display sample graph
def graph_page():
    st.title("Sample Graph")
    data = {
        "X": [1, 2, 3, 4],
        "Y": [10, 20, 30, 40],
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df["X"], df["Y"], marker="o")
    plt.title("Sample Line Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    st.pyplot(plt)

# an exmaple page to share the bubble graph
def bubble_page():
    st.title("Bubble Graph Example")
    url = "http://3.107.231.237/bubble.html"
    st.link_button(label='Click here to interact with bubbles', url=url, type='primary')
    st.markdown(f"![Foo](https://raw.githubusercontent.com/terrales/AIP_NLS_data/b41f08459c99d5e3374e9dae4392f886cb53b9a8/frontend/images/bubble.png)")


# Main function to control app flow
def main():

    # Optional Responsible NLP logo at top of sidebar
    st.sidebar.markdown("""
        <a class="RR-NLP-logo" href="https://www.responsiblenlp.org/">
            <img src="https://fau-res.cloudinary.com/image/upload/common/bespoke-pages/custom-pages/bpid11765/img2025103539.jpg" alt="CDT website (opens in new tab)" height="40"/>
        </a>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Motivation", "Background", "Date", "Type", "Language", "Creator", "Subject", 'Bubble Example'])

    if page == "Home":
        home()
    elif page == "Motivation":
        data_page()
    elif page == "Background":
        graph_page()
    elif page == "Date":
        graph_page()
    elif page == "Type":
        graph_page()
    elif page == "Language":
        graph_page()
    elif page == "Creator":
        graph_page()
    elif page == "Subject":
        graph_page()
    elif page == 'Bubble Example':
        bubble_page()

if __name__ == "__main__":
    main()


# Add the Design System
ds.init()

