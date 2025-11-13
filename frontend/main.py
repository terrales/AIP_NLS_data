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
def topic_page():
    st.title("What are the common topics?")
    languages = ['All', 'English', 'French', 'German', 'Latin', 'Undetermined', 'Italian', 'Russian', 'Spanish', 'Non-linguistic', 'Welsh']
    language2code = {
        'English': 'english',
        'French': 'french',
        'German': 'german',
        'Latin': 'latin',
        'Undetermined': 'und',
        'Italian': 'ita',
        'Russian': 'rus',
        'Spanish': 'spa',
        'Non-linguistic': 'zxx',
        'Welsh': 'wel',
    }
    all_lang_url = 'http://3.107.231.237/common_topic_100.html'
    lang_url = "http://3.107.231.237/{code}_top100_topic.html"

    lang = st.selectbox('Select a language', languages)
    if lang == 'All':
        url = all_lang_url
    else:
        url = lang_url.format(code=language2code[lang])

    st.link_button(label='Click here to check topics', url=url, type='primary')
    st.markdown(f"![Foo](https://raw.githubusercontent.com/terrales/AIP_NLS_data/refs/heads/main/frontend/images/topics.png)")

# Main function to control app flow
def main():

    # Optional Responsible NLP logo at top of sidebar
    st.sidebar.markdown("""
        <a class="RR-NLP-logo" href="https://www.responsiblenlp.org/">
            <img src="https://fau-res.cloudinary.com/image/upload/common/bespoke-pages/custom-pages/bpid11765/img2025103539.jpg" alt="CDT website (opens in new tab)" height="40"/>
        </a>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Motivation", "Background", "Date", "Type", "Language", "Creator", "Subject", 'What are the common topics?'])

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
    elif page == 'What are the common topics?':
        topic_page()

if __name__ == "__main__":
    main()


# Add the Design System
ds.init()

