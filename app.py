import streamlit as st
st.title("Fabric Quality Classification")
st.write("Upload a fabric image to classify its quality.")

api_key = st.secrets["ROBOFLOW_API_KEY"]
