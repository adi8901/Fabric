import streamlit as st
st.title("Fabric Quality Classification")
st.write("Upload a fabric image to classify its quality.")

uploaded_file = st.file_uploader(
    "Choose a fabric image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Fabric Image",
        use_container_width=True
    )

api_key = st.secrets["ROBOFLOW_API_KEY"]
