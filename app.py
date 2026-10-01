import tempfile
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import streamlit as st

api_key = st.secrets["ROBOFLOW_API_KEY"]

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
if uploaded_file is not None:
    if st.button("Classify Fabric"):
        try:
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key
            ).configure(
                InferenceConfiguration(
                    api_key_transport="header"
                )
            )

            with tempfile.NamedTemporaryFile(
                suffix=".jpg"
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file.flush()

                with st.spinner("Analyzing fabric..."):
                    result = client.run_workflow(
                        workspace_name="sanya-mohanty",
                        workflow_id=(
                            "classification-of-fabric-quality-vmy-first-project-mdcnu-1-resnet18-t1-logic"
                        ),
                        images={"image": temp_file.name},
                        use_cache=True
                    )

            try:
    # Your existing Roboflow client code
    # Your existing image upload and workflow call

    # Extract the prediction from the Roboflow response
    data = result[0] if isinstance(result, list) else result

    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        data = data["predictions"]

    label = data.get("top", "Unknown")
    confidence = float(data.get("confidence", 0))

    st.markdown("---")
    st.subheader("🔍 Classification Result")
    st.success(f"Predicted Fabric Defect: **{label.title()}**")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Class", label.title())

    with col2:
        st.metric("Confidence Score", f"{confidence:.1%}")

    st.write("Model Confidence")
    st.progress(max(0.0, min(confidence, 1.0)))

    st.caption(
        "Prediction generated using the trained ResNet18 model "
        "through Roboflow."
    )

except Exception as e:
    st.error(f"Classification failed: {e}")

        except Exception as e:
            st.error(f"Classification failed: {e}")

