import os
import tempfile

import streamlit as st
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration,
)

# Roboflow API key from Streamlit Secrets
api_key = st.secrets["ROBOFLOW_API_KEY"]


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fabric Quality Classification",
    page_icon="🧵",
    layout="centered",
)

st.title("🧵 Fabric Quality Classification")
st.write(
    "Upload a fabric image to identify its quality defect."
)


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a fabric image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Fabric Image",
        use_container_width=True,
    )

    if st.button("🔍 Classify Fabric", type="primary"):

        try:
            # Connect to Roboflow
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            ).configure(
                InferenceConfiguration(
                    api_key_transport="header"
                )
            )

            # Save uploaded image temporarily
            file_extension = os.path.splitext(
                uploaded_file.name
            )[1] or ".jpg"

            with tempfile.NamedTemporaryFile(
                suffix=file_extension,
                delete=False,
            ) as temp_file:
                temp_file.write(
                    uploaded_file.getvalue()
                )
                temp_path = temp_file.name

            try:
                # Run the Roboflow workflow
                with st.spinner(
                    "Analyzing fabric quality..."
                ):
                    result = client.run_workflow(
                        workspace_name="sanya-mohanty",
                        workflow_id=(
                            "classification-of-fabric-quality-"
                            "vmy-first-project-mdcnu-1-resnet18-t1-logic"
                        ),
                        images={
                            "image": temp_path
                        },
                        use_cache=True,
                    )

            finally:
                # Remove temporary image
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # ------------------------------------------
            # Extract classification result
            # ------------------------------------------

            data = (
                result[0]
                if isinstance(result, list)
                else result
            )

            if (
                isinstance(data, dict)
                and isinstance(
                    data.get("predictions"), dict
                )
            ):
                data = data["predictions"]

            label = data.get("top", "Unknown")
            confidence = float(
                data.get("confidence", 0)
            )

            # Handle confidence expressed as a percentage
            if confidence > 1:
                confidence = confidence / 100

            confidence = max(
                0.0, min(confidence, 1.0)
            )

            # ------------------------------------------
            # Display clean result
            # ------------------------------------------

            st.divider()
            st.subheader("Classification Result")

            st.success(
                f"Predicted Fabric Defect: **{label.title()}**"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Class",
                    label.title(),
                )

            with col2:
                st.metric(
                    "Confidence Score",
                    f"{confidence:.1%}",
                )

            st.write("Model Confidence")
            st.progress(confidence)

            st.caption(
                "Prediction generated using the "
                "trained ResNet18 model through Roboflow."
            )

        except Exception as e:
            st.error(
                f"Classification failed: {e}"
            )
