import streamlit as st
import pandas as pd
import pickle5 as pickle
import plotly.graph_objects as go
import numpy as np


# Function to read and clean the data
def get_clean_data():
    """
    Read and clean the breast cancer data.

    Returns:
        pd.DataFrame: Cleaned data with 'diagnosis' column mapped to binary values.
    """
    data = pd.read_csv("data/data.csv")
    # Drop unnecessary columns
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    # Map diagnosis to binary
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


# Function to create sidebar sliders for user input
def add_sidebar():
    """
    Create sidebar sliders for user to input cell nuclei measurements.

    Returns:
        dict: User input values for the measurements.
    """
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    # Create sliders for each measurement and store user input in a dictionary
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )

    return input_dict


# Function to make predictions based on user input
def add_predictions(input_data):
    """
    Predict the diagnosis based on user input using a pre-trained model.

    Args:
        input_data (dict): User input values for the measurements.
    """
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # Convert input data to DataFrame to maintain feature names
    input_df = pd.DataFrame([input_data])

    # Apply the scaler transform
    input_array_scaled = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write(
            "<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True
        )

    # Display probabilities of each class
    st.write(
        f"Probability of being benign: {model.predict_proba(input_array_scaled)[0][0]:.2f}"
    )
    st.write(
        f"Probability of being malignant: {model.predict_proba(input_array_scaled)[0][1]:.2f}"
    )

    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis."
    )


# Function to scale user input values
def get_scaled_values(input_dict):
    """
    Scale the user input values based on the dataset.

    Args:
        input_dict (dict): User input values for the measurements.

    Returns:
        dict: Scaled values for the user input.
    """
    data = get_clean_data()
    X = data.drop(["diagnosis"], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


# Function to generate a radar chart based on user input
def get_radar_chart(input_data):
    """
    Generate a radar chart based on user input values.

    Args:
        input_data (dict): User input values for the measurements.

    Returns:
        plotly.graph_objects.Figure: Radar chart visualization.
    """
    input_data = get_scaled_values(input_data)

    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]

    fig = go.Figure()

    # Add trace for mean values
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_mean"],
                input_data["texture_mean"],
                input_data["perimeter_mean"],
                input_data["area_mean"],
                input_data["smoothness_mean"],
                input_data["compactness_mean"],
                input_data["concavity_mean"],
                input_data["concave points_mean"],
                input_data["symmetry_mean"],
                input_data["fractal_dimension_mean"],
            ],
            theta=categories,
            fill="toself",
            name="Mean Value",
        )
    )

    # Add trace for standard error values
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_se"],
                input_data["texture_se"],
                input_data["perimeter_se"],
                input_data["area_se"],
                input_data["smoothness_se"],
                input_data["compactness_se"],
                input_data["concavity_se"],
                input_data["concave points_se"],
                input_data["symmetry_se"],
                input_data["fractal_dimension_se"],
            ],
            theta=categories,
            fill="toself",
            name="Standard Error",
        )
    )

    # Add trace for worst values
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_worst"],
                input_data["texture_worst"],
                input_data["perimeter_worst"],
                input_data["area_worst"],
                input_data["smoothness_worst"],
                input_data["compactness_worst"],
                input_data["concavity_worst"],
                input_data["concave points_worst"],
                input_data["symmetry_worst"],
                input_data["fractal_dimension_worst"],
            ],
            theta=categories,
            fill="toself",
            name="Worst Value",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )

    return fig


# Main function to run the Streamlit app
def main():
    """
    Main function to run the breast cancer prediction app.
    """
    page_title = "Breast Cancer Prediction"
    des = (
        "Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. "
        "This app predicts using a machine learning model whether a breast is benign or malignant. "
        "You can also update the measurements by hand using the sliders in the sidebar."
    )

    st.set_page_config(
        page_title=page_title,
        page_icon="female-doctor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    input_data = add_sidebar()

    with st.container():
        st.title(page_title)
        st.write(des)

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
