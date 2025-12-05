import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.patches
import matplotlib.colors


class XGBoostRegressorWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


custom_style = """
    <style>
    .custom-header {
        font-size: 30px;
        margin-top: -50px;
        text-align: center;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .input-section-header {
        font-size: 22px;
        margin: 15px 0;
        font-weight: bold;
        color: #2E86AB;
        text-align: left;
    }
    .stNumberInput {
        margin-bottom: 8px;
        margin-top: 2px;
    }
    .section-divider {
        border: none;
        border-top: 4px solid #dee2e6;
        margin: 0px 0;
    }
    .stButton > button {
        width: 100%;
        padding: 10px 0;
    }
    </style>
"""

st.markdown(custom_style, unsafe_allow_html=True)

# 主标题
st.markdown(
    "<h1 class='custom-header'>Explainable prediction model for high-temperature compressive strength of self-compacting concrete</h1>",
    unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("<div class='input-section-header'>Input Parameters </div>",
            unsafe_allow_html=True)

col_input1, col_input2, col_input3 = st.columns(3)
inputs = {}
# ___________________________
with col_input1:
    inputs["Cement"] = st.number_input("Cement (kg/m$^3$)", min_value=220.0, max_value=635.0, value=440.0, key="Cement")
    inputs["fine aggregate"] = st.number_input("Fine aggregate (kg/m$^3$)", min_value=250.0, max_value=1018.0,value=863.0, key="fine aggregate")
    Fiber_type = st.selectbox(label="Fiber type", options=["0", "1", "2", "3"], index=0, key="Fiber type")
    inputs["Heating rate"] = st.number_input("Heating rate ($℃$/min)", min_value=1.0, max_value=20.0, value=1.0,key="Heating rate")

# ___________________________
with col_input2:
    inputs["SCMs"] = st.number_input("SCMs (kg/m$^3$)", min_value=0.0, max_value=450.0, value=110.0, key="SCMs")
    inputs["Coarse aggregate"] = st.number_input("Coarse aggregate (kg/m$^3$)", min_value=280.0, max_value=1149.0,value=465.0, key="Coarse aggregate")
    inputs["Fiber content"] = st.number_input("Fiber content (kg/m$^3$)", min_value=0.0, max_value=39.25, value=0.0,key="Fiber content")
    inputs["Exposure temperature"] = st.number_input("Exposure temperature ($℃$)", min_value=20.0, max_value=1029.0,value=600.0, key="Exposure temperature")

# ___________________________
with col_input3:
    inputs["water-binder ratio"] = st.number_input("water-binder ratio", min_value=0.26, max_value=0.64, value=0.33,key="water-binder ratio")
    inputs["Sand ratio"] = st.number_input("Sand ratio", min_value=0.38, max_value=0.75, value=0.65, key="Sand ratio")
    inputs["superplasticizer"] = st.number_input("Superplasticizer (kg/m$^3$)", min_value=2.0, max_value=50.0,value=9.63, key="Superplasticizer")
    inputs["Fire time"] = st.number_input("Fire time (hours)", min_value=1.0, max_value=4.0, value=3.0, key="Fire time")

inputs["Fiber type"] = float(Fiber_type)

st.markdown("<div class='input-section-header'>Range of application of the models  </div>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    param_text1 = st.text_input(label="Cement", value="220-635", key="param_range_text1")
    param_text4 = st.text_input(label="SCMs", value="0-450", key="param_range_text4")
    param_text7 = st.text_input(label="water-binder ratio", value="water-binder ratio: 0.26-0.64", key="param_range_text7")
    param_text10 = st.text_input(label="Fiber type:", value="0-3", key="param_range_text10")
with col2:
    param_text2 = st.text_input(label="Fine aggregate", value="250-1018", key="param_range_text2")
    param_text5 = st.text_input(label="Coarse aggregate", value="280-1149", key="param_range_text5")
    param_text8 = st.text_input(label="Sand ratio", value="0.38-0.75", key="param_range_text8")
    param_text11 = st.text_input(label="Fiber content", value="0-39.25", key="param_range_text11")
with col3:
    param_text3 = st.text_input(label="Heating rate", value="1-20", key="param_range_text3")
    param_text6 = st.text_input(label="Exposure temperature", value="20-1029", key="param_range_text6")
    param_text9 = st.text_input(label="Superplasticizer", value="2-50", key="param_range_text9")
    param_text12 = st.text_input(label="Fire time", value="1-4", key="param_range_text12")

st.markdown("<div class='input-section-header'>Output</div>",unsafe_allow_html=True)

col_submit, col_empty = st.columns([1, 5])
with col_submit:
    submit_btn = st.button("Prediction", type="primary")

prediction_result = ""
if submit_btn:
    try:

        regressor = joblib.load("SSA-Xgboost.pkl")

        feature_order = ['Cement', 'SCMs', 'water-binder ratio', 'fine aggregate',
                         'Coarse aggregate', 'Sand ratio', 'superplasticizer',
                         'Fiber type', 'Fiber content', 'Heating rate',
                         'Exposure temperature', 'Fire time']

        input_values = [inputs[feature] for feature in feature_order]
        X_input = pd.DataFrame([input_values], columns=feature_order)

        scaler_params = np.load('scaler_parameters.npy', allow_pickle=True).item()
        X_mean = scaler_params['X_mean']
        X_std = scaler_params['X_std']
        y_mean = scaler_params['y_mean']
        y_std = scaler_params['y_std']
        X_input_scaled = (X_input - X_mean) / X_std
        prediction_scaled = regressor.predict(X_input_scaled)[0]

        prediction = prediction_scaled * y_std + y_mean

        st.info(f"The predicted High-temperature Compressive Strength is: **{prediction:.2f} MPa**")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Please make sure 'SSA-Xgboost.pkl' and 'scaler_parameters.npy' are in the same directory.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.subheader("Model Interpretation")
uploaded_file = st.file_uploader("Select the CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[0] >= 1:
            expected_features = ['Feature0', 'Feature1', 'Feature2', 'Feature3',
                                 'Feature4', 'Feature5', 'Feature6', 'Feature7',
                                 'Feature8', 'Feature9', 'Feature10', 'Feature11']

            if not all(feature in df.columns for feature in expected_features):
                st.error(f"CSV file must contain these columns: {expected_features}")
            else:
                X = df[expected_features]

                regressor = joblib.load("SSA-Xgboost.pkl")
                scaler_params = np.load('scaler_parameters.npy', allow_pickle=True).item()
                X_mean = scaler_params['X_mean']
                X_std = scaler_params['X_std']
                y_mean = scaler_params['y_mean']
                y_std = scaler_params['y_std']

                X_scaled = (X - X_mean) / X_std

                predictions_scaled = regressor.predict(X_scaled)
                predictions = predictions_scaled * y_std + y_mean

                result_df = df.copy()
                result_df['Predicted Strength (MPa)'] = predictions

                explainer = shap.TreeExplainer(regressor.model)
                shap_values = explainer.shap_values(X_scaled)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<div class='input-section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)

                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_values, X_scaled, feature_names=expected_features, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.clf()

                with col2:
                    st.markdown("<div class='input-section-header'>SHAP Summary Plot</div>",unsafe_allow_html=True)

                    from matplotlib.colors import LinearSegmentedColormap
                    colors = ["#FF8C00", "#FFFFFF", "#000080"]
                    orange_navy_cmap = LinearSegmentedColormap.from_list("OrangeNavy", colors, N=100)
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_values, X_scaled, feature_names=expected_features,
                                      cmap=orange_navy_cmap,plot_type="dot", show=False)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.clf()


                col_ice, col_pdp = st.columns(2)

                with col_ice:
                    st.markdown("<div class='input-section-header'>ICE plot</div>", unsafe_allow_html=True)

                    feature_options = {
                        'Cement': 0,
                        'SCMs': 1,
                        'W/C': 2,
                        'fine aggregate': 3,
                        'Coarse aggregate': 4,
                        'Sand ratio': 5,
                        'superplasticizer': 6,
                        'Fiber type': 7,
                        'Fiber content': 8,
                        'Heating rate': 9,
                        'Exposure temperature': 10,
                        'Fire time': 11
                    }

                    selected_feature = st.selectbox(
                        "Please select features:",
                        options=list(feature_options.keys()),
                        index=10,
                        key="ice_feature_select"
                    )

                    feature_idx = feature_options[selected_feature]

                    feature_units = {
                        'Cement': '(kg/m$^3$)',
                        'SCMs': '(kg/m$^3$)',
                        'W/C': '',
                        'fine aggregate': '(kg/m$^3$)',
                        'Coarse aggregate': '(kg/m$^3$)',
                        'Sand ratio': '',
                        'superplasticizer': '(kg/m$^3$)',
                        'Fiber type': '',
                        'Fiber content': '(kg/m$^3$)',
                        'Heating rate': '($℃$/min)',
                        'Exposure temperature': '($℃$)',
                        'Fire time': '(hours)'
                    }

                    fig_ice, ax_ice = plt.subplots(figsize=(8, 6))

                    feature_min = X.iloc[:, feature_idx].min()
                    feature_max = X.iloc[:, feature_idx].max()

                    if selected_feature in ['W/C', 'Sand ratio']:
                        feature_range = np.arange(feature_min, feature_max + 0.01, 0.01)
                    elif selected_feature in ['Fiber type']:
                        feature_range = np.arange(feature_min, feature_max + 1, 1)
                    else:
                        step = max(0.1, (feature_max - feature_min) / 100)
                        feature_range = np.arange(feature_min, feature_max + step, step)

                    all_preds = []

                    n_samples = min(50, X.shape[0])
                    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)

                    with st.spinner(f"ICE analysis in progress"):
                        for i in sample_indices:
                            feature_values = np.tile(X.iloc[i].values, (len(feature_range), 1))
                            feature_values[:, feature_idx] = feature_range
                            feature_values_scaled = (feature_values - X_mean) / X_std
                            preds_scaled = regressor.predict(feature_values_scaled)
                            preds = preds_scaled * y_std + y_mean
                            all_preds.append(preds)
                            ax_ice.plot(feature_range, preds, color='#FFDCB1', alpha=0.5)

                        mean_preds = np.mean(np.array(all_preds), axis=0)
                        ax_ice.plot(feature_range, mean_preds, color='#000080', linewidth=3)
                        ax_ice.set_xlabel(f'{selected_feature} {feature_units[selected_feature]}', fontweight='bold')
                        ax_ice.set_ylabel('Compressive strength (MPa)', fontweight='bold')
                        ax_ice.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_ice)
                        plt.clf()

                with col_pdp:
                    st.markdown("<div class='input-section-header'>PDP-3D plot</div>", unsafe_allow_html=True)

                    feature_options = {
                        'Cement': 0,
                        'SCMs': 1,
                        'W/C': 2,
                        'fine aggregate': 3,
                        'Coarse aggregate': 4,
                        'Sand ratio': 5,
                        'superplasticizer': 6,
                        'Fiber type': 7,
                        'Fiber content': 8,
                        'Heating rate': 9,
                        'Exposure temperature': 10,
                        'Fire time': 11
                    }

                    feature_units = {
                        'Cement': '(kg/m³)',
                        'SCMs': '(kg/m³)',
                        'W/C': '',
                        'fine aggregate': '(kg/m³)',
                        'Coarse aggregate': '(kg/m³)',
                        'Sand ratio': '',
                        'superplasticizer': '(kg/m³)',
                        'Fiber type': '',
                        'Fiber content': '(kg/m³)',
                        'Heating rate': '(℃/min)',
                        'Exposure temperature': '(℃)',
                        'Fire time': '(hours)'
                    }

                    col_feat1, col_feat2 = st.columns(2)

                    with col_feat1:
                        feature1 = st.selectbox(
                            "Feature1 (X-axis):",
                            options=list(feature_options.keys()),
                            index=3,
                            key="pdp_feature1"
                        )

                    with col_feat2:
                        available_features = [f for f in feature_options.keys() if f != feature1]
                        feature2 = st.selectbox(
                            "Feature2 (Y-axis):",
                            options=available_features,
                            index=9,
                            key="pdp_feature2"
                        )

                    features = [feature_options[feature1], feature_options[feature2]]
                    n_points = 50

                    with st.spinner(f"Generating 3D PDP plot for {feature1} and {feature2}..."):
                        try:
                            fig_pdp = plt.figure(figsize=(8, 7))
                            ax_pdp = fig_pdp.add_subplot(111, projection='3d')

                            x_min, x_max = X.iloc[:, features[0]].min(), X.iloc[:, features[0]].max()
                            y_min, y_max = X.iloc[:, features[1]].min(), X.iloc[:, features[1]].max()

                            x_axis = np.linspace(x_min, x_max, n_points)
                            y_axis = np.linspace(y_min, y_max, n_points)
                            xx, yy = np.meshgrid(x_axis, y_axis)

                            all_samples = np.zeros((len(xx.ravel()), X.shape[1]))
                            for i in range(len(xx.ravel())):
                                all_samples[i, features[0]] = (xx.ravel()[i] - X_mean[features[0]]) / X_std[features[0]]
                                all_samples[i, features[1]] = (yy.ravel()[i] - X_mean[features[1]]) / X_std[features[1]]

                            preds_scaled = regressor.predict(all_samples)
                            preds = preds_scaled * y_std + y_mean
                            preds = preds.reshape(xx.shape)

                            colors = ["#FF8C00", "#FFFFFF", "#000080"]
                            orange_navy_cmap = LinearSegmentedColormap.from_list("OrangeNavy", colors, N=500)

                            surface = ax_pdp.plot_surface(xx, yy, preds, cmap=orange_navy_cmap,
                                                          alpha=0.8, antialiased=True)

                            ax_pdp.set_xlabel(f'{feature1} {feature_units[feature1]}',
                                              labelpad=10, fontweight='bold')
                            ax_pdp.set_ylabel(f'{feature2} {feature_units[feature2]}',
                                              labelpad=10, fontweight='bold')
                            ax_pdp.set_zlabel('Compressive strength (MPa)',
                                              labelpad=10, fontweight='bold')

                            ax_pdp.view_init(elev=30, azim=45)

                            cbar = fig_pdp.colorbar(surface, ax=ax_pdp, shrink=0.5, aspect=20)
                            cbar.set_label('Strength (MPa)', fontweight='bold')

                            plt.tight_layout()
                            st.pyplot(fig_pdp)
                            plt.clf()

                        except Exception as e:
                            st.error(f"Error generating 3D plot: {str(e)}")

#___________________________________________________________________________________________________________________________________
                default_pos_color = "#ff0051"
                default_neg_color = "#008bfb"
                positive_color = "#FF8C00"
                negative_color = "#000080"


                def customize_waterfall_colors():
                    for fc in plt.gcf().get_children():
                        for fcc in fc.get_children():
                            if (isinstance(fcc, matplotlib.patches.FancyArrow)):
                                if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                                    fcc.set_facecolor(positive_color)
                                elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                                    fcc.set_color(negative_color)
                            elif (isinstance(fcc, plt.Text)):
                                if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                                    fcc.set_color(positive_color)
                                elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                                    fcc.set_color(negative_color)


                st.markdown("<div class='input-section-header'>Individual sample explanation</div>",
                            unsafe_allow_html=True)

                col3, col4 = st.columns(2)

                with col3:
                    sample_index_1 = st.number_input(
                        "Enter the sample index:",
                        min_value=0, max_value=len(X) - 1, value=174, step=1, key="sample_index_1")

                with col4:
                    sample_index_2 = st.number_input(
                        "Enter the sample index:",
                        min_value=0, max_value=len(X) - 1, value=63, step=1, key="sample_index_2")

                with col3:
                    if sample_index_1 < len(X):
                        sample_original_1 = X.iloc[sample_index_1]
                        sample_scaled_1 = X_scaled.iloc[sample_index_1]

                        predicted_value_scaled_1 = regressor.predict([sample_scaled_1])[0]
                        predicted_value_1 = predicted_value_scaled_1 * y_std + y_mean

                        explanation_1 = shap.Explanation(
                            values=shap_values[sample_index_1],
                            base_values=explainer.expected_value,
                            data=sample_scaled_1.values,
                            feature_names=expected_features
                        )

                        st.write(f"Predicted value: {predicted_value_1:.2f}")
                        plt.figure(figsize=(8, 6))
                        shap.waterfall_plot(explanation_1, show=False)
                        customize_waterfall_colors()
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.close()

                with col4:
                    if sample_index_2 < len(X):
                        sample_original_2 = X.iloc[sample_index_2]
                        sample_scaled_2 = X_scaled.iloc[sample_index_2]

                        predicted_value_scaled_2 = regressor.predict([sample_scaled_2])[0]
                        predicted_value_2 = predicted_value_scaled_2 * y_std + y_mean

                        explanation_2 = shap.Explanation(
                            values=shap_values[sample_index_2],
                            base_values=explainer.expected_value,
                            data=sample_scaled_2.values,
                            feature_names=expected_features
                        )

                        st.write(f"Predicted value: {predicted_value_2:.2f}")
                        plt.figure(figsize=(8, 6))
                        shap.waterfall_plot(explanation_2, show=False)
                        customize_waterfall_colors()
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.close()

    except Exception as e:
        st.error(f"Error generating explanation: {e}")