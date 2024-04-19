import json
import mlflow

import numpy as np
import pandas as pd
import streamlit as st

from modules.config import Config

title = "Concrete Compressive Strength Prediction for Specific Ages with Modified SNF Admixture (Beta version) for 1 m3 Concrete"
footer = """(c) [2024](https://github.com/hyperforest/)"""
MODEL_DIR = "models/31f693930f4646c0a2767a9aea2b037d/artifacts/model"
CONFIG_DIR = "configs/config_v7.json"


with open(CONFIG_DIR, "r") as file:
    config = Config(**json.load(file))


def predict_slump(data):
    water = data["water"]
    cement = data["cement"]
    fas = water / cement
    kg_sikacim = data["sikacim_kg"]

    slump = (
        57.34 * fas +
        0.26 * water +
        0.96 * kg_sikacim +
        0.00 * np.log1p(fas) +
        0.00 * np.log1p(water) +
        3.79 * np.log1p(kg_sikacim) +
        -73.48
    )

    return slump


def main():
    st.set_page_config(layout="centered", page_icon="🪨", page_title=title)
    st.title(title)

    model = mlflow.sklearn.load_model(MODEL_DIR)

    sample_data = pd.read_csv("datasets/sample_input.csv")

    st.markdown("## 📂 Multiple Data Prediction")
    st.warning(
        "❗ All of the calculation uses standard setting: height = 300 mm, diameter = 150 mm"
    )

    with st.container():
        upform = st.form("Upload Data")
        submit_file = upform.file_uploader("⬆️ Upload CSV file")
        submit_up = upform.form_submit_button("Predict", use_container_width=True)
        see_sample = upform.form_submit_button(
            "CSV format sample", use_container_width=True
        )

        if see_sample:
            st.dataframe(sample_data.head(3))
            st.download_button(
                "👉 Download CSV sample",
                sample_data.to_csv(index=False).encode("utf-8"),
                file_name="sample_input.csv",
                mime="text/csv",
                use_container_width=True
            )

        if submit_up:
            data = pd.read_csv(submit_file)

            data["height"] = 300.0
            data["diameter"] = 150.0

            data["fas"] = data["water"] / data["cement"]
            data["area"] = np.pi * (data["diameter"] / 2) ** 2
            pred_kN = model.predict(data[config.features])
            pred_MPa = 1000 * pred_kN / data["area"]

            data["predicted_max_load (kN)"] = pred_kN.round(2)
            data["predicted_max_load (MPa)"] = pred_MPa.round(2)
            data["predicted_slump (cm)"] = data.apply(predict_slump, axis=1).round(2)

            top_n = min(5, len(data))
            st.success(f"Prediction success! Here is the top {top_n} rows of the result:")
            st.dataframe(data.head(top_n))

            # click to download
            st.download_button(
                "👉 Download result",
                data.to_csv(index=False).encode("utf-8"),
                file_name="output.csv",
                mime="text/csv",
                use_container_width=True
            )

    with st.container():
        st.markdown("## 📝 Or... Try One Example!")
        form = st.form("Input Data")

        age = form.number_input(
            "Concrete age (days)", min_value=None, max_value=None, value=1
        )
        water = form.number_input(
            "Water content (L)", min_value=None, max_value=None, value=205.0
        )
        cement = form.number_input(
            "Cement content (kg)", min_value=None, max_value=None, value=408.0
        )
        fine_aggregate = form.number_input(
            "Fine aggregates (kg)", min_value=None, max_value=None, value=715.0
        )
        coarse_aggregate = form.number_input(
            "Coarse aggregates (kg)", min_value=None, max_value=None, value=1072.0
        )
        sikacim = form.number_input(
            "Modified SNF Admixture (kg)", min_value=None, max_value=None, value=0.0,
        )

        diameter, height = 150.0, 300.0
        fas = water / cement
        area = np.pi * (diameter / 2) ** 2

        submit_indiv = form.form_submit_button("Predict", use_container_width=True)

        if submit_indiv:
            if not (1 <= age <= 28):
                st.error("Data out of simulation: age must be between 1 and 28 days")
            elif not (164 <= water <= 205):
                st.error("Data out of simulation: water content must be between 164 and 205 L")
            elif not (327 <= cement <= 440):
                st.error("Data out of simulation: cement content must be between 327 and 440 kg")
            elif not (715 <= fine_aggregate <= 764):
                st.error("Data out of simulation: fine aggregates must be between 715 and 764 kg")
            elif not (1072 <= coarse_aggregate <= 1146):
                st.error("Data out of simulation: coarse aggregates must be between 1072 and 1146 kg")
            elif not ((0 <= sikacim) and (sikacim <= 0.02 * cement)):
                max_lim = 0.02 * cement
                st.error(f"SNF dose exceeds the maximum allowance (2% of cement mass = {max_lim:.2f} kg)")
            else:
                data = {
                    "age_days": age,
                    "cement": cement,
                    "water": water,
                    "fas": fas,
                    "fine_aggregate_kg": fine_aggregate,
                    "coarse_aggregate_kg": coarse_aggregate,
                    "sikacim_kg": sikacim,
                    "fas_kg": fas,
                    "diameter": diameter,
                    "height": height,
                }

                data = pd.Series(data).to_frame(name=0).T
                pred_kN = model.predict(data[config.features])[0]
                pred_MPa = 1000 * pred_kN / area
                pred_slump = predict_slump(data.iloc[0])

                st.success("Prediction success!")

                text = f'''
                Prediction result:

                - Maximum load: {pred_MPa:.2f} MPa ({pred_kN:.2f} kN)                
                - Slump: {pred_slump:.2f} cm
                '''
                st.markdown(text)

    st.write(footer)


if __name__ == "__main__":
    main()
