import json
import mlflow

import numpy as np
import pandas as pd
import streamlit as st

from modules.config import Config

title = "Mix Design Project"
footer = """(c) 2024"""
MODEL_DIR = "../models/31f693930f4646c0a2767a9aea2b037d/artifacts/model"
CONFIG_DIR = "../configs/config_v7.json"


with open(CONFIG_DIR, "r") as file:
    config = Config(**json.load(file))


def main():
    st.set_page_config(
        layout="centered", page_icon="🪨", page_title="Mix Design Project"
    )
    st.title(title)

    model = mlflow.sklearn.load_model(MODEL_DIR)

    sample_data = pd.read_csv("../datasets/sample_input.csv")

    st.markdown("## 📂 Prediksi banyak data sekaligus")
    with st.container():

        upform = st.form("Upload Data")
        submit_file = upform.file_uploader("⬆️ Upload file CSV")
        submit_up = upform.form_submit_button("Prediksi ✨", use_container_width=True)
        see_sample = upform.form_submit_button(
            "Lihat contoh format CSV 🔎", use_container_width=True
        )

        if see_sample:
            st.dataframe(sample_data.head(3))
            st.download_button(
                "👉 Download contoh CSV",
                sample_data.to_csv(index=False).encode("utf-8"),
                file_name="sample_input.csv",
                mime="text/csv",
            )

        if submit_up:
            data = pd.read_csv(submit_file)
            data["fas"] = data["water"] / data["cement"]
            data["area"] = np.pi * (data["diameter"] / 2) ** 2
            pred_kN = model.predict(data[config.features])
            pred_MPa = 1000 * pred_kN / data["area"]

            data["Prediksi (kN)"] = pred_kN.round(2)
            data["Prediksi (MPa)"] = pred_MPa.round(2)

            top_n = min(5, len(data))
            st.success(
                f"Berhasil memprediksi! Berikut prediksi dari {top_n} baris teratas"
            )
            st.dataframe(data.head(top_n))

            # click to download
            st.download_button(
                "👉 Download hasil prediksi",
                data.to_csv(index=False).encode("utf-8"),
                file_name="output.csv",
                mime="text/csv",
            )

    with st.container():
        st.markdown("## 📝 Atau... coba satu contoh!")
        form = st.form("Input Data")

        age = form.number_input("Umur (hari)", min_value=None, max_value=None, value=1)
        diameter = form.number_input(
            "Diameter (mm)", min_value=None, max_value=None, value=150.0
        )
        height = form.number_input(
            "Tinggi (mm)", min_value=None, max_value=None, value=300.0
        )
        water = form.number_input(
            "Kebutuhan air (L)", min_value=None, max_value=None, value=205.0
        )
        cement = form.number_input(
            "Kebutuhan semen (kg)", min_value=None, max_value=None, value=408.0
        )
        fine_aggregate = form.number_input(
            "Kebutuhan agregat halus (kg)", min_value=None, max_value=None, value=715.0
        )
        coarse_aggregate = form.number_input(
            "Kebutuhan agregat kasar (kg)", min_value=None, max_value=None, value=1072.0
        )
        sikacim = form.number_input(
            "Kebutuhan sikacim (kg)", min_value=None, max_value=None, value=0.0
        )

        fas = water / cement
        area = np.pi * (diameter / 2) ** 2

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

        submit_indiv = form.form_submit_button("Prediksi ✨", use_container_width=True)

        if submit_indiv:
            data = pd.Series(data).to_frame(name=0).T
            pred_kN = model.predict(data[config.features])[0]
            pred_MPa = 1000 * pred_kN / area

            text = (
                f"Beban maksimum yang diprediksi: {pred_MPa:.2f} MPa ({pred_kN:.2f} kN)"
            )
            st.success(text)

    st.write(footer)


if __name__ == "__main__":
    main()
