import streamlit as st
import pickle
import pandas as pd
import requests


st.header("Formation Energy prediction for Alloy Design!")

api = "http://localhost:8000/" 


def make_ui():
    try:
        st.title("RF Interpolate")
        response_interpol = requests.get(api+"get_interpol_testdata")
        interpol_data = response_interpol.json()
        # df = pd.DataFrame(interpol_data)
        for i in range(len(interpol_data)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.text(f"{interpol_data[i]["formula"]}")  

            with col2:
                checkbox_val = st.checkbox(f"Show features", key=f'rf_in_checkbox_{i+1}')
            
            if checkbox_val:
                df = pd.DataFrame(interpol_data[i],index=[0])
                st.dataframe(df)

            with col3:
                if st.button(f"Predict {i+1}", key=f'rf_in_button_{i+1}'):
                    EF = requests.get(api+"/predict-rf_interpolate",params={"x_in":i})
                    enrgy_pred = EF.json()
                    st.write(f"Predicted EF {enrgy_pred["value"]}")
        
        st.title("XGB Interpolate")
        for i in range(len(interpol_data)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.text(f"{interpol_data[i]["formula"]}")  

            with col2:
                checkbox_val = st.checkbox(f"Show features", key=f'xgb_in_checkbox_{i+1}')
            
            if checkbox_val:
                df = pd.DataFrame(interpol_data[i],index=[0])
                st.dataframe(df)

            with col3:
                if st.button(f"Predict {i+1}", key=f'xgb_in_button_{i+1}'):
                    EF = requests.get(api+"/predict-xgb_interpolate",params={"x_in":i})
                    enrgy_pred = EF.json()
                    st.write(f"Predicted EF {enrgy_pred["value"]}")

        st.title("RF Extrapolate")
        response_extrapol = requests.get(api+"get_extrapol_testdata")
        extrapol_data = response_extrapol.json()
        # df = pd.DataFrame(interpol_data)
        for i in range(len(extrapol_data)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.text(f"{extrapol_data[i]["formula"]}")  

            with col2:
                checkbox_val = st.checkbox(f"Show features", key=f'rf_exp_checkbox_{i+1}')
            
            if checkbox_val:
                df = pd.DataFrame(extrapol_data[i],index=[0])
                st.dataframe(df)

            with col3:
                if st.button(f"Predict {i+1}", key=f'rf_exp_button_{i+1}'):
                    EF = requests.get(api+"/predict-rf_extrapolate",params={"x_in":i})
                    enrgy_pred = EF.json()
                    st.write(f"Predicted EF {enrgy_pred["value"]}")
        
        st.title("XGB Extrapolate")
        for i in range(len(extrapol_data)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.text(f"{extrapol_data[i]["formula"]}")  

            with col2:
                checkbox_val = st.checkbox(f"Show features", key=f'xgb_exp_checkbox_{i+1}')
            
            if checkbox_val:
                df = pd.DataFrame(extrapol_data[i],index=[0])
                st.dataframe(df)

            with col3:
                if st.button(f"Predict {i+1}", key=f'xgb_exp_button_{i+1}'):
                    EF = requests.get(api+"/predict-xgb_extrapolate",params={"x_in":i})
                    enrgy_pred = EF.json()
                    st.write(f"Predicted EF {enrgy_pred["value"]}")

    except Exception as e:
        print(e)

make_ui()

