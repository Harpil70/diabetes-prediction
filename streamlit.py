import pickle
import numpy as np
import streamlit as st

load_model = pickle.load(open('trained_model.sav','rb'))

def diabetic_pred(input_data):

    input_data_arr = np.array(input_data).reshape(1,-1)
    y_pred = load_model.predict(input_data_arr)
    if y_pred == 0:
        return 'Person is not diabetic'
    elif y_pred == 1:
        return 'Person is diabetic'

def main():
    st.title("Diabetes Web interface")

    Pregnancies = st.text_input('Number of Pregnancies:')
    Glucose = st.text_input('Glucose Level:')
    BloodPressure = st.text_input('BloodPressure Value: ')
    SkinThickness = st.text_input('SkinThickness Value :')
    Insulin = st.text_input('Insulin Level:')
    BMI = st.text_input('BMI Value:')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value:')
    Age = st.text_input('Age of Person:')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetic_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
