import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('svm_db.pkl','rb'))

def predict_diabetes(pregnancy,glucose,bloodpressure,insulin,BMI,age):
    input = np.array([[pregnancy,glucose,bloodpressure,insulin,BMI,age]]).astype(np.float64)
    prediction = model.predict(input)
    return float(prediction)

def main():
    st.title("Diabetes Prediction ")
    html_temp = """
    <div style="background-color= #FFFFF ;padding:10px">
    <h2 style="color:black;text-align:center;">Diabetes Prediction Application</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    pregnancy = st.text_input("Number of Pregnencies" ," ")
    glucose = st.text_input("Glucose level", " ")
    bloodpressure = st.text_input("Blood Pressure(BPM)", " ")
    insulin = st.text_input("Insulin level", " ")
    BMI = st.text_input("Body Mass Index", " ")
    age = st.text_input("Age(years)", " ")
    safe_html = """
    <div style="background-color:#F4D03F;padding:10px >
    <h2 style="color:white;text-align:center;">You are safe.Don't eat more Sugar!!</h2>
    </div>
    """
    danger_html= """
    <div style="background-color:#F00000;padding:10px >
    <h2 style="color:white;text-align:center;">You are not safe,get checked OUT!!</h2>
    </div>
    """

    if st.button("Predict"):
        output=predict_diabetes(pregnancy,glucose,bloodpressure,insulin,BMI,age)
        if not output >0.7:
            a ="You will not get Diabetes"
        else:
            a="You will get Diabetes"
        st.success("{}".format(a))

        if output <=0.7:
            st.markdown(safe_html,unsafe_allow_html=True)
        else:
            st.markdown(danger_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()