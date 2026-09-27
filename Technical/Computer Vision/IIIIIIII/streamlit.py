import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Project", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Sidebar")
st.sidebar.header("Sidebar Header")
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"])
if theme == "Light":
    st.sidebar.success("Light theme selected.")
else:
    st.sidebar.error("Dark theme selected.")

st.title("My Streamlit App")
st.caption("This is a simple Streamlit application.")
st.header("Welcome to the App")
st.subheader("This is a subheader")
st.text("This is some text.")
a = 5
st.write(f"The value of a is: {a}")

st.markdown("This is a **markdown** text with *italic* and **bold** formatting.")
st.divider()
st.image("E:\\AI\\NTI_Compter-Vision-Training\\Image Processing\\Practical\\Images\\background.jpeg")

st.success("This is a success message.")
st.warning("This is a warning message.")
st.error("This is an error message.")
st.info("This is an info message.")

name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!") 

age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
if age: 
    st.write(f"You are {age} years old.")


score = st.slider("Select your score:", min_value=0, max_value=100, step=1)
if score:
    st.write(f"Your score is: {score}")
    
    
skills = st.multiselect("Select your skills:", ["Python", "Java", "C++", "JavaScript"]) 
if skills:
    st.write(f"Your selected skills are: {', '.join(skills)}")
    
city = st.selectbox("Select your city:", ["New York", "Los Angeles", "Chicago", "Houston"])
if city:
    st.write(f"You selected: {city}")
    
agree = st.checkbox("I agree to the terms and conditions")
if agree:
    st.success("Thank you for agreeing to the terms and conditions.")
else:
    st.error("You must agree to the terms and conditions to proceed.")
    

level = st.radio("Select your level:", ["Beginner", "Intermediate", "Advanced"])
if level:
    st.write(f"You selected: {level}")
    

if st.button("Click me!"):
    st.success("Button clicked!")
    st.line_chart(np.random.randn(20, 3), use_container_width=True)

col1 , col2 = st.columns(2)
with col1:
    st.write("This is column 1")
    st.bar_chart(np.random.randn(20, 3))
with col2:
    st.write("This is column 2")
    st.scatter_chart(np.random.randn(20, 3)) 
    st.divider()
    
    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots()
    plt.hist(np.random.randn(1000), bins=30) 
    st.pyplot(fig)
    fig, ax = plt.subplots()
    plt.boxplot(np.random.randn(1000))
    st.pyplot(fig)
    st.area_chart(np.random.randn(20, 3))
    

if st.button("show progress bar"):
    with st.spinner("Loading..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
    st.success("Done!")
    
if st.button("show progress bar2"):
    with st.spinner("Loading..."):
        time.sleep(2)
    st.success("Done!")
    
