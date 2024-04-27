"""
Filename: edit_patient_data.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: This is the screen of the app called 'Patient Data'. It allows the user to input or update patient's data.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

import streamlit as st
import re
from st_pages import add_page_title
import auth.authenticate as authenticate

# Check authentication when user lands on the page.
authenticate.set_st_state_vars()

# Add login/logout buttons
if st.session_state["authenticated"]:
    authenticate.button_logout()
else:
    authenticate.button_login()

add_page_title()  # adds title and icon to current page


def validate_email(email):
    """
    Validate the email format.

    Args:
        email (str): The email address to be validated.

    Returns:
        bool: True if the email format is valid, False otherwise.
    """
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False
    return True


def validate_phone(phone):
    """Validate the phone number format (XXX-XXX-XXX).

    Args:
        phone (str): The phone number to be validated.

    Returns:
        bool: True if the phone number is in the correct format (XXX-XXX-XXX), False otherwise.
    """
    # Updated regex with start (^) and end ($) anchors
    if not re.match(r"^\d{3}-\d{3}-\d{3}$", phone):
        return False
    return True


def main():
    st.write("Please enter or update patient's data.")
    st.write("If you start a chat with Chatbot Assistant without providing patient's data, the first and last names will assume default values (i.e. John Brown).")
    st.write("If the email or phone number is not provided, Chatbot Assistant should ask for them before making an appointment. Please note, the Assistant does not send emails or text messages, so both should be an example (synthetic) data.") 
    st.write("IMPORTANT: Do NOT provide real patient data or your personal data, as this demo application does not process any personal or sensitive information.")
    
    # We want to check if the data is already saved 
    if 'my_session' in st.session_state:  # it can be saved in my_session object (ChatSession class) 
        v_first_name, v_last_name, v_email, v_phone = st.session_state.my_session.get_raw_patient_data()
    else: # or if chat session was not started, it can be saved in a temporart 'patient_data' object
        if 'patient_data' in st.session_state:
            v_first_name = st.session_state.patient_data['first_name']
            v_last_name = st.session_state.patient_data['last_name']
            v_email = st.session_state.patient_data['email']
            v_phone = st.session_state.patient_data['phone']
        else:  # or it can be uninitialized altogether
            v_first_name = ""
            v_last_name = ""
            v_email = ""
            v_phone = ""

    # Create a form for patient data
    with st.form("patient_form"):
        first_name = st.text_input("First name*", v_first_name)
        last_name = st.text_input("Last name*", v_last_name)
        email = st.text_input("Email", v_email)
        phone = st.text_input("Phone number in the format XXX-XXX-XXX", v_phone)
        
        # Create a Submit button
        submitted = st.form_submit_button("Save")
        
        if submitted:
            if first_name and last_name:  # Check if mandatory fields are filled
                # Validate email
                if email and not validate_email(email):
                    st.error("Email is not in a correct format!")
                    return
                
                # Validate phone number
                if phone and not validate_phone(phone):
                    st.error("Phone number is not in a correct format!")
                    return
                
                # We want to check if the data is already saved 
                if 'my_session' in st.session_state:  # we are saving in my_session object (ChatSession class) 
                    st.session_state.my_session.set_patient_data(first_name, last_name, email, phone)
                else: # or if chat session was not started, it can be saved in a temporart 'patient_data' object
                    if 'patient_data' not in st.session_state:  # if 'patient_data' was not previously initializedwe need to create an empty dictionary first
                        st.session_state.patient_data = {}
                    st.session_state.patient_data['first_name'] = first_name
                    st.session_state.patient_data['last_name'] = last_name
                    st.session_state.patient_data['email'] = email
                    st.session_state.patient_data['phone'] = phone
                st.write("Patient data saved successfully!")
            else:
                st.error("First name and last name are required!")

# if user logged in and belongs to admin group
if st.session_state["authenticated"] and ("admin" in st.session_state["user_cognito_groups"] or "user" in st.session_state["user_cognito_groups"]):
    main()
else:
    if st.session_state["authenticated"]:
        st.write("You do not have access to this page. Please contact administrator.")
    else:
        st.write("Please login first.")