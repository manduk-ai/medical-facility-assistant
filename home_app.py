"""
Filename: home_app.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: This is the home page of the application. It shows the main content and offers navigation to other pages.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

import streamlit as st
from st_pages import add_page_title, show_pages, Page
import auth.authenticate as authenticate

# Change this while upgrading the app
VERSION = "0.6"

# Check authentication when user lands on the home page.
authenticate.set_st_state_vars()

# Add login/logout buttons
if st.session_state["authenticated"]:
    authenticate.button_logout()
else:
    authenticate.button_login()

# Add pages to the sidebar
show_pages(
    [
        Page("home_app.py", "Home", "🏠"),
        Page("patient_data/edit_patient_data.py", "Patient Data", "👩‍🦰"),
        Page("chat/chat_app.py", "Chatbot Assistant", "🤖"),
        Page("sql_db/view_sql_db.py", "Doctors & Appointments", "👩‍⚕️"),
        Page("unstructured_data/view_unstructured_data.py", "Procedures & Policies", "📜"),        
    ]
)

add_page_title()  # adds title and icon to current page

st.write(
    f"""
Welcome to the intelligent Chatbot Assistant - a sandbox demonstration app designed for a fictitious 'My Medical Facility.' This app showcases how a Chatbot Assistant could operate within a medical institution, offering patients the ability to interact about various services including booking appointments, exploring available treatments, understanding pricing, and learning about policies and procedures.\n

The chatbot leverages Retrieval-Augmented Generation (RAG) technology to provide accurate information, minimize errors, and facilitate simple transactions like appointment bookings.\n 

Please note that the app is still a work in progress, with significant enhancements planned to improve both reliability and quality.\n

In the 'Patient Data' section, you can input specific patient details to personalize your interactions - which is recommended. However, this is optional as the bot can also assume default values or request clarification when needed.\n

Explore the 'Doctors & Appointments' section to browse through synthetic data about specialties, available doctors, and appointment slots at 'My Medical Facility.'\n

The 'Procedures & Policies' section provides an overview of the types of information the bot can manage, offering guidance on what you can inquire about during your interaction.\n

The 'Chatbot Assistant' section is where you can interact with the bot. You may also parametrize the bot by selecting the language and type of models used, with GPT-4 being the most capable. The app calculates conversation cost by counting number of tokens at input and output and multiplying by the OpenAI's rates.\n

Feel free to explore and interact with our Chatbot Assistant as part of your experience in this simulated medical environment.\n

© 2024 All rights reserved. Version {VERSION} developed by [Szymon Manduk](https://manduk.ai/).
    """
)
