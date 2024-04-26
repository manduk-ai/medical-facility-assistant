"""
Filename: page_template.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: This is a template for a page in the application. It shows how to add authentication to the page.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

import streamlit as st
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

def main():
    st.write(
        """
        This shows that you have access to the page. Enjoy!
        """
    )

# if user logged in and belongs to group group2
if st.session_state["authenticated"] and ("admin" in st.session_state["user_cognito_groups"] or "user" in st.session_state["user_cognito_groups"]):
    main()
else:
    if st.session_state["authenticated"]:
        st.write("You do not have access to this page. Please contact administrator.")
    else:
        st.write("Please login first.")