"""
Filename: view_unstructured_data.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: This is the screen called from 'Procedures & Policies' menu. It provides information on the types of data that the Chatbot Assistant can handle through vector database.
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
    st.write("This Chatbot Assistant utilizes a technique known as **retrieval-augmented generation** (RAG) to deliver precise and reliable information efficiently.")
    st.write("""RAG can access factual and up-to-date information through a traditional Relational Database Management System (RDBMS), as outlined in the ‘Doctors & Appointments’ section, or a vector store — a specialized database designed to handle large volumes of diverse, often unstructured data.""")
    st.write(
"""In this setup, the vector database is employed to store, search, and retrieve the following types of information:
- **General Information on 'My Medical Facility'**: This covers contact data, data protection policies, and information about cookies. For details on the general information that can be retrieved, refer to section I. below.
- **Medical Procedures and Guidance**: This includes detailed descriptions of procedures and treatments, pre-treatment guidance, and post-operative recommendations. For a comprehensive list of medical information accessible through the Chat Assistant, see section II. below.\n
\n
""")
    st.write(
"""
The detailed breakdowns provided below will help you understand the scope of information that the Chat Assistant can handle and respond to effectively.\n
\n**I. GENERAL INFORMATION**\n
- **Contact Data**: Details of 'My Medical Facility' including location, tax ID, email, and telephone number.
- **Cookies Policy**: Explanation of cookie use on the website, detailing types of cookies (technical, analytical, marketing), their purposes, and user rights regarding cookie management.
- **Policy on Personal Data Protection**: Information on the principles of personal data protection followed by My Medical Facility.
- **Data Processing Details**: Specifics on how personal data is collected, processed, stored, and used, including data of customers, employees, and others.
- **Security Measures**: Details on technical and organizational measures in place to protect personal data, including system security protocols, data integrity measures, and procedures for managing data breaches.
- **Rights of Data Subjects**: Information on the rights of individuals whose data is processed, including access to data, correction, deletion, and how these requests are handled.
- **Incident Management**: Protocols for handling security breaches or other incidents that impact personal data, detailing steps for mitigation, documentation, and compliance with legal obligations.
""")
    st.write(
"""**II. MEDICAL INFORMATION**\n
**Dental Procedures**:
- Examinations and Cleanings: Routine check-ups, dental scaling, sandblasting, and fluoridation.
- Restorative Treatments: Fillings, root canals, crowns, bridges, and veneers.
- Preventive Care: Sealing to prevent caries, dental hygiene packages.
- Cosmetic Dentistry: Teeth whitening methods (in-office and overlay), dental aesthetics like adhesion bridges and veneers.
- Prosthetics: Various types of dentures (skeletal, flexible, acrylic).

**Dermatological Treatments**:
- Skin Screening: Dermoscopy for moles, biopsy and histopathological exams.
- Minor Surgical Procedures: Cryosurgery, curettage for removing small skin lesions.
- Hair and Scalp Analysis: Trichoscopy for diagnosing hair loss types.
- Allergy Testing: Patch tests for contact allergies, PRICK tests for immediate allergies.

**Cosmetic Surgeries**:
- Breast Modifications: Augmentation, lifts, reductions.
- Body Contouring: Tummy tucks, liposuction, thigh and arm lifts.
- Facial Cosmetic Surgeries: Rhinoplasty, face lifts, eyelid surgeries, neck lifts, and otoplasty.
- Obesity Treatments: Gastric balloon and OverStitch™ stomach stitching.
- Gynecomastia: Male breast reduction.
- Other Procedures: Buttock implants, chin corrections.
- Post-op / recovery protocols recommendations for cosmetic surgeries
""")

# if user logged in and belongs to group group3
if st.session_state["authenticated"] and ("admin" in st.session_state["user_cognito_groups"] or "user" in st.session_state["user_cognito_groups"]):
    main()
else:
    if st.session_state["authenticated"]:
        st.write("You do not have access. Please contact administrator.")
    else:
        st.write("Please login first.")
