"""
Filename: view_sql_db.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: This is a page to view the SQL database for 'My Medical Facility'. It is called from 'Doctors & Appointments' menu option. It shows medical specialties, doctors, and appointment slots.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

import streamlit as st
from streamlit_calendar import calendar
import auth.authenticate as authenticate
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from st_pages import add_page_title

# database address
database = "./sql_db/medical_facility.db"


def create_connection(db_file):
    """
    Create a database connection to a SQLite database.

    Parameters:
    db_file (str): The path to the SQLite database file.

    Returns:
    conn (sqlite3.Connection): The database connection object.

    Raises:
    Exception: If an error occurs while connecting to the database.
    """    
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)
    return conn


def get_medical_specialties(conn):
    """
    Retrieves a list of medical specialties from the database.

    Parameters:
    - conn: A database connection object.

    Returns:
    - A list of medical specialties.
    """
    query = "SELECT name FROM medical_specialties ORDER BY LOWER(name);"
    specialties = pd.read_sql_query(query, conn)
    return specialties['name'].tolist()


def get_doctors_by_specialty(conn, specialty_name):
    """
    Retrieve a list of doctors by specialty name from the database.

    Args:
        conn (Connection): The database connection object.
        specialty_name (str): The name of the specialty.

    Returns:
        DataFrame: A pandas DataFrame containing the doctors' information.

    """
    query = """
    SELECT m.name as "Specialty Name                                    ",
           d.first_name as "Doctor's First Name      ",
           d.last_name as "Doctor's Last Name      "
    FROM medical_specialties m
    JOIN doctors_specialties ds ON ds.specialty_id = m.id
    JOIN doctors d ON d.id = ds.doctor_id
    WHERE m.name = ?;
    """
    doctors = pd.read_sql_query(query, conn, params=(specialty_name,))
    return doctors


def get_doctors(conn):
    """
    Retrieves a list of doctors sorted by last name.

    Args:
        conn (connection): The database connection object.

    Returns:
        DataFrame: A pandas DataFrame containing the full names and IDs of the doctors, sorted by last name.
    """
    query = "SELECT last_name || ', ' || first_name AS full_name, id FROM doctors ORDER BY last_name;"
    doctors = pd.read_sql_query(query, conn)
    return doctors


def get_specialties_by_doctor(conn, doctor_id):
    """
    Retrieve specialties of selected doctor.

    Parameters:
    conn (connection): The database connection object.
    doctor_id (int): The ID of the doctor.

    Returns:
    pandas.DataFrame: A DataFrame containing the specialties of the doctor.
    """
    query = """
    SELECT m.name as "Specialty name                                    "
    FROM medical_specialties m
    JOIN doctors_specialties ds ON m.id = ds.specialty_id
    WHERE ds.doctor_id = ?;
    """
    specialties = pd.read_sql_query(query, conn, params=(doctor_id,))
    return specialties


def get_doctors_by_specialty(conn, specialty_name):
    """
    Retrieve a list of doctors by specialty name.

    Args:
        conn (connection): The database connection object.
        specialty_name (str): The name of the specialty.

    Returns:
        pandas.DataFrame: A DataFrame containing the full names and IDs of doctors with the specified specialty,
        ordered by last name.

    """
    query = """
    SELECT d.last_name || ', ' || d.first_name AS full_name, ds.id
    FROM medical_specialties m
    JOIN doctors_specialties ds ON ds.specialty_id = m.id
    JOIN doctors d ON d.id = ds.doctor_id
    WHERE m.name = ?
    ORDER BY d.last_name;
    """
    doctors = pd.read_sql_query(query, conn, params=(specialty_name,))
    return doctors


def get_appointments_by_ds_id(conn, ds_id):
    """
    Retrieves appointments from the appointment_slots table based on the given doctor specialty ID.

    Args:
        conn (connection): The database connection object.
        ds_id (int): The doctor specialty ID.

    Returns:
        appointments (DataFrame): A DataFrame containing the appointments with columns 'id', 'date', 'time', and 'status'.
    """
    query = """
    SELECT id, date, time, status
    FROM appointment_slots
    WHERE doctor_specialty_id = ?;
    """
    appointments = pd.read_sql_query(query, conn, params=(ds_id,))
    return appointments


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
    st.write("""RAG can access factual and up-to-date information through a traditional Relational Database Management System (RDBMS), as outlined in this section, or a vector store - a specialized database designed to handle large volumes of diverse, often unstructured data - as described in 'Procedures & Policies' section in the menu on the left.""")
    st.write(
"""In this setup, an RDBMS is employed to store, search, and retrieve information about: Medical Specialties and Doctors available at 'My Medical Facility', as well as the Appointments calendar for each doctor.
""")
    st.write("You may browse data by switching between options of the sidebar selector on the left.")
    st.write(
"""**Please note: this is a sandbox environment, data is static and for demonstration purposes only. All data is synthetic and generated for simulation; it does not reflect real-world conditions. Feel free to interact with the Chatbot Assistant, and if needed, validate chatbot responses with the data displayed here.**
""")
    
    # Add Streamlit divider
    st.sidebar.divider()
    
    # Option selection in the sidebar
    option = st.sidebar.radio('Doctors & Appointments options:', ('Medical Specialties', 'Doctor Specialties', 'Appointment Calendar'))

    if option == 'Medical Specialties':
        st.header('Medical Specialties')
        
        # Connect to the database
        conn = create_connection(database)
        
        if conn is not None:
            # Get and display the selector for medical specialties
            specialties = get_medical_specialties(conn)
            selected_specialty = st.selectbox("Select a Medical Specialty:", specialties)
            
            if selected_specialty:
                # Retrieve and display doctors for the selected specialty
                doctors_df = get_doctors_by_specialty(conn, selected_specialty)
                if not doctors_df.empty:
                    st.write(doctors_df)
                else:
                    st.write("No doctors found for this specialty.")
            conn.close()
        else:
            st.error("Error! Unable to connect to the database.")

    elif option == 'Doctor Specialties':
        st.header("Doctors' Specialties")
        # Connect to the database
        conn = create_connection(database)  # Ensure this is defined elsewhere
        
        if conn is not None:
            # Get and display the selector for doctors
            doctors_df = get_doctors(conn)
            # Create a dictionary mapping full name to id for use in the selector
            doctors_dict = pd.Series(doctors_df.id.values, index=doctors_df.full_name).to_dict()
            selected_doctor_name = st.selectbox("Select a Doctor:", doctors_dict.keys())
            
            if selected_doctor_name:
                # Retrieve and display specialties for the selected doctor
                selected_doctor_id = doctors_dict[selected_doctor_name]
                specialties_df = get_specialties_by_doctor(conn, selected_doctor_id)
                if not specialties_df.empty:
                    st.write(specialties_df)
                else:
                    st.write("No specialties found for this doctor.")
            conn.close()
        else:
            st.error("Error! Unable to connect to the database.")

    elif option == 'Appointment Calendar':
        st.header('Appointments Calendar')
 
        conn = create_connection(database)
        if conn is not None:
            # Selector for medical specialties
            specialties = get_medical_specialties(conn)
            selected_specialty = st.selectbox("Select a Medical Specialty:", specialties)
            # Get and display the selector for doctors based on the selected specialty
            if selected_specialty:
                doctors_df = get_doctors_by_specialty(conn, selected_specialty)
                doctors_dict = pd.Series(doctors_df.id.values, index=doctors_df.full_name).to_dict()
                selected_doctor = st.selectbox("Select a Doctor:", doctors_dict.keys())
                
                # Retrieve and display appointments for the selected doctor-specialty combination
                if selected_doctor:
                    ds_id = doctors_dict[selected_doctor]
                    appointments_df = get_appointments_by_ds_id(conn, ds_id)
                    if not appointments_df.empty:
                        st.write(appointments_df)
                        
                        # Prepare events for the calendar
                        events = []
                        for _, row in appointments_df.iterrows():
                            start_time = datetime.strptime(row['date'] + ' ' + row['time'], '%Y-%m-%d %H:%M')
                            end_time = start_time + timedelta(minutes=30)  # Assuming 30 minutes for each slot
                            doctor_last_name = selected_doctor.split(',')[0]
                            events.append({
                                "title": f"{doctor_last_name} {row['time']}-{end_time.strftime('%H:%M')}",
                                "start": start_time.strftime('%Y-%m-%dT%H:%M'),
                                "end": end_time.strftime('%Y-%m-%dT%H:%M'),
                                "color": "#FF4B4B" if row['status'] == 'booked' else ("#FFFF00" if row['status'] == 'blocked' else "#3DD56D")
                            })
                            
                        # Choose a supported calendar mode
                        # For a monthly view, you can use "dayGridMonth" which is a valid FullCalendar view type
                        calendar_mode = "dayGridMonth"  # Corrected to a supported mode

                        # Set initial date for the calendar to focus on, e.g., the current month
                        # You might need to adjust this based on your application's needs
                        initial_date = datetime.now().strftime('%Y-%m-%d')

                        # Define the calendar options
                        calendar_options = {
                            "initialView": calendar_mode,
                            "initialDate": initial_date,
                            "headerToolbar": {
                                "left": "prev,next today",
                                "center": "title",
                                "right": "dayGridMonth,timeGridWeek"
                            },
                        }

                        # Display the calendar with appointments
                        calendar(events=events, options=calendar_options, key=selected_doctor)                           

                    else:
                        st.write("No appointments found for this selection.")
            conn.close()
        else:
            st.error("Error! Unable to connect to the database.")

# if user logged in and belongs to group group2
if st.session_state["authenticated"] and ("admin" in st.session_state["user_cognito_groups"] or "user" in st.session_state["user_cognito_groups"]):
    main()
else:
    if st.session_state["authenticated"]:
        st.write("You do not have access to this page. Please contact administrator.")
    else:
        st.write("Please login first.")
