"""
Filename: booking.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: Implements/simulates booking process. Used as a 'function calling' target -> GPT returns parameters for this function and then we call it with those parameters. Booking simulation is done by creating a text file in 'specified_location' with the appointment details.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

import datetime
from pathlib import Path


def book_appointment(appointment_date, appointment_time, dr_first_name, dr_last_name, specialization, price, patient_first_name, patient_last_name, patient_email, patient_phone):
    """
    Book an appointment and create a text file with the appointment details.

    Parameters:
    - appointment_date (str): The date of the appointment in the format "YYYY-MM-DD".
    - appointment_time (str): The time of the appointment in the format "HH:MM".
    - dr_first_name (str): The first name of the doctor.
    - dr_last_name (str): The last name of the doctor.
    - specialization (str): The specialization of the doctor.
    - price (str): The price of the appointment.
    - patient_first_name (str): The first name of the patient.
    - patient_last_name (str): The last name of the patient.
    - patient_email (str): The email address of the patient.
    - patient_phone (str): The phone number of the patient.

    Returns:
    - str: A success message indicating that the appointment has been booked and instructing the patient to check their email for confirmation.

    Example usage:
    book_appointment("2024-03-21", "14:00", "John", "Doe", "Cardiology", "400", "Patient info here")
    """
    # Generating current date and time for file name
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"appointment_{current_time}.txt"

    specified_location = Path("./logs/")
    specified_location.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    file_path = specified_location / file_name

    # Preparing the content to write to the file
    content = f"""
        Appointment Date: {appointment_date}
        Appointment Time: {appointment_time}
        Doctor's Name: Dr. {dr_first_name} {dr_last_name}
        Specialization: {specialization}
        Price: {price}
        Patient Data: 
        First name: {patient_first_name} 
        Last name: {patient_last_name}
        Email: {patient_email}
        Phone: {patient_phone}
    """

    # Writing to the file
    with open(file_path, "w") as file:
        file.write(content)

    return f"Appointment successfully booked ({appointment_date}, {appointment_time}, {dr_first_name} {dr_last_name}, {specialization}). Please check your email to confirm booking. Is there anything else I can help you with?"
