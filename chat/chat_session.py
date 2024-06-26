"""
Filename: chat_session.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: Class for handling the chat session. It is used to store the data that is being collected during the chat session.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""


from datetime import datetime, timedelta

# class for the chat session 
class ChatSession:
    """
    Class for handling the chat session. It is used to store the data that is being collected during the chat session.

    Attributes:
        medical_specialties (list): A list of medical specialties collected during the chat session.
        medical_specialties_new (bool): A flag indicating if the medical specialties have been updated.
        doctors (list): A list of doctors collected during the chat session.
        doctors_new (bool): A flag indicating if the doctors have been updated.
        temporal_date1 (datetime.date): The first temporal date collected during the chat session.
        temporal_date2 (datetime.date): The second temporal date collected during the chat session.
        temporal_date_from (datetime.date): The starting date of a temporal range collected during the chat session.
        temporal_date_to (datetime.date): The ending date of a temporal range collected during the chat session.
        temporal_data_new (bool): A flag indicating if the temporal data has been updated.
        MAX_SPECIALTIES (int): The maximum number of medical specialties that can be stored in the session.
        MAX_DOCTORS (int): The maximum number of doctors that can be stored in the session.
        patient_first_name (str): The first name of the patient.
        patient_last_name (str): The last name of the patient.
        patient_email (str): The email address of the patient.
        patient_phone (str): The phone number of the patient.
        clinic_name (str): The name of the medical facility.
        clinic_address (str): The address of the medical facility.
        clinic_phone (str): The phone number of the medical facility.
        clinic_email (str): The email address of the medical facility.
        previous_class (str): The previous class of the conversation.
        current_class (str): The current class of the conversation.
        prompt_tokens_gpt3 (int): The number of tokens used for GPT-3.5 prompts in the session.
        response_tokens_gpt3 (int): The number of tokens received from GPT-3.5 in the session.
        COST_PER_PROMPT_TOKEN_GPT3 (float): The cost per prompt token in USD for GPT-3.5 Turbo.
        COST_PER_RESPONSE_TOKEN_GPT3 (float): The cost per response token in USD for GPT-3.5 Turbo.
        prompt_tokens_gpt4 (int): The number of tokens used for prompts in the session.
        response_tokens_gpt4 (int): The number of tokens received in the session.
        COST_PER_PROMPT_TOKEN_GPT4 (float): The cost per prompt token in USD for GPT-4 Turbo.
        COST_PER_RESPONSE_TOKEN_GPT4 (float): The cost per response token in USD for GPT-4 Turbo.

    Methods:
        set_medical_specialties(specialties): Sets the medical specialties collected during the chat session.
        get_medical_specialties(debug): Retrieves the medical specialties collected during the chat session.
        reset_medical_specialties(): Resets the medical specialties to an empty list.
        set_doctors(doctors, extend_medical_specialties): Sets the doctors collected during the chat session.
        get_doctors(debug): Retrieves the doctors collected during the chat session.
        reset_doctors(): Resets the doctors to an empty list.
        set_temporal_data(data, data_check): Sets the temporal data collected during the chat session.
        extend_temporal_data(days): Extends the temporal data by the specified number of days.
    """

    def __init__(self, max_specialties=4, max_doctors=4, patient_first_name="John", patient_last_name="Brown", patient_email=None, patient_phone=None):
        """
        Initializes a new instance of the ChatSession class.

        Args:
            max_specialties (int, optional): The maximum number of medical specialties that can be stored in the session. Defaults to 4.
            max_doctors (int, optional): The maximum number of doctors that can be stored in the session. Defaults to 4.
            patient_first_name (str, optional): The first name of the patient. Defaults to "John".
            patient_last_name (str, optional): The last name of the patient. Defaults to "Brown".
            patient_email (str, optional): The email address of the patient. Defaults to None.
            patient_phone (str, optional): The phone number of the patient. Defaults to None.
        """
        self.medical_specialties = []
        self.medical_specialties_new = False  # flag to indicate if the value has been just updated

        self.doctors = []
        self.doctors_new = False  # flag to indicate if the value has been just updated

        self.temporal_date1 = None
        self.temporal_date2 = None
        self.temporal_date_from = None
        self.temporal_date_to = None
        self.temporal_data_new = False # flag to indicate if the value has been just updated

        # max_specialties and max_doctors values need to be decided empirically. For now 4 seems to be a reasonable value that do not clutter the session too much.
        self.MAX_SPECIALTIES = max_specialties  # maximum number of medical specialties that can be stored in the session
        self.MAX_DOCTORS = max_doctors  # maximum number of doctors that can be stored in the session

        # Patient data can be collected inother app and passed to this chat sessionor they will have default values
        self.patient_first_name = patient_first_name
        self.patient_last_name = patient_last_name
        self.patient_email = patient_email
        self.patient_phone = patient_phone

        # TODO: clinic's data for tests. For now we assume that either clinic's data will be constant 
        # or will be provided by the user at some point of the chat. 
        self.clinic_name = "My Medical Facility"
        self.clinic_address = "GW68+FH London, United Kingdom"
        self.clinic_phone = "+44 20 7123 4567"
        self.clinic_email = "contact@mymedicalfacility.com"

        self.previous_class = None  # previous class of the conversation
        self.current_class = None  # current class of the conversation

        self.prompt_tokens_gpt3 = 0  # number of tokens used for gpt-3.5 prompts in a given session
        self.response_tokens_gpt3 = 0  # number of tokens received from GPT-3.5 in a given session
        self.COST_PER_PROMPT_TOKEN_GPT3 = 0.50/1_000_000  # cost per prompt token in USD for gpt-3.5-turbo-0125 (03.2024)
        self.COST_PER_RESPONSE_TOKEN_GPT3 = 1.50/1_000_000  # cost per response token in USD for gpt-3.5-turbo-0125 (03.2024)
        self.prompt_tokens_gpt4 = 0  # number of tokens used for prompts in a given session
        self.response_tokens_gpt4 = 0  # number of tokens received in a given session
        self.COST_PER_PROMPT_TOKEN_GPT4 = 10.00/1_000_000  # cost per prompt token in USD for GPT-4 Turbo (03.2024)
        self.COST_PER_RESPONSE_TOKEN_GPT4 = 30.00/1_000_000  # cost per response token in USD for GPT-4 Turbo (03.2024)


    def set_medical_specialties(self, specialties):
        """
        Sets the medical specialties for the chat session.

        Args:
            specialties (list): A list of medical specialties.

        Notes:
            - Medical specialties are kept as a list of strings.
            - If the specialties list is not empty, the method sets the value and sets the flag to True.
            - If the specialties list is empty, the method leaves the value as it is.

        Algorithm:
            - Iterate over all elements of the input parameter 'specialties'.
            - Check if each element is already present in self.medical_specialties.
            - If the element doesn't exist, add it to self.medical_specialties.
            - If self.medical_specialties exceeds the maximum number of specialties (MAX_SPECIALTIES),
                remove the first element to maintain a FIFO (First-In, First-Out) behavior.

        """        
        # medical specialties are kept as a list: ['general practitioner', 'orthopedist'], up to elements
        # if specialties are not empty, we set the value and set the flag to True, otherwise we left value as it is
        if specialties:
            # algo:
            # a) I iterate over all elements of the input parameter 'specialties'
            # b) Check if element already present in self.medical_specialties
            # c)  If yes, I do nothing, if it doesn't exist, I add it and at the same time I check if self.medical_specialties exceeds 4 elements and if yes remove the first one (so it is always MAX_SPECIALTIES elements list, in the form of FIFO)
            for specialty in specialties:
                if not self.medical_specialties or specialty not in self.medical_specialties:
                    self.medical_specialties.append(specialty)
                    if len(self.medical_specialties) > self.MAX_SPECIALTIES:
                        self.medical_specialties.pop(0) # Remove the first element to maintain FIFO
            self.medical_specialties_new = True  # Indicate that new specialties might have been added


    def get_medical_specialties(self, debug=False):
        """
        Get the medical specialties.

        Parameters:
        - debug (bool): If True, only the values of the attributes will be shown without resetting the flag.

        Returns:
        - list: The medical specialties.

        """        
        if not debug:  # if debug is True, it means we just want to see the values of the attributes, so we don't reset the flag
            self.medical_specialties_new = False  # Reset the flag after the data has been read
        return self.medical_specialties


    def reset_medical_specialties(self):
        """
        Resets the list of medical specialties and sets the flag for new specialties to False.
        
        This method clears the existing list of medical specialties and sets the flag indicating whether new specialties have been added to False.
        """
        self.medical_specialties = []
        self.medical_specialties_new = False


    def set_doctors(self, doctors, extend_medical_specialties=False):
        """
        Sets the list of doctors for the chat session.

        Args:
            doctors (list): A list of dictionaries representing the doctors.
                Each dictionary should contain the following keys:
                - 'Specialty name': The name of the doctor's specialty.
                - 'Doctor specialty ID': The ID of the doctor's specialty.

            extend_medical_specialties (bool, optional): Flag indicating whether to extend the list of medical specialties.
                If set to True, the method will add new specialties to the existing list of medical specialties.
                If set to False (default), the method will only add new specialties if they are not already present in the list.

        Returns:
            None

        Notes:
            - The method adds the doctors to the list of doctors for the chat session.
            - If the list of medical specialties needs to be extended, the method adds new specialties to the existing list,
                maintaining a maximum of MAX_SPECIALTIES elements in FIFO order.
            - If the list of doctors exceeds MAX_DOCTORS elements, the method removes the first element to maintain FIFO order.

        """
        # if doctors are not empty, we set the value and set the flag to True, otherwise we left value as it is
        if doctors:
            # a) I iterate over all elements of the input parameter 'doctors'
            # if extend_medical_specialties is True:
            #   b) Check if 'Specialty name' is already present in self.medical_specialties
            #   c) If yes, I do nothing, if it doesn't exist, I add it and at the same time I check if self.medical_specialties exceeds 4 elements,
            #      and if yes remove the first one (so it is always MAX_SPECIALTIES elements list, in the form of FIFO)
            #   The reason why extending is parametrized and False by default is that I observed that self.medical_specialties 
            #   are sometime clutters with data that is not relevant to the user's query.
            # d) check if self.doctors has the same element already present - this can be checked using 'Doctor specialty ID'.
            # e) If present, I do nothing. If it doesn't exist, I add it and at the same time I check if self.doctors exceeds MAX_DOCTORS elements and if yes remove the first one (so it is always MAX_SPECIALTIES elements list, in the form of FIFO)
            for doctor in doctors:
                specialty_name = doctor.get('Specialty name')
                doctor_specialty_id = doctor.get('Doctor specialty ID')

                # Add the specialty name to self.medical_specialties if not already present, maintaining a maximum of MAX_SPECIALTIES elements in FIFO order.
                if extend_medical_specialties and specialty_name and (not self.medical_specialties or specialty_name not in self.medical_specialties):
                    self.medical_specialties.append(specialty_name)
                    if len(self.medical_specialties) > self.MAX_SPECIALTIES:
                        self.medical_specialties.pop(0)  # Remove the first element to maintain FIFO

                # Check if the doctor is already present in self.doctors by 'Doctor specialty ID'
                existing_doctor_ids = [d.get('Doctor specialty ID') for d in self.doctors]
                if doctor_specialty_id not in existing_doctor_ids:
                    self.doctors.append(doctor)
                    # Maintain a maximum of MAX_DOCTORS elements in FIFO order for self.doctors
                    if len(self.doctors) > self.MAX_DOCTORS:
                        self.doctors.pop(0)  # Remove the first element to maintain FIFO

            self.doctors_new = True  # Indicate that new doctors might have been added         


    def get_doctors(self, debug=False):
        """
        Retrieve the list of doctors.

        Parameters:
        - debug (bool): If True, only the values of the attributes will be shown without resetting the flag.

        Returns:
        - list: The list of doctors.

        """        
        if not debug:  # if debug is True, it means we just want to see the values of the attributes, so we don't reset the flag
            self.doctors_new = False  # Reset the flag after the data has been read
        return self.doctors


    def reset_doctors(self):
        """
        Resets the list of doctors and sets the flag indicating new doctors to False.

        This function clears the existing list of doctors and sets the `doctors_new` flag to False.
        Use this function when you want to start with an empty list of doctors and indicate that there are no new doctors.

        Parameters:
        None

        Returns:
        None
        """
        self.doctors = []
        self.doctors_new = False


    def set_temporal_data(self, data, data_check=False):
        """
            Sets the temporal data attributes based on the input data.

            Args:
                    data (tuple): A tuple containing four elements representing the temporal data.
                                                The elements are in the following order:
                                                - temporal_date1: The first temporal date.
                                                - temporal_date2: The second temporal date.
                                                - temporal_date_from: The start date of the temporal range.
                                                - temporal_date_to: The end date of the temporal range.
                    data_check (bool, optional): A flag indicating whether to perform data checking.
                                                                             Defaults to False.

            Returns:
                    None

            Notes:
                    - If any of the elements in the input data is not None, the values of all attributes are set
                        and the flag `temporal_data_new` is set to True.
                    - If `data_check` is True, additional checks are performed:
                        - If `temporal_date1`, `temporal_date_from`, and `temporal_date_to` are not None,
                            and `temporal_date1` is within the range of `temporal_date_from` and `temporal_date_to`,
                            the attributes are not set, but the flag `temporal_data_new` is set to True.
                        - If `temporal_date2` is None or `temporal_date2` is within the range of `temporal_date_from`
                            and `temporal_date_to`, the attributes are not set, but the flag `temporal_data_new` is set to True.
                        - In other words, if `temporal_date1` is the only element present or both `temporal_date1` and
                            `temporal_date2` are within the range of `temporal_date_from` and `temporal_date_to`,
                            the range is not modified, but the flag `temporal_data_new` is set to True.
        """        
        # if any of element of data (the input parameter) is not None, we set the values of all attributes 
        # and set the flag to True, otherwise we left values as they are 
        if data[0] or data[1] or data[2] or data[3]:
            # data_check is a mechanism that checks:
            # if data[0], self.temporal_date_from and self.temporal_date_to are not None and data[0] is in the range of self.temporal_date_from and self.temporal_date_to
            # then we don't set the values of the attributes (we leave the range as it is) but we set the flag to True
            if data_check:
                if data[0] and self.temporal_date_from and self.temporal_date_to and self.temporal_date_from <= data[0] <= self.temporal_date_to:
                    # if data[1] is None or data[1] is not None and data[1] is in the range of self.temporal_date_from and self.temporal_date_to
                    # then we don't set the values of the attributes (we leave the range as it is) but we set the flag to True
                    # in other words if data[0] if only one present or data[0] and data[1] if two present are in the range of self.temporal_date_from and self.temporal_date_to
                    # we leave the range as it is but we set the flag to True
                    if not data[1] or (data[1] and self.temporal_date_from <= data[1] <= self.temporal_date_to):
                        self.temporal_data_new = True
                        return

            self.temporal_date1, self.temporal_date2, self.temporal_date_from, self.temporal_date_to = data
            self.temporal_data_new = True


    # this method extends the temporal data by the number of days
    def extend_temporal_data(self, days):
        """
        Extends the temporal data based on the given number of days.

        Args:
            days (int): The number of days to extend the temporal data.

        Description:
            This method extends the temporal data based on the given number of days. It updates the values of `self.temporal_date_from` and `self.temporal_date_to` 
            based on the current values of `self.temporal_date1`, `self.temporal_date2`, `self.temporal_date_from`, and `self.temporal_date_to`.

            If only `self.temporal_date1` is present, the method changes the value of `self.temporal_date1` to `None` and sets the value of `self.temporal_date_from` 
            to `self.temporal_date1 - days` and `self.temporal_date_to` to `self.temporal_date_from + days`.

            If both `self.temporal_date1` and `self.temporal_date2` are present, the method changes the values of `self.temporal_date1` and `self.temporal_date2` to `None` 
            and sets the value of `self.temporal_date_from` to `self.temporal_date1 - days` and `self.temporal_date_to` to `self.temporal_date2 + days`.

            If both `self.temporal_date_from` and `self.temporal_date_to` are present, the method changes the value of `self.temporal_date_from` to `self.temporal_date_from - days` 
            and `self.temporal_date_to` to `self.temporal_date_to + days`.

            If none of the dates are present, the method sets the value of `self.temporal_date_from` to today's date and `self.temporal_date_to` to today's date plus 14 days plus `days`.

            Note: The algorithm looks at the next `EXTEND_ON_NO_DATA` days if all dates are `None`, which is why the code includes the `14 + days` calculation. However, it is unclear if this part of the code is necessary.

        """        
        today = datetime.today().date()

        # if only self.temporal_date1 is present, we change the value of self.temporal_date1 to None and instead we set the value of 
        # self.temporal_date_from to self.temporal_date1 - days and self.temporal_date_to to self.temporal_date_from + days
        if self.temporal_date1 and not self.temporal_date2 and not self.temporal_date_from and not self.temporal_date_to:
            self.temporal_date_from = max(self.temporal_date1 - timedelta(days=days), today)  # but not earlier than today
            self.temporal_date_to = self.temporal_date_from + timedelta(days=days)
            self.temporal_date1 = None
            self.temporal_data_new = True
        # if self.temporal_date1 and self.temporal_date2 are present, we change the value of self.temporal_date1 and self.temportal_date2 to None 
        # and instead we set the value of self.temporal_date_from to self.temporal_date1 - days and self.temporal_date_to to self.temporal_date2 + days
        elif self.temporal_date1 and self.temporal_date2 and not self.temporal_date_from and not self.temporal_date_to:
            self.temporal_date_from = max(self.temporal_date1 - timedelta(days=days), today)  # but not earlier than today
            self.temporal_date_to = self.temporal_date2 + timedelta(days=days)
            self.temporal_date1, self.temporal_date2 = None, None
            self.temporal_data_new = True
        # if self.temporal_date_from and self.temporal_date_to are present, we change the value of self.temporal_date_from to self.temporal_date_from - days
        # and self.temporal_date_to to self.temporal_date_to + days
        elif self.temporal_date_from and self.temporal_date_to:
            self.temporal_date_from = max(self.temporal_date_from - timedelta(days=days), today)  # but not earlier than today
            self.temporal_date_to = self.temporal_date_to + timedelta(days=days)            
            self.temporal_data_new = True
        # if none dates are present, we set the value of self.temporal_date_from to today and self.temporal_date_to to today + 14 + days
        # why '14 + days'? Because if all dates are None, the algo looks at the next EXTEND_ON_NO_DATA (03.2024=31) days anyway 
        # Because of that, I'm not really sure if this part of the code is really necessary, but I left it for now.
        elif not self.temporal_date1 and not self.temporal_date2 and not self.temporal_date_from and not self.temporal_date_to:
            self.temporal_date_from = today
            self.temporal_date_to = today + timedelta(days=(14 + days))   
            self.temporal_data_new = True
    

    def get_temporal_data(self, debug=False, limit_dates_to_today=True):
        """
        Retrieves temporal data based on the specified date criteria.

        Args:
            debug (bool, optional): If True, only the attribute values are returned without resetting the flag. 
                Defaults to False.
            limit_dates_to_today (bool, optional): If True, limits the date range to today's date if the specified 
                date range is earlier than today. Defaults to True.

        Returns:
            str: SQL query string representing the date criteria for retrieving temporal data.
        """
        if not debug:  # if debug is True, it means we just want to see the values of the attributes, so we don't reset the flag
            self.temporal_data_new = False  # Reset the flag after the data has been read
        if self.temporal_date1 is not None and self.temporal_date2 is None:  # if we have only one date 
            return f" AND date = '{str(self.temporal_date1)}' "
        elif self.temporal_date1 is not None and self.temporal_date2 is not None:  # if we have two dates it means we talk about one of the dates
            return f" AND (date = '{str(self.temporal_date1)}' OR date = '{str(self.temporal_date2)}') "
        elif self.temporal_date_from is not None and self.temporal_date_to is not None:  # if we have date range
            v_date_from = self.temporal_date_from
            v_date_to = self.temporal_date_to

            if limit_dates_to_today:
                # if the v_date_from is earlier than today, we set it to today
                v_date_from = max(v_date_from, datetime.today().date())
                # if the v_date_to is earlier than today, we set it to today
                v_date_to = max(v_date_to, datetime.today().date())

            return f" AND date >= '{str(v_date_from)}' AND date <= '{str(v_date_to)}' "
        else:
            return ""


    def reset_temporal_data(self):
        """
        Resets the temporal data variables to their initial values.

        This function sets the `temporal_date1`, `temporal_date2`, `temporal_date_from`, and `temporal_date_to` variables to `None`,
        and sets the `temporal_data_new` flag to `False`.

        Parameters:
            None

        Returns:
            None
        """
        self.temporal_date1, self.temporal_date2, self.temporal_date_from, self.temporal_date_to = None, None, None, None
        self.temporal_data_new = False


    def summarize_session(self):
        """
        Summarizes the session by providing details about the medical specialties, doctors, and temporal data.

        Returns:
            A list of strings containing the session details. If no details are available, an empty string is returned.
        """
        details = []
        if self.medical_specialties:
            details.append(f"Specialties:\n {self.medical_specialties}.\\n**********\\n")
        if self.doctors:
            details.append(f"Doctors:\n {self.doctors}.\n**********\n")
        details.append(f"Temporal data: {self.get_temporal_data(debug=True)}.\n********************\n")
        
        if details:
            return details
        else:
            return ""


    def get_status(self):
        """
        Get the status of the chat session.

        Returns:
            A tuple containing the following information:
            - ms_present (bool): Indicates if medical specialties are present.
            - ms_new (bool): Indicates if there are new medical specialties.
            - doctors_present (bool): Indicates if doctors are present.
            - doctors_new (bool): Indicates if there are new doctors.
            - temporal_data_present (bool): Indicates if temporal data is present.
            - temporal_data_new (bool): Indicates if there is new temporal data.
        """        
        ms_present, doctors_present, temporal_data_present = False, False, False
        ms_new, doctors_new, temporal_data_new = False, False, False
        if self.medical_specialties:
            ms_present = True
            ms_new = self.medical_specialties_new
        if self.doctors:
            doctors_present = True
            doctors_new = self.doctors_new
        if self.temporal_date1 or self.temporal_date2 or self.temporal_date_from or self.temporal_date_to:
            temporal_data_present = True
            temporal_data_new = self.temporal_data_new

        return ms_present, ms_new, doctors_present, doctors_new, temporal_data_present, temporal_data_new


    def reset_session(self):
        """
        Resets the chat session by resetting the medical specialties, doctors, temporal data, and stats.
        """
        self.reset_medical_specialties()
        self.reset_doctors()
        self.reset_temporal_data()
        self.reset_stats()


    def reset_and_init_messages(self, session_state, language, reset=True):
        """
        Reset the session and initialize messages based on the selected language.

        Args:
            session_state (dict): The session state dictionary.
            language (str): The selected language.
            reset (bool, optional): Whether to reset the session. Defaults to True.
        """
        if reset:
            self.reset_session()

        # Initialize messages based on the selected language
        welcome_messages = {
            "English": "How can I help you?",
            "Polish": "W czym mogę pomóc?",
            "German": "Womit kann ich Ihnen helfen?"
        }

        session_state["messages"] = [{"role": "assistant", "content": welcome_messages["English"]}]
        session_state["original_messages"] = [{"role": "assistant", "content": welcome_messages.get(language, "How can I help you?")}]


    def display_title(self, st, language):
        """
        Display the title and description of the medical facility based on the selected language.

        Parameters:
        - st (object): Streamlit object used for displaying the title and description.
        - language (str): The selected language for the title and description.

        Returns:
        None
        """
        welcome_title = {
            "English": "Welcome to our medical facility!",
            "Polish": "Witamy w naszej placówce medycznej!",
            "German": "Willkommen in unserer medizinischen Einrichtung!"
        }

        welcome_description = {
            "English": "Talk to the assistant about booking an appointment, available treatments, pricing, policies and procedures.",
            "Polish": "Porozmawiaj z naszym asystentem aby umówić wizytę, zasięgnąć informacji na temat zabiegów, cen oraz sposobu działania naszej placówki.",
            "German": "Sprechen Sie mit dem Assistenten über die Terminvereinbarung, verfügbare Behandlungen, Preise, Richtlinien und Verfahren."
        }

        st.title(welcome_title.get(language))
        st.write(welcome_description.get(language))
    

    def get_raw_patient_data(self):
        """
        Returns the raw patient data.

        Returns: 
        A tuple containing the patient's first name, last name, email, and phone number.
        """
        return self.patient_first_name, self.patient_last_name, self.patient_email, self.patient_phone


    def get_patient_data(self):
        """
        Retrieves the patient data and returns it as a formatted string.

        Returns:
            str: The patient data including first name, last name, email, and phone number.
        """        
        patient_first_name = f"Patient first name: {self.patient_first_name}\n" if self.patient_first_name else ""
        patient_last_name = f"Patient last name: {self.patient_last_name}\n" if self.patient_last_name else ""
        patient_email = f"Patient email: {self.patient_email}\n" if self.patient_email else ""
        patient_phone = f"Patient phone number: {self.patient_phone}\n" if self.patient_phone else ""

        patient_data = patient_first_name + patient_last_name + patient_email + patient_phone

        return patient_data


    def get_missing_patient_data(self):
        """
        Checks for missing patient data and returns a dictionary of missing data fields and a message.

        Returns:
            missing_data (dict): A dictionary containing the missing data fields as keys and their descriptions as values.
            message (str): A message indicating the missing patient data fields.
        """        
        missing_data = {}
        if not self.patient_first_name:
            missing_data["first_name"] = "patient first name"
        if not self.patient_last_name:
            missing_data["last_name"] = "patient last name"
        if not self.patient_email:
            missing_data["email"] = "patient email"
        if not self.patient_phone:
            missing_data["phone"] = "patient phone number in format XXX-XXX-XXX"

        if missing_data:
            # Join the missing_data values into a single string and prepend the message.
            message = "- There are missing patient data you need to ask for before the booking can take place: " + ", ".join(missing_data.values()) + ". \n"
            return missing_data, message
        else:
            return {}, ""


    def get_clinic_data(self):
        """
        Returns a string containing the clinic's name, address, phone number, and email.

        Returns:
            str: A string containing the clinic's name, address, phone number, and email.
        """        
        return f"{self.clinic_name}, {self.clinic_address}, {self.clinic_phone}, {self.clinic_email}"


    def set_patient_data(self, first_name, last_name, email, phone):
        """
        Method to set the patient's data. 
        If any input parameter is set to None, the value is not changed.
        TODO: We might want to think about validations if data are modified; right now they are overwritten without any validation.
        """
        if first_name:
            self.patient_first_name = first_name
        if last_name:
            self.patient_last_name = last_name
        if email:
            self.patient_email = email
        if phone:
            self.patient_phone = phone


    def set_clinic_data(self, name, address, phone, email):
        pass  # TODO


    def set_classification(self, classification):
        """
        Sets the classification of the chat session.

        Parameters:
        - classification: The classification to be set.

        Returns:
        None
        """
        self.previous_class = self.current_class
        self.current_class = classification


    def get_classification(self):
        """
        Returns a tuple containing the previous classification and the current classification.

        Returns:
            tuple: A tuple containing the previous classification and the current classification.
        """        
        return (self.previous_class, self.current_class)


    def set_stats(self, prompt_tokens, response_tokens, model):
        """
        Sets the statistics for the given model.

        Args:
            prompt_tokens (list): The prompt tokens to be added to the statistics.
            response_tokens (list): The response tokens to be added to the statistics.
            model (str): The model for which the statistics are being set.

        Returns:
            None
        """        
        if model == "gpt-3.5-turbo":
            self.prompt_tokens_gpt3 += prompt_tokens
            self.response_tokens_gpt3 += response_tokens
        elif model == "gpt-4-0125-preview":
            self.prompt_tokens_gpt4 += prompt_tokens
            self.response_tokens_gpt4 += response_tokens


    def get_stats(self):
        """
        Calculates the statistics and cost for the chat session.

        Returns:
            A tuple containing the following information:
            - prompt_tokens_gpt3: The number of prompt tokens used for GPT-3.5-turbo.
            - response_tokens_gpt3: The number of response tokens used for GPT-3.5-turbo.
            - cost_gpt3: The cost of using GPT-3.5-turbo.
            - prompt_tokens_gpt4: The number of prompt tokens used for GPT-4-0125-preview.
            - response_tokens_gpt4: The number of response tokens used for GPT-4-0125-preview.
            - cost_gpt4: The cost of using GPT-4-0125-preview.
        """
        # calculate the cost for "gpt-4-0125-preview"
        cost_gpt4 = self.prompt_tokens_gpt4 * self.COST_PER_PROMPT_TOKEN_GPT4 + self.response_tokens_gpt4 * self.COST_PER_RESPONSE_TOKEN_GPT4

        # calculate the cost for "gpt-3.5-turbo"
        cost_gpt3 = self.prompt_tokens_gpt3 * self.COST_PER_PROMPT_TOKEN_GPT3 + self.response_tokens_gpt3 * self.COST_PER_RESPONSE_TOKEN_GPT3

        return self.prompt_tokens_gpt3, self.response_tokens_gpt3, cost_gpt3, self.prompt_tokens_gpt4, self.response_tokens_gpt4, cost_gpt4


    def reset_stats(self):
        """
        Resets the statistics for the chat session.

        This method sets the prompt and response token counts for both GPT-3 and GPT-4 models to zero.
        """
        self.prompt_tokens_gpt3 = 0  
        self.response_tokens_gpt3 = 0  
        self.prompt_tokens_gpt4 = 0  
        self.response_tokens_gpt4 = 0
