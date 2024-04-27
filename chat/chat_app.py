"""
Filename: chat_app.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: This is the main screen of the app called from 'Chatbot Assistant' menu. A Streamlit application that uses OpenAI's GPT-3.5 conversational model and alternatively Llama 2 to provide information about medical facilities, doctors, medical specialties, appointments, etc. and also allows to book a consultation.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

import streamlit as st
from st_pages import add_page_title
import auth.authenticate as authenticate
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import sqlite3
from datetime import datetime, timedelta
from chat.finders import DoctorFinder, MedicalSpecialtyFinder, TemporalDataAnalyzer, find_email_phone
from chat.chat_session import ChatSession
from chat.vector_store import VectorStore
from chat.model_prompt_engine import (
    openai_prompt_template,
    apply_openai_prompt_template,
    llama2_prompt_template,
    apply_llama2_prompt_template,
    classification_prompt_template,
    classification_with_history_prompt_template,
    extract_answer_prompt_template,
    apply_extract_answer_prompt_template,
    translate_prompt_template,
    apply_translate_prompt_template,
    make_openai_call as oryginal_make_openai_call, # we need to import as oryginal_make_openai_call to decorate it later
    tools_booking,
)

my_model_api_url = "https://wmr7p2vg7ax26x-8080.proxy.runpod.net/"  # if we use runpod.io this has this form in case of AWS it's just the IP, like "http://64.247.206.126:8080"
my_model_headers = { "Content-Type": "application/json" }  # for my cherrypy configuration
database = "./sql_db/medical_facility.db"
pinecone_index_name = 'my-medical-facility'  # name of the index in Pinecone

VECTOR_STORE_TOP_K=3  # how many top answers we want to get from the vector store
MAX_TURNS=3  # maximum number of turns in the conversation that are included in the prompt conversation history
EXTEND_ON_USER_REQUEST=14  # how many days we extend the temporal data in the session when user asks for later or earlier without giving a date
EXTEND_ON_NO_DATA=31  # period of how many days we assume when there is no temporal information in the session

VERBOSE = True  # do we debug on local terminal. If False all debug information will not be printed
VERBOSE_PROMPTS = True  # if we want to show the prompts used to call OpenAI API. If False it will turn off prompt debugging even if VERBOSE is True
VERBOSE_HISTORY = True  # if we want to show the conversation history, incl. translation. If False it will turn off debugging even if VERBOSE is True
VERBOSE_LANGUAGE_MODEL_SELECTION = True  # if we want to show the language model selection. If False it will turn off language model selection debugging even if VERBOSE is True
VERBOSE_TRANSLATION = True  # if we want to show the translation of the user_prompt. If False it will turn off translation debugging even if VERBOSE is True
VERBOSE_PROMPT_ANALYSIS = True  # if we want to show the temporal data analysis. If False it will turn off temporal data analysis debugging even if VERBOSE is True
VERBOSE_CLASSIFICATION = True  # if we want to show the classification of the user_prompt. If False it will turn off classification debugging even if VERBOSE is True
VERBOSE_VECTOR_STORE = True  # if we want to show the vector store query. If False it will turn off vector store debugging even if VERBOSE is True
VERBOSE_RAG = True  # if we want to show the RAG data. If False it will turn off RAG debugging even if VERBOSE is True
VERBOSE_FINAL_PROMPT = True  # if we want to show the final prompt. If False it will turn off final prompt debugging even if VERBOSE is True
VERBOSE_CATEGORY = True  # if we want to show the category of the user_prompt. If False it will turn off category debugging even if VERBOSE is True
VERBOSE_RECOMMENDATION = True  # if we want to log recommendations from Assistant 

SHOW_COSTS = True  # if we want to show the cost of the conversation on the screen

# Check authentication when user lands on the page.
authenticate.set_st_state_vars()

# Add login/logout buttons
if st.session_state["authenticated"]:
    authenticate.button_logout()
else:
    authenticate.button_login()

add_page_title()  # adds title and icon to current page


# helper function decorator to track token usage within the session. It is used to decorate make_openai_call function
def track_token_usage(func):
    def wrapper(*args, **kwargs):
        response_content, token_usage = func(*args, **kwargs)

        st.session_state.my_session.set_stats(token_usage['prompt_tokens'], token_usage['completion_tokens'], token_usage['model'])

        return response_content
    return wrapper
make_openai_call = track_token_usage(oryginal_make_openai_call)

def main():
    if 'my_session' not in st.session_state:
        # If session was not yet initialized, we create it here but also check if patient data was already initialized in Patient Data app. 
        # if the patient data was not initialized, they are stored in st.session_state.patient_data which is a dictionary with keys: first_name, last_name, email, phone
        if 'patient_data' not in st.session_state:
            st.session_state.my_session = ChatSession() # The ChatSession will be initialized with default values
        else:
            st.session_state.my_session = ChatSession(
                patient_first_name = st.session_state.patient_data['first_name'],
                patient_last_name = st.session_state.patient_data['last_name'],
                patient_email = st.session_state.patient_data['email'],
                patient_phone = st.session_state.patient_data['phone'],
            )
        # load the environment variables from the .env file
        _ = load_dotenv(find_dotenv()) 
        my_model_api_key = os.environ['API_SECRET']  # API to my own fine-tuned model requires key
        pinecone_api_key  = os.environ['PINECONE_API_KEY']
        openai_api_key = os.environ['OPENAI_API_KEY']

    if 'doctor_finder' not in st.session_state: 
        st.session_state.doctor_finder = DoctorFinder(database)  
    if 'specialty_finder' not in st.session_state:
        st.session_state.specialty_finder = MedicalSpecialtyFinder()  
    if 'temporal_analyzer' not in st.session_state:
        st.session_state.temporal_analyzer = TemporalDataAnalyzer()

    if 'conversational_model' not in st.session_state:
        st.session_state.conversational_model="gpt-3.5-turbo"  # "gpt-4-0125-preview"  # main conversational model. 
        st.session_state.classification_model="gpt-3.5-turbo"  # model we use to classify the question.  
        st.session_state.extraction_model="gpt-3.5-turbo"  # model we use to extract the information from the Pinecone index. 
        st.session_state.translation_model="gpt-3.5-turbo"  # model we use to translate the prompt to English. 
        st.session_state.embedding_model="text-embedding-3-small"  # model we use to embed the information into the Pinecone index.

    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=openai_api_key)
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore(pinecone_api_key, st.session_state.openai_client, pinecone_index_name, st.session_state.embedding_model)

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "All GPT-3.5"
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = "English"
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = st.session_state.my_session.get_patient_data()
    if 'clinic_data' not in st.session_state:
        st.session_state.clinic_data = "\n\nMedical facility data: " + st.session_state.my_session.get_clinic_data()
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False  # if we want to show debug info on the screen


    # Button that resets a session and restarts the chat
    st.sidebar.divider()
    if st.sidebar.button('Start over a chat'):
        st.session_state.my_session.reset_and_init_messages(st.session_state, st.session_state.selected_language)        

    # Model selector
    model_selection = st.sidebar.selectbox("Conversational model to work with:", ["All GPT-3.5", "All GPT-4", "GPT-4/GPT-3.5"], index=0)
    if model_selection == "All GPT-3.5":
        if model_selection != st.session_state.selected_model:
            st.session_state.selected_model = model_selection
            st.session_state.conversational_model="gpt-3.5-turbo"
            st.session_state.classification_model="gpt-3.5-turbo"
            st.session_state.extraction_model="gpt-3.5-turbo"
            st.session_state.translation_model="gpt-3.5-turbo"
            st.session_state.embedding_model="text-embedding-3-small"
            st.session_state.my_session.reset_and_init_messages(st.session_state, st.session_state.selected_language)        
    elif model_selection == "All GPT-4":
        if model_selection != st.session_state.selected_model:
            st.session_state.selected_model = model_selection
            st.session_state.conversational_model="gpt-4-0125-preview"
            st.session_state.classification_model="gpt-4-0125-preview"
            st.session_state.extraction_model="gpt-4-0125-preview"
            st.session_state.translation_model="gpt-4-0125-preview"
            st.session_state.embedding_model="text-embedding-3-small"
            st.session_state.my_session.reset_and_init_messages(st.session_state, st.session_state.selected_language)
    elif model_selection == "GPT-4/GPT-3.5":
        if model_selection != st.session_state.selected_model:
            st.session_state.selected_model = model_selection
            st.session_state.conversational_model="gpt-4-0125-preview"
            st.session_state.classification_model="gpt-4-0125-preview"
            st.session_state.extraction_model="gpt-3.5-turbo"
            st.session_state.translation_model="gpt-3.5-turbo"
            st.session_state.embedding_model="text-embedding-3-small"
            st.session_state.my_session.reset_and_init_messages(st.session_state, st.session_state.selected_language)
    elif model_selection == "Llama 2":
        pass
    else:
        print("Error: Unknown model selected.")
    if VERBOSE and VERBOSE_LANGUAGE_MODEL_SELECTION:
        print(f"0a. Selected models: Conversational: {st.session_state.conversational_model}, Classification: {st.session_state.classification_model}, Extraction: {st.session_state.extraction_model}, Translation: {st.session_state.translation_model}, Embedding: {st.session_state.embedding_model}")

    # language selector
    prev_lang = st.session_state.selected_language
    st.session_state.selected_language = st.sidebar.selectbox("Language:", ["English", "Polish", "German"], index=0)
    if prev_lang != st.session_state.selected_language:
        # we need to reset the session and initialize welcome message if the language was changed
        st.session_state.my_session.reset_and_init_messages(st.session_state, st.session_state.selected_language)
        if VERBOSE and VERBOSE_LANGUAGE_MODEL_SELECTION:
            print(f"0b. Selected language: {st.session_state.selected_language}")

    # # Patient data:
    # st.sidebar.text(f"{st.session_state.my_session.get_patient_data()}")

    # print the welcome message in the selected language
    st.session_state.my_session.display_title(st, st.session_state.selected_language)

    # a check box to show or not debug info
    v_show_debug = st.sidebar.checkbox("Show debug info", value=st.session_state.show_debug)
    if v_show_debug != st.session_state.show_debug:
        st.session_state.show_debug = v_show_debug
        st.rerun()

    if st.session_state.show_debug:
        # sidebar styling
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 750px;
            max-width: 950px;
        }
        """,
        unsafe_allow_html=True,
        ) 

    # initial message in the chat
    if "messages" not in st.session_state:
        # We need initialize welcome message if it wasn't done before, but without resetting the session
        st.session_state.my_session.reset_and_init_messages(st.session_state, st.session_state.selected_language, reset=False)

    # print the conversation so far
    if VERBOSE and VERBOSE_HISTORY:
        print("Original messages:")
    for msg in st.session_state.original_messages:
        if VERBOSE and VERBOSE_HISTORY:
            print(msg)
        st.chat_message(msg["role"]).write(msg["content"])

    if VERBOSE and VERBOSE_HISTORY:
        print("Messages translated into English:")
        for msg in st.session_state.messages:
            print(msg)

    # Construct conversation history for the model prompt, limiting the number of turns to MAX_TURNS
    conversation_history = ""
    turns_count = 0
    messages_processed = 0
    total_messages = len(st.session_state.messages)
    # Iterate over the messages in reverse order
    for msg in reversed(st.session_state.messages):
        messages_processed += 1

        # Check if we have reached the maximum number of turns
        if turns_count >= MAX_TURNS:
            break

        # Skip the very first assistant message in the conversation
        if messages_processed == total_messages:
            break

        # Construct the conversation history
        if msg["role"] == "user":
            conversation_history = f"Patient: {msg['content']} " + conversation_history
            # Increment the turns count after encountering a patient's response OR ...
            turns_count += 1
        else:
            conversation_history = f"Assistant: {msg['content']} " + conversation_history
            # ... OR this assistant message, that marks a turn. For now I assume that former favors better understanding of the context. So this is commented out.
            # turns_count += 1

    if VERBOSE and VERBOSE_HISTORY:
        print(f"1a. Conversation history limited to {MAX_TURNS} turns: {conversation_history}")

    # collect the question from the user
    input_prompt = st.chat_input()
    if input_prompt:
        # if the language is other than English, we need to translate the user_prompt to English
        if st.session_state.selected_language == "English":
            user_prompt = input_prompt
        else:
            translation_prompt = apply_translate_prompt_template(src_language=st.session_state.selected_language, dst_language="English", text=input_prompt)
            user_prompt = make_openai_call(st.session_state.openai_client, st.session_state.translation_model, translation_prompt, verbose=VERBOSE)
            if VERBOSE and VERBOSE_TRANSLATION:
                print(f"1b. Translation prompt: {translation_prompt}")
                print(f"1c. Translated answer: {user_prompt}\n")

        # append the question to the conversation history and display it
        st.session_state.original_messages.append({"role": "user", "content": input_prompt})
        st.chat_message("user").write(input_prompt)
        
        # Also append the question to the translated (if non-English) conversation history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        ###################################################################
        ################## Analyzing English user_prompt #######################
        # check if the user_prompt contains a question about a medical specialty
        st.session_state.my_session.set_medical_specialties(
            st.session_state.specialty_finder.find_medical_specialties(user_prompt)
        )

        # check if the user_prompt contains a question about a doctor
        st.session_state.my_session.set_doctors(
            st.session_state.doctor_finder.find_doctors(user_prompt, "")
        )

        # check if the user_prompt contains temporal data
        if VERBOSE and VERBOSE_PROMPT_ANALYSIS:
            data = st.session_state.temporal_analyzer.find_temporal_data(user_prompt)
            print(f"1f. Prompt: {user_prompt}\nTemporal data: {data}")
        st.session_state.my_session.set_temporal_data(
            st.session_state.temporal_analyzer.find_temporal_data(user_prompt), 
            data_check=True
        )
        
        if st.session_state.show_debug:
            # take session summary and display it in the sidebar
            st.sidebar.text_area(label=":orange[Data state:]", value=st.session_state.my_session.summarize_session(), height=200)

        # We want to check user prompt to see what the question is about. 
        # classification_prompt = classification_prompt_template.format(message=user_prompt)  # the old one with the last message only
        
        # history placeholder in the template is used to insert the history of the conversation. It has spacial structure we need to respect.
        # It should begin with "This is the history of the conversation between the patient and the assistant so far: ####" 
        # and end with "####\nUse the above history and the last patient message to"  
        enriched_conversation_history = f"This is the history of the conversation between the patient and the assistant so far:\n####{conversation_history}####\nUse the above history and the last patient message to"
        classification_prompt = classification_with_history_prompt_template.format(history=enriched_conversation_history, message=user_prompt)  # the new one with the conversation history

        response_json = make_openai_call(st.session_state.openai_client, st.session_state.classification_model, classification_prompt, verbose=VERBOSE, response_format={ "type": "json_object" })
        try:
            msg_class = json.loads(response_json)["category"] 
            if VERBOSE and VERBOSE_CLASSIFICATION:
                print(f"1g. Classification prompt: {classification_prompt}\nClassified into: {msg_class}")
        except Exception as e:
            msg_class = "S"
            if VERBOSE and VERBOSE_CLASSIFICATION:
                print(f"An error occurred while processing classification request: {str(e)} - assuming S category.")
        st.session_state.my_session.set_classification(msg_class)

        if VERBOSE and VERBOSE_CATEGORY:
            print(f"CATEGORY: {msg_class}")

        #################################################################
        ##### Deciding what to do based on the user_prompt analisis #####
        # get info on what is available in the session
        ms_present, ms_new, doctors_present, doctors_new, temporal_data_present, temporal_data_new = st.session_state.my_session.get_status()
        if VERBOSE and VERBOSE_PROMPT_ANALYSIS:
            print(f"3d. Session status: MS present: {ms_present}, new? {ms_new}, Doctors present: {doctors_present}, new? {doctors_new}, Temporal present: {temporal_data_present}, new? {temporal_data_new}")

        rag_data = ""
        additional_rag_rules = ""
        
        # doctors and medical specialties are absent, we want to add to rag rules suggestion to ask for medical specialties or doctors
        if not ms_present and not doctors_present:
            additional_rag_rules += "- It seams patient didn't mention any medical specialties or doctors. Maybe ask patient if she/he is looking for a specific medical specialty or doctor.\n"

        # There are 5 possible categories returned in msg_class - see classification_prompt_template above for details.
        # If the class is "G" or "M" we generally use pinceone to get the data for the model. But classification can obviously be wrong, 
        # so I also added condition that checks if we have new temporal data, medical specialties or doctors in the session.
        # if we have new data, we override classification and we go with regular SQL extraction.
        rag_extraction_data = ""
        if msg_class == "G" or msg_class == "M": 
            if VERBOSE and VERBOSE_VECTOR_STORE:
                print("4a. Using Pinecone to get the data for the model.")

            # if we are in "G" or "M" category, we want to add precautious additional rule to the prompt, so the model does not respond to unrelated questions
            additional_rag_rules += """- If the question is obviously not related to the medical field, tell the patient that you are not able to help with this topic.\n"""

            # get classification from the session
            prev_classification, _ = st.session_state.my_session.get_classification() # we are only interested in the previous classification
            if VERBOSE and VERBOSE_VECTOR_STORE:
                print(f"4b. First, checking the previous classification: {prev_classification}")
            # if previous user_prompt classification was also G or M, we want to add previous user_prompt to the current, so that vector query will be more accurate.
            # The last user_prompt is the current user_prompt, so its index is -1. The penultimate user_prompt is Assistant's response for the previous user_prompt, so its index is -2.
            # The antepenultimate user_prompt is the User's question for the previous user_prompt, so its index is -3.
            previous_patient_prompt = ""
            if prev_classification == "G" or prev_classification == "M":
                if len(st.session_state.messages) >= 3:
                    previous_patient_prompt = st.session_state.messages[-3]["content"]
                    if VERBOSE and VERBOSE_VECTOR_STORE:
                        print(f"4c. Combined current and previous user_prompt that will be sent to Pinecone: \n{(previous_patient_prompt + ' ' + user_prompt).strip()}\n")

            # get the top_k answers from the Pinecone index. If previous user_prompt was also G or M, we add it to the current user_prompt to get more accurate answers.  
            answers = st.session_state.vector_store.query((previous_patient_prompt + ' ' + user_prompt).strip(), "general" if msg_class == "G" else "medical", top_k=VECTOR_STORE_TOP_K)
            answers_str = "\n********\n".join(answers)
            if answers_str:
                extraction_prompt = apply_extract_answer_prompt_template(question=user_prompt, information=answers_str)
                extraction_answer = make_openai_call(st.session_state.openai_client, st.session_state.extraction_model, extraction_prompt, verbose=VERBOSE)
                if VERBOSE and VERBOSE_VECTOR_STORE:
                    print(f"4d. Extraction prompt:\n {extraction_prompt}\nExtraction answer: {extraction_answer}\n")
                if extraction_answer.strip() == "NONE" or extraction_answer.strip() == "":
                    rag_extraction_data = ""
                else:
                    rag_extraction_data = extraction_answer 

        # if the message is "D" category - user wants to extend period of time we are looking at
        if msg_class == "D":
            if VERBOSE and VERBOSE_RAG:
                print(f"5a. Extending the temporal data in the session in 2 directions by {EXTEND_ON_USER_REQUEST} days.")
                print(f"5b. Current temporal data in the session: {st.session_state.my_session.get_temporal_data(debug=True)}") 
            # extend the current temporal data in the session in 2 directions by EXTEND_ON_USER_REQUEST days. We do it in case a user asks for anything later or earlier, 
            # but without specific date or time, so the 'temporal_analyzer' we called before didn't catch it.
            st.session_state.my_session.extend_temporal_data(EXTEND_ON_USER_REQUEST)              
            if VERBOSE and VERBOSE_RAG:
                print(f"5.c Temporal data after extention: {st.session_state.my_session.get_temporal_data(debug=True)}") 

        # if the message is "C" category - user wants to change, reschedule or cancel an appointment
        if msg_class == "C":
            # We ask AI through additional_rag_rules to instruct patient to contact the facility directly to change, reschedule or cancel an appointment.
            additional_rag_rules += f"- If patient wants to change, reschedule or cancel an appointment, please instruct patient to contact the facility directly and give medical facility's phone and email. {st.session_state.clinic_data}\n"

        # if the message is "B" category - user wants to book an appointment
        if msg_class == "B":
            missing_data, v_message = st.session_state.my_session.get_missing_patient_data()  # we want to add a rule to ask for missing patient data
            print(f"Kategoria B, missing_data: {missing_data}")            
            # if we have any missing data, we check if we got email or phone number in the current patient's message; we don't assume first or last name might be missing
            if missing_data:
                # TODO: if we assume that first and last name are always present,the 'if' below is not needed and its content should execute unconditionally
                if "email" in missing_data or "phone" in missing_data: 
                    email, phone = find_email_phone(user_prompt)  # returned values are either None or the found email and/or phone number
                    if email or phone:  # we need to write down what we have and analyze what is still missing
                        st.session_state.my_session.set_patient_data(None, None, email, phone)
                        missing_data, v_message = st.session_state.my_session.get_missing_patient_data() 
                additional_rag_rules += v_message
                # TODO: this might be a good place to force "B" category for the next turn, but we need to be ware not to create a loop
            else: # if no missing data, we want to add hint for a model to use tools
                additional_rag_rules += "- Very important: if you have all necessary patient data, use 'book_appointment' tool to book an appointment. Do not confirm booking unless you use this tool! Check if date and hour confirmed by the patient match available slots.\n"

        # if the message is "R" category - patient describes his condition wants to get recommendation which doctor to visit
        if msg_class == "R":
            additional_rag_rules += "- If patient describes her/his condition and seeks recommendation which doctor to visit, use your knowledge to instruct patient what medical specialty to choose. Don't asks for medical details, but use general knowledge to recommend the best medical specialty.\n"

        if msg_class == "N":
            # if we have "N" category, we want to add precautious additional rule to the prompt, so the model does not respond to unrelated questions
            additional_rag_rules += f"- If the question is not related to the medical field, tell the patient that you are not able to help with this topic. Alternatively patient may contact the clinic directly {st.session_state.clinic_data}\n"

        # "S" (we need to reach out to SQL DB), but we always do the SQL extraction. 
        #######################################################################
        ###### Getting SQL DB data based on the previous user_prompt analisis ######
        # Connect to the SQLite database
        connection = sqlite3.connect(database)  # TODO: analyze if connection should be outside of the main function

        # get temporal data as they will be need for like every case below. 
        # If they are not available, we assume data_from = today and date_to = today + EXTEND_ON_NO_DATA days
        if temporal_data_present:
            temporal_query = st.session_state.my_session.get_temporal_data()
        else:
            today = datetime.now().date()
            date_from = today.strftime('%Y-%m-%d')  # Explicitly format the date, even though date() returns a string in the format 'YYYY-MM-DD'
            date_to = (today + timedelta(days=EXTEND_ON_NO_DATA)).strftime('%Y-%m-%d')  # Explicitly format the date
            temporal_query = f" AND date >= '{date_from}' AND date <= '{date_to}'"
            st.session_state.my_session.set_temporal_data((None, None, today, (today + timedelta(days=EXTEND_ON_NO_DATA))))                
        if VERBOSE and VERBOSE_RAG:
            print(f"6. Temporal data empty, so adding temporal conditions: {temporal_query}")

        rag_appointments = []
        rag_doctors = {} # specialty name is a key, and names of doctors are values

        if ms_present and not doctors_present:
            # In this scenarios we have medical specialty but no doctors
            # We need a) get doctors for given specialties and b) get available appointment dates for each doctor
            list_of_medical_specialties = st.session_state.my_session.get_medical_specialties()
            specialties_tuple = tuple(list_of_medical_specialties)
            if VERBOSE and VERBOSE_RAG:
                print(f"7. MS present and doctors not present. List of medical specialties: {list_of_medical_specialties}")

            # Handling the case where the list is empty or contains only one item
            if not list_of_medical_specialties:
                # Handle the case where there are no medical specialties even though ms_present is True. 
                # This could be returning an error, or skipping the database query
                print("Warning: Possible error as no medical specialties present in the session even though ms_present is True.")
            elif len(list_of_medical_specialties) == 1:
                # For a single item, avoid the trailing comma
                query_part = f"('{list_of_medical_specialties[0]}')"
            else:
                # For multiple items, convert the list to a tuple directly
                query_part = str(specialties_tuple)

            query_doctors = f"""
                SELECT m.name,
                    ds.specialty_id,
                    d.first_name,
                    d.last_name,
                    ds.price,
                    ds.currency
                FROM medical_specialties m, doctors_specialties ds, doctors d
                WHERE m.name IN {query_part} AND ds.specialty_id = m.id AND d.id = ds.doctor_id
            """
            if VERBOSE and VERBOSE_RAG:
                print(f"8a. Query for doctors: {query_doctors}")

            # Execute the query only if list_of_medical_specialties is not empty
            if list_of_medical_specialties:
                cursor = connection.cursor()
                cursor.execute(query_doctors)
                results_doctors = cursor.fetchall()
                cursor.close()
                
                # row[0] - medical specialty name, row[1] - id of doctors specialties (this is what we looking for to get appointment dates), 
                # row[2] and row[3] - first and last name of the doctor, row[4] - price, row[5] - currency
                # let's first build a list of doctors for medical specialties
                for row in results_doctors:
                    if row[0] in rag_doctors:
                        rag_doctors[row[0]].append(f"Dr. {row[2]} {row[3]}")
                    else:
                        rag_doctors[row[0]] = [f"Dr. {row[2]} {row[3]}"]
                if VERBOSE and VERBOSE_RAG:
                    print(f"8b. Doctors for medical specialties: {rag_doctors}")

                # Then let's build a list of possible appointments for each doctor
                cursor = connection.cursor()
                for row in results_doctors:
                    query_apps = """
                        SELECT date, time, status
                        FROM appointment_slots
                        WHERE doctor_specialty_id = ? AND status = 'available'""" + temporal_query
                    if VERBOSE and VERBOSE_RAG:
                        print(f"8c. Query appointments 1: {query_apps}")
                    cursor.execute(query_apps, (row[1],))
                    results_apps = cursor.fetchall()
                    for appointment in results_apps:
                        # Specialty name Dr. first and second name: status, appointment date and time (day of week), price and currency
                        # First, calculate the day of the week
                        appointment_date = datetime.strptime(appointment[0], '%Y-%m-%d')
                        day_of_week = appointment_date.strftime("%A")
                        rag_appointments.append(f"{row[0]} Dr. {row[2]} {row[3]}: {appointment[2]} {day_of_week} {appointment[0]} at {appointment[1]}. Cost {int(row[4])} {row[5]}")
                cursor.close()

        elif ms_present and doctors_present:
            # In this scenarios we select doctors that have specialization that is on the list of medical specialties
            list_of_medical_specialties = st.session_state.my_session.get_medical_specialties()  #First, get the list of medical specialties
            list_of_doctors = st.session_state.my_session.get_doctors()  # Then, get the list of doctors
            # Then construct a filtered list of doctors by selecting only those doctors from list_of_doctors that have 'Specialty name' in list_of_medical_specialties
            filtered_list_of_doctors = [
                (doctor['First name'], doctor['Last name'], doctor['Doctor specialty ID'], doctor['Price'], doctor['Currency'], doctor['Specialty name'])
                for doctor in list_of_doctors
                if doctor['Specialty name'] in list_of_medical_specialties
            ]

            # Based on filtered list of doctors, we can build a list of doctors for rag
            for doctor in filtered_list_of_doctors:
                if doctor[5] in rag_doctors:
                    rag_doctors[doctor[5]].append(f"Dr. {doctor[0]} {doctor[1]}")
                else:
                    rag_doctors[doctor[5]] = [f"Dr. {doctor[0]} {doctor[1]}"]
            if VERBOSE and VERBOSE_RAG:
                print(f"9a. Doctors for rag: {rag_doctors}")

            # Let's build a list of possible appointments for each filtered doctor
            cursor = connection.cursor()
            for doctor in filtered_list_of_doctors:
                query_apps = """
                    SELECT date, time, status
                    FROM appointment_slots
                    WHERE doctor_specialty_id = ? AND status = 'available'""" + temporal_query
                if VERBOSE and VERBOSE_RAG:
                    print(f"9b. Query appointments 2: {query_apps}")
                cursor.execute(query_apps, (doctor[2],))
                results_apps = cursor.fetchall()
                for appointment in results_apps:
                    # Specialty name Dr. first and second name: status, appointment date and time (day of week), price and currency
                    # First, calculate the day of the week
                    appointment_date = datetime.strptime(appointment[0], '%Y-%m-%d')
                    day_of_week = appointment_date.strftime("%A")                    
                    rag_appointments.append(f"{doctor[5]} Dr. {doctor[0]} {doctor[1]}: {appointment[2]} {day_of_week} {appointment[0]} at {appointment[1]}. Cost {int(doctor[3])} {doctor[4]}")
            cursor.close()        

        elif not ms_present and doctors_present:
            # In this scenarios we analyze all doctors that are in the session and get available appointments for each of them
            list_of_doctors = st.session_state.my_session.get_doctors()  # Then, get the list of doctors
            # every entry in list_of_doctors is in the form: 
            # {'Doctor ID': 32, 'First name': 'Linda', 'Last name': 'Lewis', 'Specialty ID': 32, 'Specialty name': 'diabetologist', 'Doctor specialty ID': 32, 'Price': 180.0, 'Currency': 'USD'}

            # Based on list of doctors, we can build a list of doctors for rag
            for doctor in list_of_doctors:
                if doctor['Specialty name'] in rag_doctors:
                    rag_doctors[doctor['Specialty name']].append(f"Dr. {doctor['First name']} {doctor['Last name']}")
                else:
                    rag_doctors[doctor['Specialty name']] = [f"Dr. {doctor['First name']} {doctor['Last name']}"]
            if VERBOSE and VERBOSE_RAG:
                print(f"10a. Doctors for rag: {rag_doctors}")

            # Let's build a list of possible appointments for each doctor
            cursor = connection.cursor()
            for doctor in list_of_doctors:
                query_apps = """
                    SELECT date, time, status
                    FROM appointment_slots
                    WHERE doctor_specialty_id = ? AND status = 'available'""" + temporal_query
                if VERBOSE and VERBOSE_RAG:
                    print(f"10b. Query appointments 3: {query_apps}")
                cursor.execute(query_apps, (doctor['Doctor specialty ID'],))
                results_apps = cursor.fetchall()
                for appointment in results_apps:
                    # Specialty name Dr. first and second name: status, appointment date and time (day of the week), price and currency
                    # First, calculate the day of the week
                    appointment_date = datetime.strptime(appointment[0], '%Y-%m-%d')
                    day_of_week = appointment_date.strftime("%A")
                    # add collected information to the rag list
                    rag_appointments.append(f"{doctor['Specialty name']} Dr. {doctor['First name']} {doctor['Last name']}: {appointment[2]} {day_of_week} {appointment[0]} at {appointment[1]}. Cost {int(doctor['Price'])} {doctor['Currency']}")
            cursor.close()

        elif not ms_present and not doctors_present:
            pass  # in this case we do nothing

        # we don't need the connection anymore, so we can close it
        connection.close()

        # We build a final rag information for doctors. For each element in rag_doctors we have a list of doctors for a given specialty
        rag_doctors_data = ""
        for specialty, doctors in rag_doctors.items():
            rag_doctors_data += f"{specialty}: {', '.join(doctors)}\n"
        # if rag_doctors_data is not empty, we add a header to it
        if rag_doctors_data:
            rag_doctors_data = "Doctors available:\n" + rag_doctors_data

        # We build a final rag information for appointments
        rag_appointments_data = "\n".join(rag_appointments)
        if rag_appointments_data == "":
            # We don't want to show model "No appointments available.", if we havn't looked for any appointments yet.
            # It usually happens at the beginning of the conversation, when we don't have any data yet.
            if ms_present or doctors_present:
                rag_appointments_data = "No appointments available."
        else:
            # if number of available appointments is greater or equal to 3, we add a suggestion that model may show patient more that one or two
            # On 22.03.2024 I eventually gave up on this idea, as it clutters the prompt and is not very useful.
            # if len(rag_appointments) >= 3:
            #     additional_rag_rules += "- if patient asks for appointments, and there are more than just 1 or 2 available, show patient more available appointments\n"
            rag_appointments_data = "Available appointments:\n" + rag_appointments_data
    
        # At the end we want to infor our model what is the current date-time
        if rag_extraction_data:
            rag_data = rag_data + rag_extraction_data + "\n"
        if rag_doctors_data:
            rag_data = rag_data + rag_doctors_data + "\n"
        if rag_appointments_data:
            rag_data = rag_data + rag_appointments_data + "\n"
        rag_data = rag_data + "Now is " + datetime.now().strftime("%Y-%m-%d (%A %H:%M)") 

        # based on the classification of prompt, we decide if we want to add clinic data or not
        # for now we add them only when patient wants to book an appointment (class B)
        # if patient asks explicitly for clinic data, they will be provided anyway using Pinecone
        if msg_class == "B":
            # Chenge 21.03.2024: we don't put clinic data to RAG anymore, as they are mistaken for patient. They should be provided by Pinecone anyway, if necessary. 
            # clinic_data = st.session_state.clinic_data
            clinic_data = ""
            tools_param = {"tools": tools_booking}
        else:
            clinic_data = ""
            tools_param = {}

        if st.session_state.show_debug:
            st.sidebar.text(f"CATEGORY: {msg_class}")
            st.sidebar.text_area(label=":orange[RAG:]", value=rag_data, height=300)
            st.sidebar.text_area(label=":orange[RULES:]", value=additional_rag_rules, height=400)

        if VERBOSE and VERBOSE_RAG:
            print(f"11. Final RAG data:\n{rag_data}\n")
        

        ###################################################################
        ### Depending on the model chosen, we generating final user_prompt, 
        ### that includes RAG and conversation history. The query model 
        ###################################################################
        if VERBOSE and VERBOSE_LANGUAGE_MODEL_SELECTION:
            print(f"12a. Selected model: {st.session_state.selected_model}")
        if st.session_state.selected_model in ["All GPT-3.5", "All GPT-4", "GPT-4/GPT-3.5"]:
            final_openai_prompt = apply_openai_prompt_template(
                    conversation=conversation_history, 
                    question=user_prompt,
                    rag=rag_data,
                    rules=additional_rag_rules,
                    patient_data=st.session_state.patient_data,
                    clinic_data=clinic_data,
            )
            if VERBOSE and VERBOSE_FINAL_PROMPT:
                print(f"12b. Final OpenAI prompt:\n {final_openai_prompt}")
            response = make_openai_call(st.session_state.openai_client, st.session_state.conversational_model, final_openai_prompt, temperature=0.5, verbose=VERBOSE_FINAL_PROMPT, **tools_param)
            if response.strip() == "":
                msg = {"role": "assistant", "content": "Assistant response failed or was cancelled."}    
            else:
                msg = {"role": "assistant", "content": response}
            if VERBOSE and VERBOSE_FINAL_PROMPT:
                print(f"12c. OpenAI response: {response}\n")

        elif st.session_state.selected_model == "Llama 2":
            final_llama2_prompt = apply_llama2_prompt_template(
                    conversation=conversation_history, 
                    question=user_prompt,
                    rag=rag_data,
                    rules=additional_rag_rules,
                    patient_data=st.session_state.patient_data,
                    clinic_data=clinic_data,
            )
            if VERBOSE and VERBOSE_FINAL_PROMPT:
                print(f"12c. Final llama 2 prompt:\n {final_llama2_prompt}\n")

            # My Llama 2 model expects the prompt in the JSON format. We also need to include the API key in the request.
            prompt_data = {"key": my_model_api_key, "prompt": final_llama2_prompt }
            prompt_json = json.dumps(prompt_data)

            try:
                response = requests.post(my_model_api_url, data=prompt_json, headers=my_model_headers)
                if response.status_code == 200: 
                    # create the response in the format expected by Streamlit
                    msg = {"role": "assistant", "content": response.text}
                else: 
                    # unless something went wrong
                    if VERBOSE and VERBOSE_FINAL_PROMPT:
                        print(f"Request failed with status code {response.status_code}: {response.reason}")
                    msg = {"role": "assistant", "content": "Assistant response failed or was cancelled."}
            except Exception as e:
                err_msg = f"An error occurred while processing the request: {str(e)}"
                if VERBOSE and VERBOSE_FINAL_PROMPT:
                    print(err_msg)
                msg = {"role": "assistant", "content": err_msg } # TODO: decide if we want to show this message to the user
        else:
            err_msg = f"Selected model {st.session_state.selected_model} not recognized."
            msg = {"role": "assistant", "content": err_msg }
            if VERBOSE and VERBOSE_LANGUAGE_MODEL_SELECTION:
                print(err_msg)

        # if current class is 'R', so we have recommendation, we want to check if assistant didn't recommend any specialty and if yes, we want to add it to the session.
        if msg_class == "R":
            recommended_specialties = st.session_state.specialty_finder.find_medical_specialties(msg["content"])
            if recommended_specialties:
                if VERBOSE and VERBOSE_RECOMMENDATION:
                    print(f"13. Recommended specialties: {recommended_specialties}")
                st.session_state.my_session.set_medical_specialties(recommended_specialties)

        # append the response to the conversation history and display it
        st.session_state.messages.append(msg)

        # if language is other than English, we need to translate the response to the selected language
        if st.session_state.selected_language == "English":
            translated_response = msg["content"]
            st.session_state.original_messages.append({"role": "assistant", "content": translated_response})
        else:
            translation_prompt = apply_translate_prompt_template(src_language="English", dst_language=st.session_state.selected_language, text=msg["content"])
            translated_response = make_openai_call(st.session_state.openai_client, st.session_state.translation_model, translation_prompt, verbose=False)
            st.session_state.original_messages.append({"role": "assistant", "content": translated_response})
            if VERBOSE and VERBOSE_TRANSLATION:
                print(f"13a. Translation prompt: {translation_prompt}\nTranslated answer: {translated_response}\n")

        st.chat_message("assistant").write(translated_response)

    # Update and display patient data:
    st.session_state.patient_data = st.session_state.my_session.get_patient_data()
    st.sidebar.text(f"{st.session_state.patient_data}")

    # Costs statistics
    if SHOW_COSTS:
        prompt_tokens_gpt3, completion_tokens_gpt3, cost_gpt3, prompt_tokens_gpt4, completion_tokens_gpt4, cost_gpt4 = st.session_state.my_session.get_stats()
        cost_display = (
            f"COSTS:\n"
            f"GPT-3.5 costs: ${cost_gpt3:.6f}\n(input: {prompt_tokens_gpt3} | output: {completion_tokens_gpt3})\n\n" 
            f"GPT-4 costs: ${cost_gpt4:.6f}\n(input: {prompt_tokens_gpt4} | output: {completion_tokens_gpt4})\n\n"
            f"Total costs: ${cost_gpt3 + cost_gpt4:.6f}"
        )
        st.sidebar.text(cost_display)


# if user logged in and belongs to group group2
if st.session_state["authenticated"] and ("admin" in st.session_state["user_cognito_groups"] or "user" in st.session_state["user_cognito_groups"]):
    main()
else:
    if st.session_state["authenticated"]:
        st.write("You do not have access to this page. Please contact administrator.")
    else:
        st.write("Please login first.")
