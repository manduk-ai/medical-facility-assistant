"""
File: model_prompt_engine.py
Author: Szymon Manduk
Date: 2024-03-09

Description:
    Contains:
    - prompt templates for models / tasks used in chat app,
    - helper functions to apply prompt template
    - helper function to call OpenAI model call.
    
License:
    To be defined. Contact the author for more information on the terms of use for this software.
"""

import json
from chat.booking import book_appointment

##########################################################
#### main prompt for openai conversation intelligence ####
##########################################################
openai_prompt_template = (
f"""You are a helpful Assistant on the hotline of medical facility. Your task is to talk to patients.
Below is the history of the conversation with the patient (between four hashes):
####
{{conversation}}
####

The patient asks now another question: {{question}}

----

Here are some Additional Information you may find useful while answering patient's question (between four hashes):
####
{{rag}}

{{patient_data}}

{{clinic_data}}
####
Important rules for you to follow while answering the question:
- Respond as an Assistant. Keep your answer short and precise.
- Do not generate 'Patient:' turn, just respond as an Assistant.
- Do not make up information.
{{rules}}
####

Once again, this is the Patient's question: '{{question}}'
Respond as an Assistant.
Response: {{response}}"""
)


# helper function to apply the main prompt template
def apply_openai_prompt_template(conversation, question, rag, rules='', patient_data='', clinic_data='', prompt=openai_prompt_template): 
    return prompt.format(
            conversation=conversation,
            question=question,
            rag=rag,
            rules=rules, # optional additional rules for the LLM to follow
            patient_data=patient_data,
            clinic_data=clinic_data,
            response=''
        )

##########################################################
#### main prompt for llama2 conversation intelligence ####
##########################################################
llama2_prompt_template = (
f"""You are a helpful Assistant on the hotline of medical facility. Your task is to talk to patients.
Below is the history of the conversation with the patient (between four hashes):
####
{{conversation}}
####

The patient asks now another question: {{question}}

----

Here are some Additional Information you may find useful while answering patient's question (between four hashes):
####
{{rag}}

{{patient_data}}

{{clinic_data}}
####
Important rules for you to follow while answering the question:
- Respond as an Assistant. Keep your answer short and precise.
- Do not generate 'Patient:' turn, just respond as an Assistant.
- Do not make up information.
{{rules}}
####

Once again, this is the Patient's question: '{{question}}'
Respond as an Assistant.
Response: {{response}}"""
)

# llama2_prompt_template_old = (
# f"""You are a helpful Assistant on the hotline of a medical facility. Your task is to talk to a Patient. Below is the history of the conversation with a Patient (between four hashes): #### {{conversation}} ####
# The Patient asks now another question (below between four hashes): #### {{question}} ####
# Additional information:
# ####
# {{rag}}
# ####
# Respond as an Assistant.
# Use history of the conversation and additional information, if given, to answer the Patient's question.
# Do not make up any information. If you cannot find the answer, respond with 'I cannot answer your question. Could you please make it more precise?'.
# Keep your answer short and precise. 
# Do not continue as a Patient, just respond as an Assistant. 
# Answer the Patient's question as an Assistant.
# Response: {{response}}"""
# )

# helper function to apply the main prompt template
def apply_llama2_prompt_template(conversation, question, rag, rules='', patient_data='', clinic_data='', prompt=llama2_prompt_template): 
    return prompt.format(
            conversation=conversation,
            question=question,
            rag=rag,
            rules=rules, # optional additional rules for the LLM to follow
            patient_data=patient_data,  
            clinic_data=clinic_data,
            response=''
        )


##########################################################
# prompt to classify a current state of conversation 
# into one of 5 categories 

classification_prompt_template = (
f"""You work at a medical facility and your task is to assign the correct category to the patient message.
This is the patient message (between four hashes): 
####
{{message}} 
####
Below are the categories and rules for classification:
- 'S' category for messages about doctors, medical specializations, available dates, available appointments, specific dates, or prices.
- 'G' category for messages about general non-medical information, including: opening hours, payment methods, clinic address and phone, policies, complaints, data governance, privacy, or security. 
- 'M' category for messages about general medical information described in medical facility documentation, including: preparation for medical procedures, course of medical procedures, post-op recommendations, medical risks. But not related to dates nor time.
- 'B' category when a patient confirms an appointment booking.
- 'D' category for requests about date or time changes without specifying exact dates, e.g.: 'anything later', 'I need some earlier dates', 'following week', or similar.
- 'C' category for changing, rescheduling, or cancelling appointments.
- 'R' category when a patient briefly describes their medical problem or condition or asks for recommendations or advice on which medical specialization or doctor to visit.
- 'N' category for messages that do not fit any other category or include general greetings.
---
Once again, this is the patient message:
####
{{message}} 
####
Respond with the category of the patient message in JSON format.""")

# This another version of prompt that also uses the history of the conversation so far
# history tag is used to insert the history of the conversation. It should begin with "This is the history of the conversation between the patient and the assistant so far: ####"
# and end with "####\nUse the above history and the last patient message to"  
classification_with_history_prompt_template = (
f"""You work at a medical facility and your task is to assign the correct category to the patient message.
{{history}} classify the last patient message below. This is the last patient message (between four hashes): 
####
{{message}} 
####
Below are the categories and rules for classification:
- 'S' category for messages about doctors, medical specializations, available dates, available appointments, specific dates, or prices.
- 'G' category for messages about general non-medical information, including: opening hours, payment methods, clinic address and phone, policies, complaints, data governance, privacy, or security. 
- 'M' category for messages about general medical information described in medical facility documentation, including: preparation for medical procedures, course of medical procedures, post-op recommendations, medical risks. But not related to dates nor time.
- 'B' category when a patient confirms an appointment booking or provide missing data necessary to book an appointment.
- 'D' category for requests about date or time changes without specifying exact dates, e.g.: 'anything later', 'I need some earlier dates', 'following week', or similar.
- 'C' category for changing, rescheduling, or cancelling appointments.
- 'R' category when a patient briefly describes their medical problem or condition or asks for recommendations or advice on which medical specialization or doctor to visit.
- 'N' category for messages that do not fit any other category or include general greetings.
---
Once again, this is the patient message:
####
{{message}} 
####
Respond with the category of the patient message in JSON format.""")



###################################################
# prompt to extract the answer from text extracts #
###################################################
extract_answer_prompt_template = (
f"""Your task it to analyze and search 'The information' I am providing you with, in order to check if 'The information' is relevent for 'The question'.
If 'The information' is relevant for 'The question' then please extract and respond with the most import parts of 'The information'.
If 'The information' is not relevant for 'The question' or you cannot find the answer, please respond with a single word NONE
Here is 'The Information':
####
{{information}}
####
Here is 'The question': '{{question}}'

Additional rules to follow:
1. Do not make up any data if it's not present in the information provided.
2. Keep your answer short, precise and clear.
3. Do not include any additional comments from you.
""")


# helper function to apply the main prompt template
def apply_extract_answer_prompt_template(question, information, prompt=extract_answer_prompt_template): 
    return prompt.format(
            question=question,
            information=information,
        )


###################################################
################ translation prompt ###############
translate_prompt_template = (
f"""You are an experienced translator working at medical facility. You stick to the terminology and professional tone that are used at medical facilities. While translating you leave first and last names as they are (untranslated). You reply with the translation only (without any additional comments).
Translate text from {{src_language}} to {{dst_language}}:
{{text}}
""")


# helper function to apply the main prompt template
def apply_translate_prompt_template(src_language, dst_language, text, prompt=translate_prompt_template): 
    return prompt.format(
            src_language=src_language,
            dst_language=dst_language,
            text=text,
        )


###################################################
### OpenAI API call with token usage statistics ###
# Documentation:
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
# https://platform.openai.com/docs/api-reference/chat/object
def make_openai_call(open_ai_client, model, prompt, temperature=0.1, verbose=False, **kwargs):
    try:
        response = open_ai_client.chat.completions.create(
            model=model,  
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )

        # Check if there are tool calls in the response
        if response.choices[0].finish_reason == 'tool_calls' and response.choices[0].message.tool_calls:
            # For now, we have one tool call in the response
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            arguments_json = tool_call.function.arguments

            # Process the function call if it matches 'book_appointment'
            if function_name == "book_appointment":
                # Parse the arguments JSON string into a dictionary
                arguments = json.loads(arguments_json)
                if verbose:
                    print(f"Function call requested: {function_name} with arguments {arguments}")
                
                # Check for missing arguments
                required_keys = [
                    "appointment_date", "appointment_time", "dr_first_name", "dr_last_name",
                    "specialization", "price", "patient_first_name", "patient_last_name",
                    "patient_email", "patient_phone"]
                is_valid, error_message = check_required_arguments(arguments, required_keys)
                if not is_valid:
                    response_content = error_message
                else:
                    # Extract arguments
                    appointment_date = arguments["appointment_date"]
                    appointment_time = arguments["appointment_time"]
                    dr_first_name = arguments["dr_first_name"]
                    dr_last_name = arguments["dr_last_name"]
                    specialization = arguments["specialization"]
                    price = arguments["price"]
                    patient_first_name = arguments["patient_first_name"]
                    patient_last_name = arguments["patient_last_name"]
                    patient_email = arguments["patient_email"]
                    patient_phone = arguments["patient_phone"]

                    # Call the 'book_appointment' function with the extracted arguments
                    # TODO: As some data might be hallucinated, I assume that chat will display confirmation dialog to the potient before making a booking.
                    # In that context a) we may not be so worried about completness of the data, b) probably we should also pass patient data from session.
                    response_content = book_appointment(appointment_date, appointment_time, dr_first_name, dr_last_name, specialization, price, patient_first_name, patient_last_name, patient_email, patient_phone)
                     
            else:
                response_content = f"Function {function_name} is not supported."            
           
        else:
            response_content = response.choices[0].message.content        
        
        
        # Extracting token usage statistics
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        # if verbose:
        #     print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Model: {model}")

        return response_content, {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'model': model}
        
    except Exception as e:
        err_msg = f"An error occurred: {str(e)}"
        if verbose:
            print(err_msg)
        return "", {'prompt_tokens': 0, 'completion_tokens': 0, 'model': ''}


def check_required_arguments(arguments, required_keys):
    """
    Checks if all required arguments are present in the given arguments dictionary.
    Returns a tuple (is_valid, message) where is_valid is True if all required keys are present,
    and message is either an empty string if is_valid is True or a message detailing missing keys.
    """
    missing_keys = [key for key in required_keys if key not in arguments or arguments[key] is None or arguments[key] == ""]
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        message = f"To proceed with booking the appointment, please provide all necessary but missing information: {missing_keys_str}."
        return False, message
    else:
        return True, ""

# Tools parameter for OpenAI API call - it is used when the model is about to book an appointment
tools_booking = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment with a doctor",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_date": {
                        "type": "string",
                        "description": "The date of the appointment, in YYYY-MM-DD format",
                    },
                    "appointment_time": {
                        "type": "string",
                        "description": "The time of the appointment, in HH:MM format",
                    },
                    "dr_first_name": {
                        "type": "string",
                        "description": "The first name of the doctor",
                    },
                    "dr_last_name": {
                        "type": "string",
                        "description": "The last name of the doctor",
                    },
                    "specialization": {
                        "type": "string",
                        "description": "The doctor's specialization",
                    },
                    "price": {
                        "type": "number",
                        "description": "The price of the appointment",
                    },
                    "patient_first_name": {
                        "type": "string",
                        "description": "The first name of the patient",
                    },
                    "patient_last_name": {
                        "type": "string",
                        "description": "The last name of the patient",
                    },
                    "patient_email": {
                        "type": "string",
                        "description": "The email address of the patient",
                    },
                    "patient_phone": {
                        "type": "string",
                        "description": "The phone number of the patient",
                    },
                },
                "required": [
                    "appointment_date", 
                    "appointment_time", 
                    "dr_first_name", 
                    "dr_last_name", 
                    "specialization", 
                    "price", 
                    "patient_first_name", 
                    "patient_last_name", 
                    "patient_email", 
                    "patient_phone"
                ],
            },
        }
    },
]
