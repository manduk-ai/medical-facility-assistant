"""
Filename: finders.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: Includes 3 classes that are used to find doctors, medical specialties and temporal data in the input text.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""
import sqlite3
import re
from datetime import datetime, timedelta
from dateutil.parser import parse
import calendar
import spacy


def find_email_phone(text):
    """
    Function that finds email and telephone number in the input text.

    Args:
        text (str): The input text to search for email and telephone number.

    Returns:
        tuple: A tuple containing the found email and telephone number (email, telephone).
               If no email or telephone number is found, the corresponding value in the tuple will be None.
    """
    # Regular expression for finding email + search
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    email = email_match.group(0) if email_match else None

    # Regular expression for finding telephone + search    
    polish_telephone_pattern = r"\b\d{3}(?:-|\s|\b)?\d{3}(?:-|\s|\b)?\d{3}\b"

    # Initialize the telephone variable so that you may want to subsequently search for a telephone number in different formats
    telephone = None
    # We want to normalize the input text so that we can search for telephone numbers in a consistent way
    normalized_input_text = " " + text + " "
    for punct in [".", ",", ";", ":", "?", "!", "—", "-", "+48", "+49"]:
        normalized_input_text = normalized_input_text.replace(punct, " ")

    # search for Polish mobile numbers
    if not telephone:
        polish_match = re.search(polish_telephone_pattern, normalized_input_text)
        if polish_match:
            if polish_match:
                telephone = polish_match.group(0)
                telephone = telephone.replace(" ", "") # remove spaces
                if len(telephone) == 9:
                    telephone = f"{telephone[:3]}-{telephone[3:6]}-{telephone[6:]}"                

    return (email, telephone)


class DoctorFinder:
    """
    A class that helps find doctors based on input text and patient name.

    Attributes:
        results_by_last_name (dict): A dictionary to store the results by last name.
        conn (sqlite3.Connection): A connection to the SQLite database.
        doctor_indicators (list): A list of strings that might be a doctor prefix.
        pattern (str): A regex pattern to match words.

    Methods:
        __init__(self, database): Initializes the DoctorFinder object.
        find_doctors(self, input_text, patient_name): Finds doctors based on input text and patient name.
        test(self): Example usage of the DoctorFinder class.
    """

    def __init__(self, database):
        """
        Initializes an instance of the Finders class.

        Args:
            database (str): The path to the SQLite database.

        Attributes:
            results_by_last_name (dict): A dictionary to store the search results by last name.
            conn (sqlite3.Connection): The connection to the SQLite database.
            doctor_indicators (list): A list of strings that might be a doctor prefix.
            pattern (str): A regex pattern to match words.
        """                    
        self.results_by_last_name = {}  # Initialize Python dictionary
        self.conn = sqlite3.connect(database)
        self.doctor_indicators = ["dr", "dr.", "doctor", "doc", "doc.", "doktor", "physician", "medic"]  # Define list of strings that might be a doctor prefix
        self.pattern = r'\b\w+\b'  # Define regex pattern to match words
        
        # build the dictionary of doctors and their specialties
        cur = self.conn.cursor()
        cur.execute("""
        SELECT d.id as "Doctor ID",
            d.first_name as "First name",
            d.last_name as "Last name",
            m.id as "Specialty ID",
            m.name as "Specialty name",
            ds.specialty_id as "Doctor specialty ID",
            ds.price as "Price",
            ds.currency as "Currency"
        FROM doctors d, medical_specialties m, doctors_specialties ds
        WHERE ds.doctor_id = d.id AND m.id = ds.specialty_id;
        """)
        query_results = cur.fetchall()  # Fetch all results

        # Populate the structure
        for row in query_results:
            doctor_id, first_name, last_name, specialty_id, specialty_name, doctor_specialty_id, price, currency = row
            if last_name not in self.results_by_last_name:
                self.results_by_last_name[last_name] = []
            self.results_by_last_name[last_name].append({
                "Doctor ID": doctor_id,
                "First name": first_name,
                "Last name": last_name,
                "Specialty ID": specialty_id,
                "Specialty name": specialty_name,
                "Doctor specialty ID": doctor_specialty_id,
                "Price": price,
                "Currency": currency
            })

        self.conn.close()

    def find_doctors(self, input_text, patient_name):
        """
        Finds doctors based on the input text and patient name.

        Args:
            input_text (str): The input text to search for doctor names.
            patient_name (str): The name of the patient.

        Returns:
            list: A list of dictionaries containing the details of the matched doctors.
        """
        # Use regex to split input text into words, ignoring punctuation
        words = re.findall(self.pattern, input_text)

        # Output list to store matching doctor details
        matched_doctors = []

        # Search for doctor names following the keywords
        for i, word in enumerate(words):
            if word.lower() in self.doctor_indicators:
                try:
                    # Initialize potential_last_name as None
                    potential_last_name = None

                    # Check the next word to determine if it could be the last name
                    next_word = words[i + 1]

                    # Assuming the format could be "indicator last_name"
                    if next_word in self.results_by_last_name:
                        potential_last_name = next_word
                    # If the next word is not a last name, check for "indicator first_name last_name" format
                    elif len(words) > i + 2:  # Ensure there are enough words left
                        next_next_word = words[i + 2]
                        if next_next_word in self.results_by_last_name:
                            potential_last_name = next_next_word

                    # If a potential last name was identified, extend the matched doctors
                    if potential_last_name and potential_last_name in self.results_by_last_name:
                        matched_doctors.extend(self.results_by_last_name[potential_last_name])
                except IndexError:
                    # Reached the end of the list, no further action needed
                    continue

        # If we found matches in Step 2, return them
        if matched_doctors:
            return matched_doctors

        # Step 3: If no matches found in Step 2, look for any last name in the input text
        for last_name, doctors in self.results_by_last_name.items():
            if last_name in input_text and last_name != patient_name:
                matched_doctors.extend(doctors)

        return matched_doctors

    def test(self):
        """
        Example usage of the DoctorFinder class.
        """
        input_text = "I was treated by doc Hernandez and doktor Lewis Linda, but not by the patient Adams."
        patient_name = "Adams"
        matched_doctors = self.find_doctors(input_text, patient_name)

        print("Matched Doctors:")
        for doctor in matched_doctors:
            print(doctor)


class MedicalSpecialtyFinder:
    """
    This class provides methods to find medical specialties based on keywords.

    Attributes:
        medical_specialties (dict): A dictionary mapping medical specialties to their corresponding keywords.
        medical_specialties_plural (dict): A dictionary mapping medical specialties to their plural forms.

    Methods:
        find_specialty(keyword: str) -> str: Finds the medical specialty based on the given keyword.
        find_specialty_plural(keyword: str) -> str: Finds the plural form of the medical specialty based on the given keyword.
    """    
    def __init__(self):
        self.medical_specialties = {
            "general practitioner": ["general practitioner", "family doctor", "family doc", "GP", "family physician", "general practice", "primary care"],
            "pediatrician": ["pediatrician", "paediatrician", "child doctor", "child doc", "pediatric", "kids doctor", "kids doc", "pediatrics"],
            "ophthalmologist": ["ophthalmologist", "eye doctor", "eye doc", "eye specialist", "vision doctor", "vision doc", "ophthalmology", "ophthalmological"],
            "dermatologist": ["dermatologist", "skin" , "acne", "dermatology", "dermatological"], 
            "gynecologist": ["gynecologist", "OB/GYN", "OB-GYN", "OBGYN", "OB GYN", "gynaecologist", "gynecology", "gynaecology", "gynecological", "gynaecological"],
            "urologist": ["urologist", "urinary specialist", "urology", "urological"],
            "orthopedist": ["orthopedist", "orthopaedist", "bone", "orthopedic", "orthopaedic", "orthopedics", "orthopaedics", "joint", "orthopedical", "orthopaedical"],
            "cardiologist": ["cardiologist", "heart doctor", "heart doc", "heart specialist", "cardiac", "cardiology", "cardiological"],
            "neurologist": ["neurologist", "brain doctor", "brain doc", "neurology specialist", "nerve doctor", "nerve doc", "neurology", "neurological"],
            "endocrinologist": ["endocrinologist", "gland", "hormone", "thyroid", "endo", "endocrinology", "endocrinological"],
            "psychiatrist": ["psychiatrist", "mental health", "mental health", "psychiatric", "psychiatry"],
            "rheumatologist": ["rheumatologist", "rheumatism", "rheumatic", "arthritis", "rheumatology", "rheumatological"],
            "otolaryngologist": ["otolaryngologist", "ENT", "ear nose throat", "head and neck", "head and neck", "otolaryngology", "otolarngological", "laryngological"],
            "pulmonologist": ["pulmonologist", "lung doctor", "lung doc", "respiratory", "chest doctor", "chest doc", "pulmonology", "pulmonary", "pulmonological"],
            "hematologist": ["hematologist", "blood doctor", "blood doc", "blood specialist", "hematology", "hematological"],
            "gastrologist": ["gastrologist", "gastroenterologist", "digestive", "stomach", "GI doctor", "GI doc", "gastro", "gastroenterology", "gastrology", "gastroenterological", "gastrological"],
            "nephrologist": ["nephrologist", "kidney", "renal", "bladder", "nephrology", "nepfrological"],
            "oncologist": ["oncologist", "cancer", "tumor", "oncology", "oncological", "oncologic"],
            "radiologist": ["radiologist", "imaging specialist", "imaging doctor", "imaging doc", "X-ray doctor", "X-ray doc", "radiology", "radiologic", "radiological"],
            "plastic surgeon": ["plastic surgeon", "cosmetic surgeon", "aesthetic", "reconstructive"],
            "neurosurgeon": ["neurosurgeon", "brain surgeon", "spinal surgeon", "neurosurgical", "brain surgery", "spinal surgery", "neurosurgery"],
            "pediatric surgeon": ["paediatric surgeon", "pediatric surgeon", "child surgeon", "kid surgeon", "kids surgeon", "surgery for children", "paediatric surgery", "pediatric surgery"],
            "vascular surgeon": ["vascular", "vessel surgeon", "vessel surgery", "angiologist"],
            "maxillofacial surgeon": ["maxillofacial", "oral surgeon", "jaw surgeon", "facial surgeon", "oral surgery", "jaw surgery", "facial surgery"],
            "surgeon": ["surgeon", "surgical doctor", "surgical doc", "surgical specialist", "operating doctor", "operating doc", "surgery", "surgical"],
            "geriatrician": ["geriatrician", "elderly", "aging specialist", "senior health", "geriatrics", "geriatric"],
            "internist": ["internist", "internal medicine"],
            "neonatologist": ["neonatologist", "newborn", "NICU doctor", "infant", "neonatology", "neonatological"],
            "allergist": ["allergist", "allergy", "allergic", "allergology", "allergological"],
            "andrologist": ["andrologist", "men's health", "male health", "man health", "andrology", "andrological"],
            "diabetologist": ["diabetologist", "diabetes", "diabetic", "diabetology", "diabetological"],
            "pediatric cardiologist": ["paediatric cardiologist", "pediatric cardiologist", "child heart", "paediatric heart", "pediatric heart", "pediatric cardiology"],
            "pediatric neurologist": ["paediatric neurologist", "pediatric neurologist", "child brain", "paediatric brain", "pediatric brain", "pediatric neurology"],
            "child and adolescent psychiatrist": ["psychiatrist for kids", "kid psychiatrist", "child psychiatrist", "adolescent psychiatrist", "psychiatrist for teenagers", "teenager psychiatrist", "child psychiatry", "adolsecent psychiatry", "teenager psychiatry"],
            "sexologist": ["sexologist", "sexual", "sex therapy", "sexology", "sexological"],
            "transplantologist": ["transplantologist", "transplant surgeon", "transplant doctor", "transplant doc", "transplantology", "transplantological"],
            "audiologist": ["audiologist", "hearing specialist", "audiology", "audiological"],
            "dentist": ["dentist", "dental", "oral", "dentistry"],
            "orthodontist": ["orthodontist", "braces", "teeth alignment", "orthodontics", "orthodontical"],
            "psychotherapy": ["psychotherapy", "psychological", "therapy", "counseling"],
            "nutrition consultation": ["nutrition", "dietitian", "nutritionist", "dietary"],
            "acupuncture": ["acupuncture", "acupuncturist", "Chinese medicine", "needle therapy"],
            "massage": ["massage", "hand therapy", "physical therapy"],
            "osteopathy": ["osteopathy", "osteopath", "manual therapy", "osteopathic"],
            "chiropractic": ["chiropractic", "chiropractor", "spinal adjustment"],
            "podiatry": ["podiatry", "foot doctor", "foot doc", "podiatrist", "foot specialist"],
            "speech therapy": ["speech therapy", "speech therapist", "speech doctor", "speech doc", "speech treatment"],
            "psychoanalysis": ["psychoanalysis", "psychoanalyst", "depth psychology"],
            "couple therapy": ["couple therapy", "relationship therapy", "marital therapy", "couples counseling"],
            "blood test": ["blood test", "blood work", "hematology test", "blood analysis"],
            "urine test": ["urine test", "urinalysis", "urine analysis"],
            "ECG": ["ECG", "electrocardiogram", "heart test", "EKG"],
            "ultrasound": ["ultrasound", "sonogram", "ultrasonography"],
            "X-ray": ["X-ray", "xray", "radiography", "radiographic imaging"], 
            "MRI": ["MRI", "resonance", "magnetic"],
            "CT scan": ["CAT imaging", "tomography", "CT", "CAT scan"],
            "vaccination": ["vaccination", "immunization", "vaccine shot", "shots", "immunotherapy", "immunological", "immunology"],
            "physiotherapy": ["physiotherapy", "physical therapy", "rehabilitation", "rehab"],
        }        

        # This is additional dictionary with plural forms of medical specialties
        self.medical_specialties_plural = {
            "general practitioner": ["general practitioners", "family doctors", "family docs", "GPs", "family physicians", "general practices"],
            "pediatrician": ["pediatricians", "paediatricians", "child doctors", "child docs", "kids doctors", "kids docs"],
            "ophthalmologist": ["ophthalmologists", "eye doctors", "eye docs", "eye specialists", "vision doctors", "vision docs"],
            "dermatologist": ["dermatologists"],
            "gynecologist": ["gynecologists", "OB/GYNs", "OB-GYNs", "OBGYNs", "OB GYNs", "gynaecologists"],
            "urologist": ["urologists", "urinary specialists", "urology specialists"],
            "orthopedist": ["orthopedists", "orthopaedists", "bones", "orthopedics", "orthopaedics", "joints"],
            "cardiologist": ["cardiologists"],
            "neurologist": ["neurologists", "brain doctors", "brain docs", "neurology specialists", "nerve doctors", "nerve docs"],
            "endocrinologist": ["endocrinologists"],
            "psychiatrist": ["psychiatrists"],
            "rheumatologist": ["rheumatologists"],
            "otolaryngologist": ["otolaryngologists", "ENTs"],
            "pulmonologist": ["pulmonologists", "lung doctors", "lung docs", "chest doctors", "chest docs"],
            "hematologist": ["hematologists", "blood doctors", "blood docs", "blood specialists"],
            "gastrologist": ["gastrologists", "gastroenterologists", "GI doctors", "GI docs"],
            "nephrologist": ["nephrologists", "kidneys", "renals"],
            "oncologist": ["oncologists"],
            "radiologist": ["radiologists", "imaging specialists", "imaging doctors", "imaging docs", "X-ray doctors", "X-ray docs"],
            "plastic surgeon": ["plastic surgeons", "cosmetic surgeons"],
            "neurosurgeon": ["neurosurgeons", "brain surgeons", "spinal surgeons", "brain surgeries", "spinal surgeries", "neurosurgeries"],
            "pediatric surgeon": ["paediatric surgeons", "pediatric surgeons", "child surgeons", "kid surgeons", "kids surgeons"],
            "vascular surgeon": ["blood vessel surgeons", "angiologists"],
            "maxillofacial surgeon": ["maxillofacial surgeons", "oral surgeons", "jaw surgeons", "facial surgeons"],
            "surgeon": ["surgeons", "surgical doctors", "surgical docs", "surgical specialists", "operating doctors", "operating docs"],
            "geriatrician": ["geriatricians", "aging specialists"],
            "internist": ["internists"],
            "neonatologist": ["neonatologists", "NICU doctors"],
            "allergist": ["allergists"],
            "andrologist": ["andrologists"],
            "diabetologist": ["diabetologists"],
            "pediatric cardiologist": ["paediatric cardiologists", "pediatric cardiologists"],
            "pediatric neurologist": ["paediatric neurologists", "pediatric neurologists"],
            "child and adolescent psychiatrist": ["psychiatrists for kids", "child psychiatrists", "kid psychiatrists", "adolescent psychiatrists", "psychiatrists for teenagers", "teenagers psychiatrist"],
            "sexologist": ["sexologists"],
            "transplantologist": ["transplantologists", "transplant surgeons", "transplant doctors", "transplant docs"],
            "audiologist": ["audiologists", "hearing specialists"],
            "dentist": ["dentists"],
            "orthodontist": ["orthodontists"],
            "psychotherapy": ["psychotherapies", "counselings"],
            "nutrition consultation": ["dietitians", "nutritionists"],
            "acupuncture": ["acupunctures", "acupuncturists", "needle therapies", "needles therapy", "needles therapies"],
            "massage": ["massages", "hand therapies", "physical therapies"],
            "osteopathy": ["osteopathies", "osteopaths", "manual therapies", "osteopathics"],
            "chiropractic": ["chiropractics", "chiropractors", "spinal adjustments"],
            "podiatry": ["podiatries", "foot doctors", "foot docs", "podiatrists", "foot specialists", "feet doctors", "feet docs", "feet specialists"],
            "speech therapy": ["speech therapies", "speech therapists", "speech doctors", "speech docs", "speech treatments"],
            "psychoanalysis": ["psychoanalyses", "psychoanalysts", "depth psychologies"],
            "couple therapy": ["couple therapies", "relationship therapies", "marital therapies"],
            "blood test": ["blood tests", "hematology tests", "blood analyses"],
            "urine test": ["urine tests", "urinalyses", "urine analyses"],
            "ECG": ["ECGs", "electrocardiograms", "heart tests", "EKGs"],
            "ultrasound": ["ultrasounds", "sonograms", "ultrasonographies"],
            "X-ray": ["X-rays", "xrays", "radiographies"], 
            "MRI": ["MRIs", "magnetic resonances"],
            "CT scan": ["CT scans", "computed tomographies", "CAT scans", "tomographies", "CTs"],
            "vaccination": ["vaccinations", "immunizations", "vaccine shots", "shots", "immunotherapies"],
            "physiotherapy": ["physiotherapies", "physical therapies", "rehabilitations"],
        } 


    def find_medical_specialties(self, input_text):
        """
        Find medical specialties in the input text.

        Args:
            input_text (str): The input text to search for medical specialties.

        Returns:
            list: A list of found medical specialties.

        """
        # Normalize the input text: lowercase, pad with spaces, replace punctuation
        normalized_input_text = " " + input_text.lower() + " "
        for punct in [".", ",", ";", ":", "?", "!", "—", "-"]:
            normalized_input_text = normalized_input_text.replace(punct, " ")
        
        # Initialize a list to keep track of found specialties
        found_specialties = []
        
        # Iterate over the medical specialties dictionary. Some notes on this:
        # Pad the Input Text: Add spaces at the beginning and end of the input text to ensure that even if a variation is at the start or end of the text, it is matched correctly.
        # Normalize Punctuation in the Input Text: Replace common punctuation marks with spaces in the input text. This step ensures that variations at the end of a sentence or followed by a comma are still matched correctly.
        # Pad the Variations: When checking each variation, pad it with spaces at the beginning and end. This ensures that the match requires the variation to be a standalone word or phrase.
        for specialty, variations in self.medical_specialties.items():
            # Check each padded variation to see if it is in the normalized input text
            for variation in variations:
                padded_variation = " " + variation.lower() + " "
                if padded_variation in normalized_input_text:
                    found_specialties.append(specialty)
                    break  # Avoid duplicates
            
            # We do not expect more than 4 specialties in a single turn
            if len(found_specialties) == 4:
                break
        
        # only if we didn't find anything in the first (singulars) loop, we will check the plurals
        # TODO: if it works well, we can create a function to avoid code repetition
        if len(found_specialties) == 0:
            for specialty, variations in self.medical_specialties_plural.items():
                # Check each padded variation to see if it is in the normalized input text
                for variation in variations:
                    padded_variation = " " + variation.lower() + " "
                    if padded_variation in normalized_input_text:
                        found_specialties.append(specialty)
                        break

        return found_specialties
    
    def test(self):
        """
        Test method to demonstrate the usage of the find_medical_specialties function.
        """
        input_text = "I need to see bone doctor asap and also need a blood test and CT and xray."
        found_specialties = self.find_medical_specialties(input_text)
        print(found_specialties)


class TemporalDataAnalyzer:
    """
    Class that is used to extract temporal data from the input text using spacy, dateutils, regular expressions and fuzzy logic.

    Attributes:
    - nlp: Model we use for finding 'DATE' entities in the text.

    Methods:
    - word_to_number(word): Converts number words to integers.
    - parse_date_entity(entity): Parses the date entity to extract the date range.
    - get_date_range_from_text(processed_input): Extracts the date range from the preprocessed input.
    - preprocess_input(input): Preprocesses the input text.
    - process_output(range): Processes the extracted date range.
    """        
    def __init__(self):
        """
        Initializes the TemporalDataAnalyzer class.

        This method loads the spaCy English language model and initializes the mapping of number words to their numeric values.

        Parameters:
            None

        Returns:
            None
        """
        # self.nlp = spacy.load('en_core_web_trf')
        # self.nlp = spacy.load('en_core_web_lg')
        self.nlp = spacy.load('en_core_web_sm')

        # Mapping of number words to their numeric values
        self.number_word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "few": 5, "couple": 2
        }

        self.month_numbers = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}

    
    def word_to_number(self, word):
        """
        Converts a word or numeric string to its corresponding integer representation.

        Args:
            word (str): The word or numeric string to be converted.

        Returns:
            int or None: The integer representation of the word or numeric string. Returns None if the conversion fails.
        """
        try:
            # Directly convert numeric strings to integers
            return int(word)
        except ValueError:
            # Convert number words to integers
            return self.number_word_to_num.get(word.lower(), None)


    def parse_date_entity(self, entity):
        """
        Parses a date entity and returns the corresponding date range.

        Args:
            entity (str): The date entity to parse.

        Returns:
            tuple: A tuple containing the start date and end date of the parsed date entity.

        Raises:
            ValueError: If the date entity cannot be parsed.

        """
        today = datetime.now()
        if entity.lower() in ["today"]:
            return today, today + timedelta(days=1)
        elif entity.lower() in ["tomorrow"]:
            return today + timedelta(days=1), today + timedelta(days=1)
        elif entity.lower() in ["the day after tomorrow"]:
            return today + timedelta(days=2), today + timedelta(days=2)
        elif "next week" in entity.lower():
            start_date = today + timedelta(days=(7-today.weekday()))
            end_date = start_date + timedelta(days=6)
            return start_date, end_date
        elif "this week" in entity.lower():
            # if today is Saturday or Sunday, we consider the current week as the next week
            if today.weekday() >= 5:
                start_date = today + timedelta(days=(7-today.weekday()))
                end_date = start_date + timedelta(days=6)
            else:
                start_date = today
                end_date = today + timedelta(days=(6-today.weekday()))
            return start_date, end_date
        elif "the weekend" in entity.lower():
            # in this case, as a user may point out to: before, during or after the weekend, we meed start from today and end 3 days after Sunday
            days_until_sunday = (6 - today.weekday()) % 7
            start_date = today
            end_date = today + timedelta(days=days_until_sunday + 3)
            return start_date, end_date
        elif "the next" in entity.lower() and "days" in entity.lower():
            parts = entity.lower().split()
            if "or" in parts:  # Handle "the next N or M days"
                numbers = [self.word_to_number(part) for part in parts if self.word_to_number(part) is not None]
                if len(numbers) >= 2:
                    n = max(numbers)  # Focus on the larger number
                    return today, today + timedelta(days=n)
            else:  # Handle "the next N days"
                for part in parts:
                    n = self.word_to_number(part)
                    if n is not None:
                        return today, today + timedelta(days=n)
        elif "the next" in entity.lower() and "weeks" in entity.lower():  # weeks with "the next" keyword
            parts = entity.lower().split()
            if "or" in parts or "to" in parts or "-" in entity:  # Handle "the next N or M weeks", "the next N to M weeks", "N-M weeks"
                # Find all numbers or number words, and convert them to numeric values
                numbers = [self.word_to_number(part) for part in parts if self.word_to_number(part) is not None]
                # Clean '-' cases like '2-5 weeks' or '2 - 5 weeks', assuming they are already split into parts
                numbers += [self.word_to_number(part) for part in entity.replace('-', ' ').split() if self.word_to_number(part) is not None]
                if len(numbers) >= 2:
                    n = max(numbers)  # Choose the larger number of weeks
                    return today, today + timedelta(weeks=n)
            else:  # Handle "the next N weeks" or "the next WORD weeks", like "the next three weeks"
                for part in parts:
                    n = self.word_to_number(part)
                    if n is not None:
                        return today, today + timedelta(weeks=n)
        # TODO twin code snippet, with respect to the previous one. So probably we can merge them into one by dropping the first condition
        # For now, I will keep them separate for flexibility
        elif "weeks" in entity.lower():  
            parts = entity.lower().split()
            # Handle ranges specified with "or", "to", or "-"
            if "or" in parts or "to" in parts or "-" in entity:
                # Extract and convert all numbers or number words to numeric values
                numbers = [self.word_to_number(part) for part in parts if self.word_to_number(part) is not None]
                # Handle cases with dashes, e.g., "2-5 weeks", considering potential spaces around the dash
                numbers += [self.word_to_number(part) for part in entity.replace('-', ' ').split() if self.word_to_number(part) is not None]
                if len(numbers) >= 2:
                    n = max(numbers)  # Use the larger number to determine the duration in weeks
                    return today, today + timedelta(weeks=n)
            else:  # Handle "N weeks" or "WORD weeks", including "few weeks" as a special case
                for part in parts:
                    n = self.word_to_number(part)
                    if n is not None:
                        return today, today + timedelta(weeks=n)
        elif "this month" in entity.lower():
            date_from = today
            last_day = calendar.monthrange(today.year, today.month)[1]
            date_to = datetime(today.year, today.month, last_day)
            return date_from, date_to
        elif "next month" in entity.lower():
            if today.month == 12:  # If it's December, wrap around to January of the next year
                next_month = 1
                year = today.year + 1
            else:
                next_month = today.month + 1
                year = today.year
            date_from = datetime(year, next_month, 1)
            last_day = calendar.monthrange(year, next_month)[1]
            date_to = datetime(year, next_month, last_day)
            return date_from, date_to
        elif "months" in entity.lower():
            parts = entity.lower().split()
            for part in parts:
                n = self.word_to_number(part)
                if n is not None:
                    if n > 3:  # Cap the number of months at 3
                        n = 3
                    break
            else:
                n = 1  # Default to 1 month if no number is found

            # Calculate the end month considering year wrap around
            months_to_add = n
            year = today.year
            month = today.month + months_to_add
            while month > 12:  # Adjust for year change
                month -= 12
                year += 1

            date_from = today
            # Calculate the last day of the end month
            last_day = calendar.monthrange(year, month)[1]
            date_to = datetime(year, month, last_day)
            return date_from, date_to
        else:
            # This part is a little bit tricky, as dates in a form "February 29th" are both identified as a month and a day.
            # That's why we need to make two verifications first and then make some analisys
            v_date_from, v_date_to = None, None

            # check if entity contains a month name
            for month_name, month_num in self.month_numbers.items():
                if month_name in entity.lower():
                    current_month_num = today.month
                    if month_num == current_month_num:  # Case 1: Current month
                        date_from = today
                        last_day = calendar.monthrange(today.year, month_num)[1]
                        date_to = datetime(today.year, month_num, last_day)
                    elif month_num > current_month_num:  # Case 2: Future month in the current year
                        date_from = datetime(today.year, month_num, 1)
                        last_day = calendar.monthrange(today.year, month_num)[1]
                        date_to = datetime(today.year, month_num, last_day)
                    else:  # Case 3: Month in the next year
                        date_from = datetime(today.year + 1, month_num, 1)
                        last_day = calendar.monthrange(today.year + 1, month_num)[1]
                        date_to = datetime(today.year + 1, month_num, last_day)
                    v_date_from, v_date_to = date_from, date_to

            # Use `dateutil` for more general date parsing when applicable
            try:
                date = parse(entity, fuzzy=True)
                if v_date_from is None and v_date_to is None:
                    return date, date
                else:
                    # if date is in the range between v_date_from and v_date_to, we return month range as it is broader and dateutil has problems
                    if v_date_from <= date <= v_date_to:
                        return v_date_from, v_date_to
                    elif date < v_date_from: # if date is before v_date_from, we extend the range to the left
                        return date, v_date_to
                    elif date > v_date_to:  # if date is after v_date_to, we extend the range to the right
                        return v_date_from, date
                        
            except ValueError:
                # if there was as error while parsing using dateutil, we return values from the 'month' analysis (they might be None as well)
                return v_date_from, v_date_to  # Cannot be parsed


    def get_date_range_from_text(self, text):
        """
        Extracts date ranges from the given text.

        Args:
            text (str): The input text.

        Returns:
            list: A list of date ranges in the format (start_date, end_date).

        """
        doc = self.nlp(text)
        date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        ranges = []
        for entity in date_entities:
            start_date, end_date = self.parse_date_entity(entity)
            if start_date and end_date:
                ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        return ranges


    def preprocess_input(self, text):
        """
        Preprocesses the input text by replacing specific patterns and phrases with their corresponding replacements.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text with patterns and phrases replaced.

        """
        # Patterns for "24 hours" and its variations
        patterns_24 = r'24\s*(hours?|hrs?|h)'
        replacement_24 = '2 days'
        
        # Patterns for "48 hours" and its variations
        patterns_48 = r'48\s*(hours?|hrs?|h)'
        replacement_48 = '3 days'
        
        # Replace all occurrences of the patterns
        text = re.sub(patterns_24, replacement_24, text, flags=re.IGNORECASE)
        text = re.sub(patterns_48, replacement_48, text, flags=re.IGNORECASE)
        
        # List of phrases to replace with "next 4 weeks"
        replacements = [
            "next available date",
            "earliest possible opportunity",
            "soonest convenience",
            "nearest open slot",
            "first available appointment",
            "earliest opening",
            "next open appointment",
            "soonest available slot",
            "nearest available time",
            "first opportunity",
            "earliest convenience",
            "next available slot",
            "soonest appointment",
            "nearest booking",
            "earliest slot",
            "next opening",
        ]
        
        # Combine the list into a single pattern using regex alternation
        pattern_replacements = r'\b(?:' + '|'.join(re.escape(phrase) for phrase in replacements) + r')\b'
        
        # Replace all occurrences of the combined pattern with "next 4 weeks"
        text = re.sub(pattern_replacements, "next 4 weeks", text, flags=re.IGNORECASE)
        
        return text


    def process_output(self, output):
        """
        Process the output of a query and return the date range.

        Args:
            output (list): A list of tuples containing dates in string format.

        Returns:
            tuple: A tuple containing the start date and end date of the date range.

        """
        output_len = len(output)
        # If there's more than one tuple, merge into a single range.
        if output_len > 1:
            # Find the smallest and largest dates in the list of tuples.
            dates = [datetime.strptime(date, '%Y-%m-%d').date() for tup in output for date in tup]  # Convert immediately to date        
            date_from = min(dates)
            date_to = max(dates)
        elif output_len == 1:
            # If there's only one tuple, just use its values.
            date_from, date_to = [datetime.strptime(date, '%Y-%m-%d').date() for date in output[0]]  # Convert immediately to date
        else:
            date_from = None
            date_to = None

        # If date_from is in the past, we set it to today
        # If the range is 1 to 3 days, extend the end date by 7 days.
        if date_to is not None and date_from is not None:
            if date_from < datetime.now().date():
                date_from = datetime.now().date()
                # if date_to is now lower than date_from, we set it to date_from
                if date_to < date_from:
                    date_to = date_from

            if (date_to - date_from).days <= 3:
                date_to += timedelta(days=7)

        return date_from, date_to


    def find_temporal_data(self, input):
        """
        Method serving as an interface for extracting temporal data from the input text.
        
        This method takes an input text and extracts temporal data from it. It follows the same interface as the TemporalSUTimeDataAnalyzer class.
        
        Args:
            input (str): The input text from which temporal data needs to be extracted.
            
        Returns:
            tuple: A tuple containing four date objects. The first two date objects (date1 and date2) are set to None. The last two date objects (date_from and date_to) represent the calculated date range.
        """
        date1, date2, date_from, date_to = None, None, None, None  # these are the variables we want to extract from the text

        processed_input = self.preprocess_input(input)
        range = self.get_date_range_from_text(processed_input)
        date_from, date_to = self.process_output(range)

        return date1, date2, date_from, date_to  # here are date objects that will be by default printed as "YYYY-MM-DD"
 
   
    def test(self):
        """
        This method is used to test the functionality of the `get_date_range_from_text` method.
        It iterates over a list of example sentences and prints the preprocessed input, raw output,
        and final dates obtained from the `get_date_range_from_text` method.
        """        
        examples = [
            "I'd like to make an appointment next week.",
            "next week",
            "What date are available in the next 2 or 3 days?",
            "I need to see someone today.",
            "What options I have tomorrow and after tomorrow?",
            "What options I have tomorrow and the day after tomorrow?",
            "Anything on February 29th?",
            "Do you have any availability during the first week of August?",
            "How many free slot do you have June, onwards?",
            "I'd like to book something in April",
            "What about consultation in March?",
            "What about consultation in February?",
            "I have a dentist appointment next Wednesday.",
            "Can I schedule a check-up for sometime this Friday afternoon?",
            "I'd prefer an appointment early in the morning next Monday, if possible.",
            "Is there any availability on the first Monday of next month?",
            "I'm looking to book a follow-up appointment for the second week of next month.", 
            "I need to see a dermatologist by the end of this month, if possible.",
            "Want the appointment the second part of the next month.",
            "Want appointment with 2 months from now.",
            "I need to reschedule my appointment to any day late next week.",
            "Could you find me a slot for a dental check-up in the next 48 hours?", 
            "Could you find me a slot for a dental check-up in the next 24 hours?", 
            "Could you find me a slot for a dental check-up in the next 2 days?", 
            "I'm looking for the earliest available appointment before the weekend.", 
            "Do you have any slots open for a physical therapy session this Wednesday?", 
            "I'd like to book an eye exam for the day after tomorrow, please.",
            "Are there any cancellations this week that I could take advantage of?",
            "I'm trying to get an appointment for my yearly check-up within the next two weeks.", 
            "Can you check if there's an opening for a vaccination next Tuesday?",
            "I'd like to book a physiotherapy session for Wednesday.",
            "I need to see a dermatologist by the end of this month, if possible.",
            "What's the earliest time you have available for a consultation next week?", 
            "I'm looking to book a follow-up appointment for the second week of next month.", 
            "Could I get an appointment for a blood test this Thursday or Friday?",
            "I'd appreciate it if you could find me a slot for an X-ray this Saturday.",
            "Is it possible to arrange a nutritionist consultation early next week?",  
            "I need an emergency dental appointment, ideally today or tomorrow.",
            "Can I have the first appointment available on any day next week?", 
            "I need to consult with a cardiologist within the next few weeks.", 
            "I need to consult with a cardiologist within the next 4 weeks, please.", 
            "I need to consult with a cardiologist within the next 2 to 5 weeks, please.", 
            "I need to consult with a cardiologist within the next 2-5 weeks, please.", 
            "I need to consult with a cardiologist within the next three weeks.", 
            "I'm trying to schedule a mental health consultation for the next available date.",
            "earliest possible opportunity",
            "soonest convenience",
            "nearest open slot",
            "first available appointment",
            "earliest opening",
            "next open appointment",
            "soonest available slot",
            "nearest available time",
            "first opportunity",
            "earliest convenience",
            "next available slot",
            "soonest appointment",
            "nearest booking",
            "earliest slot",
            "next opening",
            "Do you have any evening appointments in the next few days for a check-up?",
            "I'd like to book a physiotherapy session for the upcoming Wednesday.", 
            "Is there a possibility to get a pediatric appointment this Friday?",
            "I need to consult with a cardiologist within the next week, please.", 
            "Can I secure a slot for an ultrasound at the earliest convenience next week?",
            "February 29",
            "February 29th",
            "February 2",
            "2nd of March",
            "2024-05-12",
            "12 May",
            "12th May",
            "2024-03-25",
            "Mar 25, 2024",
            "25th March 2024",
            "March 25th, 2024",
            "25/03/2024",
            "03/25/2024",
            "13 Apr",
            "Jun 30th",
            "July 4",
            "I'll book an appointment in May",
            "4th of July",
            "2023-11-11",
            "11 November",
            "11th November",
            "November 11th",
            "2025-01-01",
            "Jan 1, 2025",
            "1st January 2025",
            "January 1st, 2025",
            "01/01/2025",
            "01/01/25",
            "Dec 31",
            "31st December",
            "2026-02-29", # Leap year example
            "29/02/2026",
            "02/29/2026",
            "20 Oct",
            "Oct 20th",    
        ]
        for example in examples:
            print(f"{example}")
            example = self.preprocess_input(example)
            output = self.get_date_range_from_text(example)
            print(f"Raw output: {output}")
            date_from, date_to = self.process_output(output)
            print(f"Final dates: {date_from} - {date_to}\n\n")
