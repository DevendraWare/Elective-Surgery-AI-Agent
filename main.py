import os
import uuid
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import gc
import tempfile
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import time
import re
from openai import OpenAI
import sqlite3
import json
from functools import wraps
import threading
from contextlib import contextmanager
import speech_recognition as sr

 
# Define paths and load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "feedback.db")
os.makedirs(DB_DIR, exist_ok=True)
load_dotenv()
 
            
def get_db():
    """Create a new database connection for each thread."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
def init_db():
    """Initialize the database with the required schema."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                prompt_used TEXT,
                rating INTEGER,
                counselor_feedback TEXT,
                actual_outcome TEXT,
                feedback_timestamp TIMESTAMP,
                feedback_requested BOOLEAN DEFAULT FALSE
            )
        """)
        conn.commit()
    finally:
        conn.close()
 

 
def check_feedback():
    """Check for messages without feedback."""
    conn = get_db()
    try:
        c = conn.cursor()
        for message in reversed(st.session_state.messages):
            if message["role"] == "assistant" and "analysis_id" in message:
                c.execute("""
                    SELECT rating
                    FROM feedback
                    WHERE id = ? AND rating IS NOT NULL
                """, (message["analysis_id"],))
                has_feedback = c.fetchone()
                if not has_feedback:
                    with st.chat_message("assistant"):
                        st.info("Would you mind rating my last response? 👆")
                    break
    finally:
        conn.close()
 
class FeedbackManager:
    def __init__(self):
        self.last_interaction_time = None
        self.feedback_reminder_shown = False
        self.conversation_started = False
        self.feedback_acknowledged = False
   
    def update_interaction_time(self):
        self.last_interaction_time = datetime.now()
        self.feedback_reminder_shown = False
        self.conversation_started = True
       
    def acknowledge_feedback(self):
        self.feedback_acknowledged = True
        self.feedback_reminder_shown = False
 
    def should_request_feedback(self):
        """Only request feedback if:
        1. Conversation has started
        2. There's been 1 minute of inactivity
        3. There's an unrated message
        4. Reminder hasn't been shown yet
        5. Feedback hasn't been acknowledged
        """
        if not self.conversation_started or self.last_interaction_time is None or self.feedback_acknowledged:
            return False
           
        if not self.feedback_reminder_shown:
            time_since_last_interaction = (datetime.now() - self.last_interaction_time).total_seconds()
            if time_since_last_interaction >= 60:  # 1 minute
                conn = get_db()
                try:
                    c = conn.cursor()
                    c.execute("""
                        SELECT COUNT(*)
                        FROM feedback
                        WHERE rating IS NULL
                        AND id IN (
                            SELECT json_extract(value, '$.analysis_id')
                            FROM json_each(?)
                            WHERE json_extract(value, '$.role') = 'assistant'
                            AND json_extract(value, '$.analysis_id') IS NOT NULL
                        )
                    """, (json.dumps(st.session_state.messages),))
                    unrated_count = c.fetchone()[0]
                    if unrated_count > 0:
                        self.feedback_reminder_shown = True
                        return True
                finally:
                    conn.close()
        return False
 
class PatientAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))   
        init_db()
        if 'feedback_manager' not in st.session_state:
            st.session_state.feedback_manager = FeedbackManager()
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = []
        if 'mentioned_patients' not in st.session_state:
            st.session_state.mentioned_patients = []
        # New: Store the actual patient data with each response
        if 'filtered_patient_sets' not in st.session_state:
            st.session_state.filtered_patient_sets = {}
        if 'current_dataset_type' not in st.session_state:
            st.session_state.current_dataset_type = None

    def get_response_without_dataset(self, query):
        """Get a response when no dataset is uploaded, providing detailed information like ChatGPT."""
        try:
            # Retrieve conversation history
            conversation_history = st.session_state.conversation_context[-10:]  # Limiting the history to the last 10 interactions
 
            # Construct the system prompt to include a conversational context and follow-up instructions
            system_prompt = """
        You are a knowledgeable medical assistant with expertise in orthopedics, surgery, and general healthcare.
 
        When responding to medical queries:
        1. Provide detailed, specific information about potential diagnoses, conditions, and treatments.
        2. List multiple possibilities with explanations for each.
        3. Structure your response with numbered points for clarity.
        4. Include specific medical terminology where appropriate.
        5. Offer comprehensive analysis of symptoms and potential causes.
        6. DO NOT include disclaimers about not having access to patient data.
        7. Respond as if you are analyzing the case directly, similar to how a medical professional would discuss possibilities.
        8. When a specific patient is mentioned, refer to them by name and discuss their case specifically.
 
        **IMPORTANT**: After providing your response, always ask the user if they would like to explore further treatment options or get more information. 
        Ensure that the user feels guided through the process and that they can ask for more details or ask follow-up questions regarding the treatment plan, condition, or related aspects.
        Always provide the option to go deeper into treatment options or ask for further information.
        and if the query is not related to
        medical terminology you should reply that you are specifically trained for health assistance.
        """
           
            # Include the user's query along with the previous conversation
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})
 
            # Generate response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  
                messages=messages,
                temperature=0.3,
                max_tokens=800,
                stop=None
            )
           
            # Get assistant's response
            assistant_response = response.choices[0].message.content
 
            # Update the session state with the new conversation context
            st.session_state.conversation_context.append({"role": "user", "content": query})
            st.session_state.conversation_context.append({"role": "assistant", "content": assistant_response})
 
            return assistant_response
 
        except Exception as e:
            return f"I encountered an error while processing your query: {str(e)}"
        
    def detect_dataset_type(self, patient_data):
        columns = set(patient_data.columns.tolist())
        
        # Define keyword sets for each dataset type
        skin_keywords = {"Skin Type", "Skin Concerns", "Previous Skin Treatments","Allergy to Skincare Products","Sun Exposure Level","Treatment Preference","Desired Recovery Time","Preferred Treatment Areas","Keloid Formation History","Comfort with Downtime"}
        ortho_keywords = {"Pain Level", "Mobility Issues", "Joint"}
        contour_keywords = {"Target Body Areas", "Body Fat Percentage"}
        hair_keywords = {"Hair Loss Stage", "Scalp Condition", "Donor Area"}
        dental_keywords = {"Elective Procedure","Tooth Sensitivity Level","Current Dental Condition", "Bite Issues", "Gum Health","Aligner Preference","Previous Dental Procedures","Oral Hygiene Habits","Expectation from Treatment","Lifestyle Considerations","Allergies to Dental Materials"}
        
        # Check overlap with each set
        overlaps = {
            "Dermatology": len(columns.intersection(skin_keywords)),
            "Orthopedic": len(columns.intersection(ortho_keywords)),
            "Body Contouring": len(columns.intersection(contour_keywords)),
            "Hair Restoration": len(columns.intersection(hair_keywords)),
            "Cosmetic Dentistry": len(columns.intersection(dental_keywords))
        }
        
        # Return the type with the most keyword matches
        dataset_type = max(overlaps.items(), key=lambda x: x[1])[0]
        if overlaps[dataset_type] > 0:
            return dataset_type
        else:
            return "Unknown Medical Dataset" 
        
        
    def get_dynamic_prompt(self, patient_data):
        """Generate a dynamic prompt based on the dataset features with improved accuracy."""

        # First, analyze the dataset structure
        columns = patient_data.columns.tolist()
        # Ensure we have enough data to sample from
        sample_size = min(53, len(patient_data))
        sample_data = patient_data
        # Try to infer dataset type from columns
        if "Skin Type" in columns and "Skin Concerns" in columns:
            dataset_type = "Dermatology & Skin Procedures"
            
            # Extract skin-related data, ensuring no NaN values
            skin_types = patient_data["Skin Type"].dropna().unique().tolist()
            skin_concerns = patient_data["Skin Concerns"].dropna().unique().tolist()
            
            # Normalize column names for consistency
            normalized_columns = [col.lower() for col in columns]
            
            # Identify treatment history column dynamically
            treatment_history_col = None
            for col in normalized_columns:
                if "treatment" in col or "procedure" in col or "history" in col:
                    treatment_history_col = col
                    break

            # Build formatted text like Liposuction structure
            relevant_features = f"""
            **DATASET STRUCTURE**: This dataset contains detailed patient information for dermatological procedures.

            **DATASET COLUMNS**:
            Patient ID, First Name, Last Name, Age, Gender, email_address, BMI, Medical History, Skin Type, 
            Skin Concerns, Previous Skin Treatments, Allergy to Skincare Products, Sun Exposure Level, 
            Treatment Preference, Desired Recovery Time, Preferred Treatment Areas, Keloid Formation, Comfort with Downtime

            **PATIENT PROFILE ELEMENTS**:
            1. **Skin Type**: {', '.join(skin_types) if skin_types else "Various skin types documented"}
            2. **Skin Concerns**: {', '.join(skin_concerns) if skin_concerns else "Various concerns documented"}
            3. **Previous Treatments**: {"Available" if treatment_history_col else "Information may be limited"}
            4. **Risk Factors**: Allergy to Skincare Products, Sun Exposure Level, Keloid Formation
            5. **Patient Preferences**: Treatment Preference, Desired Recovery Time, Preferred Treatment Areas, Comfort with Downtime

            **ANALYSIS INSTRUCTIONS**:
            - When discussing patients, ALWAYS include their full name, age, skin type, and primary concern.
            - For follow-up questions, ONLY consider patients mentioned in your previous response.
            - When asked about candidates for specific treatments, evaluate contraindications carefully.
            - Consider skin sensitivity and allergies when recommending procedures.
            - Prioritize non-invasive options for patients with history of poor wound healing or keloid formation.
            - Factor in BMI and Medical History when evaluating procedure suitability.

            **PRIORITY CRITERIA**:
            - **High Priority**: Patients with severe skin conditions (e.g., severe acne, deep wrinkles).
            - **Medium Priority**: Patients with moderate concerns seeking improvement.
            - **Low Priority**: Patients with mild concerns or primarily cosmetic interests.

            **STRICT GUIDELINES**:
            - NEVER assume information not present in the dataset.
            - ALWAYS be explicit about which patients you're discussing.
            - Maintain strict continuity in follow-up questions about patient subsets.
            """  

        elif "pain_level" in columns and "mobility_level" in columns:
            dataset_type = "Orthopedic Surgery (Knee Replacements)"

            # Extract pain level details while handling missing values
            if patient_data["pain_level"].dtype in [int, float]:
                max_pain = patient_data["pain_level"].dropna().max()
                min_pain = patient_data["pain_level"].dropna().min()
                pain_scale = f"Scale from {min_pain} to {max_pain}"
            else:
                pain_scale = "Numeric scale (data may be incomplete)"

            # Extract mobility levels, handling missing values
            mobility_values = patient_data["mobility_level"].dropna().unique().tolist()

            # Normalize column names for consistency
            normalized_columns = [col.lower() for col in columns]

            # Build structured output similar to the Liposuction section
            relevant_features = f"""
            **DATASET STRUCTURE**: This dataset contains orthopedic patient information for knee replacement evaluations.

            **DATASET COLUMNS**:
            Patient ID, First Name, Last Name, Date of Birth, Age, Gender, Height (Feet), Weight (Pounds), Pain Level, Mobility Level, 
            Symptoms, Diagnosis, Chronic Conditions, Previous Surgeries, Medications, Allergies, Notes, Registration Date.

            **PATIENT PROFILE ELEMENTS**:
            1. **Pain Level**: {pain_scale}
            2. **Mobility Level**: Scale from 1 (low mobility) to 10 (high mobility)
            3. **Medical History**: Chronic Conditions, Previous Surgeries, Diagnosis, Medications
            4. **Additional Notes**: Clinical details & history from patient records
            5. **Time-Based Considerations**: Registration Date to analyze case urgency

            **ANALYSIS INSTRUCTIONS**:
            - Pain levels ≥7 indicate severe cases requiring priority evaluation.
            - When discussing patients, ALWAYS include their full name, age, height, weight, pain level, and primary mobility limitation.
            - For surgical candidates, evaluate:
                * Pain severity and duration
                * Impact on daily activities and quality of life
                * Previous interventions and their outcomes
                * Comorbidities affecting surgical risk
                * Medication interactions and allergies
                * Weight and height impact on surgical outcome

            **PRIORITY CRITERIA**:
            - **High Priority**: Severe pain (≥7) with significant mobility limitations and history of joint disorders.
            - **Medium Priority**: Moderate pain (4-6) with some functional impairment and relevant medical history.
            - **Low Priority**: Mild pain (<4) with minimal mobility issues and no significant orthopedic history.

            **STRICT GUIDELINES**:
            - ALWAYS maintain patient continuity in follow-up questions.
            - NEVER introduce new patients in follow-up discussions.
            - When asked about "these patients" or similar phrasing, ONLY reference patients from your most recent response.
            - Be precise about rehabilitation needs and post-surgery expectations.
            - Consider BMI, joint health, and chronic conditions when assessing candidacy.

            ---
            
            **YOU ARE AN AI ASSISTANT SPECIALIZED IN ANALYZING PATIENT DATA FOR SURGERY LIKELIHOOD.**
            
            **IMPORTANT CONTEXT INSTRUCTIONS**:
            1. When analyzing patient data, always be explicit about which patients you're referring to.
            2. When the user asks a follow-up question about "patients" or "these patients", ONLY consider the specific patients you mentioned in your most recent response, not the entire dataset.
            3. Each filter operation creates a NEW working set - subsequent questions always refer to this filtered set.
            4. DO NOT introduce patients that weren't in your previous response when handling follow-up questions.
            5. Be consistent in your patient references - use full names when first mentioning a patient.

            **CONSIDER THESE FACTORS WHEN ANALYZING PATIENTS**:
            - Pain levels, symptoms, medical history, mobility level, diagnoses, medications.
            - Prioritize urgent cases based on: severity, pain levels, impact on daily life, risk factors.

            **RESTRICTIONS**:
            - If asked about any topic unrelated to patient data or healthcare, politely explain that you are a specialized medical analysis assistant and can only answer questions related to the provided patient data.
            """


        elif "Procedure_Requested" in columns and "Procedure_Areas" in columns:
            dataset_type = "Cosmetic Surgery Assessment"
        
            # Get procedure areas from the dataset
            procedure_areas = []
            if "Procedure_Areas" in columns:
                procedure_areas = [area for areas in patient_data["Procedure_Areas"].unique() 
                                for area in str(areas).split(',')]
                procedure_areas = list(set([area.strip() for area in procedure_areas if area.strip()]))

            relevant_features = f"""
            **DATASET STRUCTURE**: This dataset contains comprehensive information on patients seeking cosmetic surgical procedures.

            **DATASET COLUMNS**:
            Patient_ID, First_Name, Last_Name, Age, Gender, Procedure_Requested, Procedure_Areas, Height_ft_in, 
            Weight_lbs, BMI, Medical_History, Current_Medications, Allergies, Previous_Surgeries, 
            Medical_Clearance, Planned_Anesthesia_Type, ASA_Physical_Status

            **PATIENT PROFILE ELEMENTS**:
            1. **Procedure Requested**: {', '.join(sorted(patient_data['Procedure_Requested'].dropna().unique()))}
            2. **Procedure Areas**: {', '.join(procedure_areas) if procedure_areas else "Various anatomical areas"}
            3. **Vital Stats**: Age, Gender, Height, Weight, BMI
            4. **Medical Factors**: Medical_History, Current_Medications, Allergies, Previous_Surgeries
            5. **Surgical Readiness**: Medical_Clearance status, ASA Physical Status, Planned Anesthesia Type

            **ANALYSIS INSTRUCTIONS**:
            - When discussing patients, ALWAYS include full name, age, gender, BMI, procedure requested, and procedure area(s)
            - Prioritize safety by evaluating ASA Physical Status and Medical Clearance
            - Identify any red flags in Medical History, Allergies, or Medications that may impact surgery
            - Consider BMI range appropriateness and potential anesthesia risks
            - For follow-up questions, ONLY refer to patients already mentioned

            **PRIORITY CRITERIA**:
            - Ideal candidates: Cleared medically, ASA I or II, no major comorbidities, realistic BMI
            - Cautionary cases: ASA III or above, pending medical clearance, allergy/anesthesia risk, complex medical history

            **STRICT GUIDELINES**:
            - Maintain strict patient reference continuity across responses
            - Flag any potential surgical contraindications
            - Always discuss anesthesia implications based on ASA status and Medical Clearance
            - Avoid assumptions; base evaluation only on documented patient data
            """   
        
        elif "Hair Loss Stage" in columns and "Scalp Condition" in columns:
            dataset_type = "Hair Restoration"
            
            # Get values from dataset
            hair_loss_stages = patient_data["Hair Loss Stage"].unique().tolist() if "Hair Loss Stage" in columns else []
            scalp_conditions = patient_data["Scalp Condition"].unique().tolist() if "Scalp Condition" in columns else []
            
            relevant_features = f"""
            **DATASET STRUCTURE**: This dataset contains patient information for hair restoration procedures.
            
            **DATASET COLUMNS**:
            Patient ID, First Name, Last Name, Age, email_address, Gender, BMI, Elective Procedure,
            Hair Loss Stage, Scalp Condition, Previous Hair Transplants, Donor Area Quality,
            Family History of Baldness, Hair Type, Expectation from Procedure, 
            Willingness for Multiple Sessions, Post-Procedure Recovery Consideration, 
            Hair Care Routine & Product Sensitivities
            
            **PATIENT PROFILE ELEMENTS**:
            1. **Hair Loss Stage**: {', '.join(map(str, hair_loss_stages)) if hair_loss_stages else "Various stages documented"}
            2. **Scalp Condition**: {', '.join(map(str, scalp_conditions)) if scalp_conditions else "Various conditions noted"}
            3. **Donor Area Quality**: Assessment of available donor hair for transplantation
            4. **Hereditary Factors**: Family History of Baldness
            5. **Treatment History**: Previous Hair Transplants
            6. **Patient Expectations**: Expectation from Procedure, Willingness for Multiple Sessions
            
            **ANALYSIS INSTRUCTIONS**:
            - When discussing patients, ALWAYS include name, age, gender, hair loss stage, and donor quality
            - Evaluate candidacy based on:
            * Pattern and progression of hair loss
            * Donor area availability and quality
            * Scalp condition and flexibility
            * Previous transplant outcomes if applicable
            * Age and family history considerations
            * Hair type and its impact on results
            * Patient's willingness for multiple sessions if needed
            
            **PRIORITY CRITERIA**:
            - Optimal candidates: Stable hair loss, good donor supply, realistic expectations, healthy scalp condition
            - Limited candidates: Advanced hair loss, poor donor quality, active scalp conditions, unrealistic expectations
            
            **STRICT GUIDELINES**:
            - Maintain strict patient continuity across responses
            - Be explicit about procedure limitations based on individual factors
            - For follow-up questions, ONLY reference patients from your most recent response
            - Consider product sensitivities in post-procedure care recommendations
            """
    
        elif "Current Dental Condition" in columns and "Bite Issues" in columns:
            dataset_type = "Cosmetic Dentistry"

            # Extract values from the dataset, ensuring no NaN values
            dental_conditions = patient_data["Current Dental Condition"].dropna().unique().tolist() if "Current Dental Condition" in columns else []
            bite_issues = patient_data["Bite Issues"].dropna().unique().tolist() if "Bite Issues" in columns else []

            # Normalize column names for consistency
            normalized_columns = [col.lower() for col in columns]

            # Build formatted text like Liposuction structure
            relevant_features = f"""
            **DATASET STRUCTURE**: This dataset contains patient information for cosmetic and functional dental procedures.

            **DATASET COLUMNS**:
            Patient ID, First Name, Last Name, email_address, Age, Gender, BMI, Elective Procedure,
            Current Dental Condition, Gum Health, Bite Issues, Tooth Sensitivity Level,
            Aligner Preference, Previous Dental Procedures, Oral Hygiene Habits,
            Expectation from Treatment, Lifestyle Considerations, Allergies to Dental Materials

            **PATIENT PROFILE ELEMENTS**:
            1. **Current Dental Condition**: {', '.join(map(str, dental_conditions)) if dental_conditions else "Various conditions documented"}
            2. **Bite Issues**: {', '.join(map(str, bite_issues)) if bite_issues else "Various issues documented"}
            3. **Oral Health Factors**: Gum Health, Tooth Sensitivity Level, Oral Hygiene Habits
            4. **Treatment History**: {"Available" if "previous dental procedures" in normalized_columns else "Information may be limited"}
            5. **Patient Preferences**: Aligner Preference, Expectation from Treatment, Lifestyle Considerations
            6. **Risk Factors**: Allergies to Dental Materials

            **ANALYSIS INSTRUCTIONS**:
            - When discussing patients, ALWAYS include their full name, age, current dental condition, and primary concern.
            - For follow-up questions, ONLY consider patients mentioned in your previous response.
            - Evaluate optimal treatment approaches based on:
                * Functional vs. cosmetic priorities
                * Oral health foundation (especially gum health)
                * Previous treatment outcomes
                * Patient preferences and constraints
                * Tooth sensitivity level
                * Aligner preferences when applicable
                * Allergies to dental materials

            **PRIORITY CRITERIA**:
            - **High Priority**: Functional issues affecting daily activities (eating, speaking).
            - **Medium Priority**: Combined functional and cosmetic concerns.
            - **Low Priority**: Purely cosmetic improvements with a healthy foundation.

            **STRICT GUIDELINES**:
            - Maintain strict patient continuity in follow-up discussions.
            - For questions about "these patients" or similar phrasing, ONLY reference patients from your most recent response.
            - Be precise about treatment timelines and maintenance requirements.
            - Consider oral hygiene habits when evaluating treatment success likelihood.
            """
        else:
            # Generic medical dataset handling with adaptive detection
            dataset_type = "Medical Procedure Evaluation"
            
            # Try to identify key columns for analysis
            numeric_columns = [col for col in columns if patient_data[col].dtype in [int, float]]
            categorical_columns = [col for col in columns if patient_data[col].dtype == 'object' and col not in ['first_name', 'last_name', 'email']]
            
            # Get some sample values for key fields
            column_samples = {}
            for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
                values = patient_data[col].unique().tolist()
                if len(values) < 10:  # Only include if we don't have too many unique values
                    column_samples[col] = values
            
            relevant_features = f"""
            **DATASET STRUCTURE**: This appears to be a medical dataset containing patient information for procedure evaluation.
            
            **DETECTED COLUMNS**:
            - Identified {len(numeric_columns)} numeric measurement columns: {', '.join(numeric_columns[:5])}...
            - Identified {len(categorical_columns)} categorical attribute columns: {', '.join(categorical_columns[:5])}...
            
            **KEY FIELD EXAMPLES**:
            {chr(10).join([f"- {col}: {', '.join(map(str, values[:5]))}" for col, values in column_samples.items()])}
            
            **ANALYSIS INSTRUCTIONS**:
            - When discussing patients, ALWAYS include their full name and key identifying characteristics
            - For follow-up questions, ONLY consider patients mentioned in your previous response
            - Make evidence-based recommendations using the available data points
            - Be transparent about limitations in the dataset
            
            **STRICT GUIDELINES**:
            - Maintain strict patient continuity across responses
            - NEVER assume information not present in the dataset
            - For follow-up questions about specific patients, verify they appeared in your previous response
            - Be explicit about which data points influenced your analysis
            """
            # Final enhanced prompt construction
        system_prompt = f"""
        You are a specialized medical AI assistant analyzing patient data for {dataset_type} procedures.
        
        {relevant_features}
        
        
        
        **FINAL GUIDELINES**:
        - Prioritize accuracy and patient safety above all else
        - Be concise but thorough in your analysis
        - Always consider medical ethics and best practices
        - When uncertain, clearly acknowledge limitations in the data
        - Answer directly and specifically to the question asked
        """
        
        return system_prompt
    
    def get_optimized_prompt(self):
        """Get the best performing prompt based on feedback."""
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("""
                SELECT prompt_used, AVG(rating) as avg_rating
                FROM feedback
                WHERE rating IS NOT NULL
                GROUP BY prompt_used
                ORDER BY avg_rating DESC
                LIMIT 1
            """)
            result = c.fetchone()
            return result[0] if result else self.get_default_prompt()
        finally:
            conn.close()
    
   
    def extract_patient_names(self, response_text):
        patients = set()
        
        # More comprehensive name patterns
        name_patterns = [
            r'(?:^|[^\w])([A-Z][a-z]+ [A-Z][a-z]+)(?:\s*[\(:\-,]|\s+(?:is|has|with|age|who))',
            r'(?:patient|candidate):\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'(?:^|\n)(?:\d+\.|\*|-)\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)(?=\s*\(\d+\s*(?:year|yr))',
            r'([A-Z][a-z]+ [A-Z][a-z]+),\s*(?:age|aged)\s*\d+'
        ]
        
        for pattern in name_patterns:
            for match in re.finditer(pattern, response_text, re.MULTILINE):
                # Handle different group positioning in regex
                group_index = 1 if len(match.groups()) >= 1 else 0
                if group_index > 0:
                    patients.add(match.group(group_index).strip())
        
        return list(patients)
    def detect_dataset_type(self, patient_data):
        columns = set(patient_data.columns.tolist())
        
        # Define keyword sets for each dataset type
        skin_keywords = {"Skin Type", "Skin Concerns", "Previous Skin Treatments"}
        ortho_keywords = {"Pain Level", "Mobility Issues", "Joint"}
        contour_keywords = {"Target Body Areas", "Body Fat Percentage"}
        hair_keywords = {"Hair Loss Stage", "Scalp Condition", "Donor Area"}
        dental_keywords = {"Dental Condition", "Bite Issues", "Gum Health"}
        
        # Check overlap with each set
        overlaps = {
            "Dermatology": len(columns.intersection(skin_keywords)),
            "Orthopedic": len(columns.intersection(ortho_keywords)),
            "Body Contouring": len(columns.intersection(contour_keywords)),
            "Hair Restoration": len(columns.intersection(hair_keywords)),
            "Cosmetic Dentistry": len(columns.intersection(dental_keywords))
        }
        
        # Return the type with the most keyword matches
        dataset_type = max(overlaps.items(), key=lambda x: x[1])[0]
        if overlaps[dataset_type] > 0:
            return dataset_type
        else:
            return "Unknown Medical Dataset"
    def is_followup_query(self, query):
        """Enhanced detection of follow-up questions using sophisticated pattern matching."""
        # Comprehensive list of follow-up indicators
        followup_indicators = [
            "above", "these patients", "those patients", "from the previous",
            "out of the", "of these", "which of", "among them", "from this list",
            "from that list", "mentioned", "previous", "earlier", "remaining",
            "listed", "above patients", "selected", "filtered", "eliminate",
            "exclude", "remove", "keep", "retain", "among these", "from these",
            "for them", "in this group", "in these results", "in the results",
            "from the result", "within the list", "in the selection",
            "further filter", "refine the list", "narrow down", "shortlist",
            "who among", "which ones", "which patients", "best candidate",
            "these cases", "those cases", "candidates from", "within these",
            "now which", "now who", "top patients", "selected patients"
        ]
       
        # Complex regex patterns for detecting follow-up questions
        followup_patterns = [
            r"(?:from|of|among)(?:\s+the)?(?:\s+\w+)?\s+(?:patients|cases|individuals|candidates|group|list)",
            r"(?:only|just|consider)(?:\s+the)?(?:\s+\w+)?\s+(?:patients|cases|individuals|remaining|mentioned)",
            r"(?:which|who|what|how\s+many)(?:\s+of)?(?:\s+\w+)?\s+(?:patients|them|those|these)",
            r"(?:exclude|include|filter|eliminate|remove)(?:\s+out|in)?(?:\s+\w+)?\s+(?:patients|cases)",
            r"(?:further|more|additional)(?:\s+\w+)?\s+(?:analysis|filter|consideration)",
            r"(?:narrow|reduce|limit|restrict)(?:\s+\w+)?(?:\s+to)?(?:\s+\w+)?\s+(?:patients|cases|selection)",
            r"(?:best|top|optimal|ideal)(?:\s+\w+)?\s+(?:candidate|choice|option|match)",
            r"(?:remain|left|stay)(?:ing|ed)?\s+(?:patients|candidates|options)"
        ]
       
        # Check for simple indicators
        if any(indicator.lower() in query.lower() for indicator in followup_indicators):
            return True
       
        # Check for regex patterns
        for pattern in followup_patterns:
            if re.search(pattern, query.lower()):
                return True
       
        # Context-based follow-up detection
        if len(st.session_state.conversation_context) > 0:
            # If query doesn't have clear "new query" indicators and is short, likely follow-up
            new_query_indicators = [
                "new search", "different patients", "another query", "start over",
                "new analysis", "reset", "begin again", "fresh search"
            ]
           
            if not any(indicator.lower() in query.lower() for indicator in new_query_indicators):
                # Short queries are often follow-ups
                if len(query.split()) < 15:
                    return True
               
                # Queries without specific patient criteria are likely follow-ups
                if not re.search(r"(pain|mobility|level|score|rating|age|gender|diagnosis)", query.lower()):
                    return True
       
        return False
 
    def analyze_patients(self, patient_data, query):
        """Analyze patient data with enhanced context management."""
        try:
            # Get base prompt
            system_prompt = self.get_dynamic_prompt(patient_data)

            # Ensure the system_prompt is not None or empty
            if not system_prompt or system_prompt.strip() == "":
                system_prompt = "You are an AI assistant analyzing patient data for elective surgeries. Provide insights based on the dataset."

            # Limit the number of rows to avoid excessive tokens (e.g., 5 rows)
            max_rows = 53  # You can adjust this based on your needs
            data_context = patient_data.to_string()

           
            # Generate a unique ID for this analysis
            analysis_id = str(uuid.uuid4())
           
            # Detect if this is a follow-up query
            is_followup = self.is_followup_query(query)
           
            # Build conversation history with enhanced context management
            messages = [{"role": "system", "content": system_prompt}]
           
            # Add strong context instruction for the LLM
            context_instruction = """
            CRITICAL CONTEXT MANAGEMENT INSTRUCTIONS:
           
            1. PATIENT REFERENCE RULE: When the user refers to "patients" or "these patients" in a follow-up question,
               they are ALWAYS referring EXCLUSIVELY to the specific patients mentioned in your most recent response.
               
            2. CONTINUITY RULE: Each filtering operation creates a new working set. All subsequent questions
               refer to this filtered set until a new explicit filtering operation is performed.
               
            3. CONSISTENCY RULE: Never introduce patients in a follow-up response that weren't explicitly
               mentioned in your previous response.
               
            4. VERIFICATION RULE: Before including any patient in a response to a follow-up question,
               verify they were explicitly mentioned in your most recent response.
            """
            messages.append({"role": "system", "content": context_instruction})
           
            # Add previous context
            for msg in st.session_state.conversation_context:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
           
            # For follow-up queries, explicitly provide the previously mentioned patients
            if is_followup and st.session_state.mentioned_patients:
                patient_context = "CRITICAL INFORMATION - PREVIOUSLY MENTIONED PATIENTS:\n\n"
                patient_context += "In your last response, you specifically mentioned ONLY these patients:\n"
                for i, patient in enumerate(st.session_state.mentioned_patients, 1):
                    patient_context += f"{i}. {patient}\n"
               
                patient_context += "\nIMPORTANT: The user's current question refers ONLY to these specific patients. "
                patient_context += "DO NOT consider any other patients from the dataset."
               
                messages.append({
                    "role": "system",
                    "content": patient_context
                })
               
                # Add user query with explicit reference to previous patients
                messages.append({
                    "role": "user",
                    "content": f"Regarding ONLY the patients I just listed (and no others), please answer: {query}"
                })
            else:
                # For first queries or explicit new queries, use the full dataset
                messages.append({
                    "role": "user",
                    "content": f"Based on this patient data:\n{data_context}\n\nQuery: {query}"
                })
           # Remove any messages that have None content
            messages = [msg for msg in messages if msg["content"] is not None and msg["content"].strip() != ""]

            if not messages:
                return {"response": "Error: No valid messages to process.", "analysis_id": None}

            # Call the OpenAI API with more robust settings
            response = self.client.chat.completions.create(
                model="gpt-4-32k",  # Can be upgraded to gpt-4 for better context handling
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000,  # Ensure complete responses
                top_p=0.95,       # High precision
                frequency_penalty=0.5,  # Reduce repetition
                presence_penalty=0.5    # Encourage diverse responses
            )
           
            response_content = response.choices[0].message.content if response.choices and response.choices[0].message else "Error: No response generated."
           
            # Extract patient names from response using enhanced methods
            mentioned_patients = self.extract_patient_names(response_content)
           
            # Store mentioned patients for future context
            st.session_state.mentioned_patients = mentioned_patients
           
            # Associate this set of patients with the analysis ID
            st.session_state.filtered_patient_sets[analysis_id] = mentioned_patients
           
            # Update conversation context
            st.session_state.conversation_context.append({
                "role": "user",
                "content": query
            })
            st.session_state.conversation_context.append({
                "role": "assistant",
                "content": response_content
            })
           
            # Limit context history
            if len(st.session_state.conversation_context) > 20:
                st.session_state.conversation_context = st.session_state.conversation_context[-20:]
               
            # Save to database
            conn = get_db()
            try:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO feedback (id, query, response, prompt_used, feedback_requested)
                    VALUES (?, ?, ?, ?, ?)
                """, (analysis_id, query, response_content, system_prompt, False))
                conn.commit()
            finally:
                conn.close()
               
            # Update interaction time for feedback manager
            st.session_state.feedback_manager.update_interaction_time()
           
            return {
                'response': response_content,
                'analysis_id': analysis_id
            }
           
        except Exception as e:
            return {'response': f"Error in analysis: {str(e)}", 'analysis_id': None}
 
    def clear_context(self):
        """Clear all context data."""
        if 'conversation_context' in st.session_state:
            st.session_state.conversation_context = []
        if 'mentioned_patients' in st.session_state:
            st.session_state.mentioned_patients = []
        if 'filtered_patient_sets' in st.session_state:
            st.session_state.filtered_patient_sets = {}
 
    def add_feedback(self, analysis_id, rating=None, counselor_feedback=None, actual_outcome=None):
        """Add comprehensive feedback for an analysis."""
        conn = get_db()
        try:
            c = conn.cursor()
            update_fields = []
            params = []
           
            if rating is not None:
                update_fields.append("rating = ?")
                params.append(rating)
           
            if counselor_feedback is not None:
                update_fields.append("counselor_feedback = ?")
                params.append(counselor_feedback)
           
            if actual_outcome is not None:
                update_fields.append("actual_outcome = ?")
                params.append(actual_outcome)
           
            if update_fields:
                update_fields.append("feedback_timestamp = CURRENT_TIMESTAMP")
                update_fields.append("feedback_requested = TRUE")
                params.append(analysis_id)
               
                query = f"""
                    UPDATE feedback
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                """
                c.execute(query, params)
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
        finally:
            conn.close()

# Streamlit interface
st.set_page_config(page_title="Patient Analysis System", layout="wide")

# CSS for professional and sophisticated UI
st.markdown("""
    <style>
        /* Base colors and variables */
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #3498db;
            --light-bg: #f8f9fa;
            --border-color: #e9ecef;
            --text-color: #2c3e50;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --error-color: #e74c3c;
        }

        /* Overall app styling */
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }

        /* Header styling */
        h1 {
            color: var(--primary-color);
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 500;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--light-bg);
            border-right: 1px solid var(--border-color);
        }

        /* Chat message styling */
        [data-testid="stChatMessage"] {
            background-color: white;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        /* User message styling */
        [data-testid="stChatMessage"][data-testid="user-message"] {
            background-color: #f0f7ff;
            border-color: #d0e3ff;
        }

        /* Feedback container */
        .feedback-container {
            padding: 8px;
            border: 1px solid var(--border-color);
            border-radius: 5px;
            margin: 8px 0;
        }

        /* Debug info */
        .debug-info {
            font-size: 0.75rem;
            color: var(--secondary-color);
            background: var(--light-bg);
            padding: 5px;
            border-radius: 5px;
            margin-top: 8px;
        }

        /* Button styling - more sophisticated and subtle */
        .stButton button {
            border-radius: 4px;
            background-color: white;
            color: var(--text-color);
            border: 1px solid var(--border-color);
            font-weight: 400;
            padding: 0.3rem 0.8rem;
            transition: all 0.2s ease;
            box-shadow: none;
        }

        .stButton button:hover {
            background-color: var(--light-bg);
            border-color: var(--secondary-color);
        }

        /* Export button styling */
        .export-button button {
            color: var(--accent-color);
            border-color: var(--accent-color);
        }

        /* Clear button styling */
        .clear-button button {
            color: var(--error-color);
            border-color: var(--error-color);
        }

        /* Bottom right button */
        .bottom-right-button {
            position: fixed;
            right: 20px;
            bottom: 80px;
            z-index: 999;
        }

        .bottom-right-button button {
            background-color: white;
            color: var(--accent-color);
            border: 1px solid var(--accent-color);
            border-radius: 4px;
            padding: 0.4rem 0.8rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        /* Fixed input container at bottom */
        .fixed-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 15px 20px;
            border-top: 1px solid var(--border-color);
            z-index: 999;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        }

        /* Chat input styling */
        .stChatInput {
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }

        /* Voice button styling */
        .voice-button button {
            border-radius: 4px;
            height: 38px;
            background-color: white;
            color: var(--accent-color);
            border: 1px solid var(--accent-color);
            transition: all 0.2s ease;
        }

        .voice-button button:hover {
            background-color: var(--accent-color);
            color: white;
        }

        /* Success message styling */
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            border-color: #c3e6cb;
            padding: 8px;
            border-radius: 4px;
        }

        /* Warning message styling */
        .stWarning {
            background-color: #fff3cd;
            color: #856404;
            border-color: #ffeeba;
            padding: 8px;
            border-radius: 4px;
        }

        /* Error message styling */
        .stError {
            background-color: #f8d7da;
            color: #721c24;
            border-color: #f5c6cb;
            padding: 8px;
            border-radius: 4px;
        }

        /* Feedback buttons */
        .feedback-button {
            margin: 3px;
        }

        /* Feedback prompt */
        .feedback-prompt {
            background-color: var(--light-bg);
            padding: 10px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            margin: 10px 0;
        }

        /* Input row */
        .input-row {
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: 1200px;
            margin: 0 auto;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 1px dashed var(--border-color);
            border-radius: 4px;
            padding: 10px;
        }

        [data-testid="stFileUploader"] small {
            color: var(--secondary-color);
        }

        /* Checkbox */
        [data-testid="stCheckbox"] {
            color: var(--text-color);
        }

        /* Divider */
        hr {
            margin: 1rem 0;
            border-color: var(--border-color);
            opacity: 0.5;
        }

        /* Add space at bottom to prevent content being hidden behind fixed input */
        .content-container {
            margin-bottom: 100px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.analyzer = PatientAnalyzer()

# if 'voice_input' not in st.session_state:
#     st.session_state.voice_input = VoiceInput()

if 'auto_speak' not in st.session_state:
    st.session_state.auto_speak = False

# Sidebar setup with professional styling
with st.sidebar:
    uploaded_file = st.file_uploader(" ", type="csv")
    
    if uploaded_file:
        try:
            patient_data = pd.read_csv(uploaded_file)
            st.success(f"✓ Data Loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    

    st.markdown('<div style="font-size: 1.2rem; font-weight: 500; color: #2c3e50; margin-bottom: 1rem;">Settings</div>', unsafe_allow_html=True)
    show_debug = st.checkbox("Show Analysis Details", value=False)

    # Clear chat button in sidebar
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.analyzer.clear_context()
        st.rerun()
        # ... inside your existing st.sidebar block

    # # Voice input button
    # if st.button("၊၊||၊", key="voice_input_button", help="Click to speak"):
    #     if uploaded_file is not None:
    #         with st.spinner("Listening..."):
    #             voice_text = st.session_state.voice_input.listen()
    #             if voice_text:
    #                 st.session_state.voice_prompt = voice_text
    #     else:
    #         st.error("Please upload patient data first.")


# Header with export button
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("""
      
            <h1>🏥Elective Surgery Chat Assistant</h1>
        
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="export-button" style="text-align: right;">', unsafe_allow_html=True)
    export_chat_button = st.button("📊 Export Analysis", key="export_chat")
    st.markdown('</div>', unsafe_allow_html=True)

if export_chat_button:
    chat_df = pd.DataFrame(st.session_state.messages)
    csv = chat_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis History",
        data=csv,
        file_name=f"patient_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Content container with bottom margin to prevent content being hidden behind fixed input
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Display chat messages with improved styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show debug info if enabled
        if show_debug and message["role"] == "assistant" and "analysis_id" in message:
            if message["analysis_id"] in st.session_state.filtered_patient_sets:
                patients = st.session_state.filtered_patient_sets[message["analysis_id"]]
                st.markdown(f"""
                    <div class='debug-info'>
                        <strong>Analysis ID:</strong> {message["analysis_id"][:8]}...<br>
                        <strong>Patients analyzed:</strong> {', '.join(patients)}
                    </div>
                """, unsafe_allow_html=True)

        # Feedback UI with improved styling
        if message["role"] == "assistant" and "analysis_id" in message:
            with st.container():
                st.markdown("<hr style='margin: 0.8rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("👍", key=f"up_{message['analysis_id']}"):
                        st.session_state.analyzer.add_feedback(message["analysis_id"], rating=1)
                        st.session_state.feedback_manager.acknowledge_feedback()
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 ", key=f"down_{message['analysis_id']}"):
                        st.session_state.analyzer.add_feedback(message["analysis_id"], rating=0)
                        st.session_state.feedback_manager.acknowledge_feedback()
                        st.success("Thank you for your feedback!")
                with col3:
                    if st.button("Add Details", key=f"detail_{message['analysis_id']}"):
                        with st.expander("Detailed Feedback"):
                            counselor_feedback = st.text_area(
                                "Clinical Feedback",
                                key=f"counselor_{message['analysis_id']}"
                            )
                            actual_outcome = st.text_area(
                                "Actual Outcome",
                                key=f"outcome_{message['analysis_id']}"
                            )
                            if st.button("Submit Details", key=f"submit_{message['analysis_id']}"):
                                st.session_state.analyzer.add_feedback(
                                    message["analysis_id"],
                                    counselor_feedback=counselor_feedback,
                                    actual_outcome=actual_outcome
                                )
                                st.success("Detailed feedback submitted successfully!")

# Automatic feedback request with better styling
if st.session_state.feedback_manager.should_request_feedback():
    st.warning("⏰ Would you mind rating my previous response?")

# Close the content container
st.markdown('</div>', unsafe_allow_html=True)

# Fixed input box at bottom
st.markdown("""
    <div class="fixed-input-container">
        <div class="input-row">
            <div style="flex-grow: 1;"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Outside any layout: safe usage
prompt = st.chat_input("Ask about patients ...")

# # Check if voice input was captured in session and override prompt
# if "voice_prompt" in st.session_state:
#     prompt = st.session_state.voice_prompt
#     del st.session_state.voice_prompt  # Clear after use

# Process input
if prompt:
    st.session_state.feedback_manager.update_interaction_time()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Show user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle clear context command
    if prompt.lower() in ["clear context", "reset context", "start over", "new analysis"]:
        st.session_state.analyzer.clear_context()
        response = "I've cleared our conversation context. We can start a fresh analysis now."
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Regular chat processing
    else:
        if uploaded_file is not None:
            # Dataset is uploaded
            non_medical_keywords = ["binary search", "code", "algorithm", "programming", "javascript", "python", "java", "c++", "sorting"]
            is_explicitly_non_medical = any(keyword in prompt.lower() for keyword in non_medical_keywords)

            if is_explicitly_non_medical:
                # Non-medical programming query with dataset uploaded
                response = "I'm a specialized medical assistant focused on patient analysis for surgery likelihood. I can only answer questions related to the patient data provided or general medical topics."
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Process with dataset as before
                with st.spinner("Analyzing patient data..."):
                    # Add a small delay for UX
                    time.sleep(0.3)
                    result = st.session_state.analyzer.analyze_patients(patient_data, prompt)
                    if isinstance(result, dict):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['response'],
                            "analysis_id": result['analysis_id']
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result
                        })
        else:
            # No dataset uploaded - let OpenAI handle the query
            with st.spinner("Processing your query..."):
                response = st.session_state.analyzer.get_response_without_dataset(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

    # Rerun to update the chat
    st.rerun()