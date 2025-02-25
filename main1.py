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
 
# Email System Functions
def send_email(to_email, subject, body):
    """Send an email using SMTP with proper TLS/SSL handling.
    If DISABLE_EMAIL_API is set, skip sending and return success."""
    if os.getenv("DISABLE_EMAIL_API", "False").lower() in ["true", "1"]:
        st.info("Email API is disabled. Skipping sending email.")
        return True
 
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT", "587")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
 
    if not smtp_server or not smtp_username or not smtp_password:
        st.error("SMTP credentials are missing. Please check your .env file.")
        return False
 
    try:
        smtp_port = int(smtp_port)
    except ValueError:
        st.error("Invalid SMTP_PORT value. It must be a number.")
        return False
 
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_username
    msg["To"] = to_email
 
    try:
        # Use SMTP_SSL if port is 465, else use STARTTLS
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.ehlo()
            server.starttls()
            server.ehlo()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False
 
def send_email_by_name(patient_data, prompt):
    """Extracts full name from prompt, finds email, and sends an email."""
   
    # First, check if required columns exist
    required_columns = ['first_name', 'last_name', 'email_address']
    missing_columns = [col for col in required_columns if col not in patient_data.columns]
   
    if missing_columns:
        return f"⚠️ Missing required columns: {', '.join(missing_columns)}"
   
    # Extract the name from the prompt
    match = re.search(r"send email to ([A-Za-z]+) ([A-Za-z]+)", prompt, re.IGNORECASE)
   
    if not match:
        return "⚠️ Could not extract name from prompt. Please use format: 'Send email to [First Name] [Last Name]'"
   
    first_name, last_name = match.groups()
   
    # Find the matching patient using both first and last name
    matching_patients = patient_data[
        (patient_data['first_name'].str.lower() == first_name.lower()) &
        (patient_data['last_name'].str.lower() == last_name.lower())
    ]
   
    if matching_patients.empty:
        return f"⚠️ No patient found with name: {first_name} {last_name}"
   
    if len(matching_patients) > 1:
        return f"⚠️ Multiple patients found with name: {first_name} {last_name}"
   
    patient = matching_patients.iloc[0]
    full_name = f"{patient['first_name']} {patient['last_name']}"
   
    # Prepare email content
    subject = "Medical Information Update"
    body = f"""Dear {full_name},
 
This is a notification regarding your medical information update.
 
Best regards,
Your Healthcare Team"""
   
    # Send the email
    try:
        if send_email(patient['email_address'], subject, body):
            return f"✓ Email successfully sent to {full_name} at {patient['email_address']}"
        else:
            return f"⚠️ Failed to send email to {full_name}"
    except Exception as e:
        return f"⚠️ Error sending email: {str(e)}"
 
def send_appointment_email(patient, appointment_type, **kwargs):
    """Send appointment-related emails based on type.
    If DISABLE_EMAIL_API is set, skip the API call and return success."""
    if os.getenv("DISABLE_EMAIL_API", "False").lower() in ["true", "1"]:
        st.info("Email API is disabled. Skipping appointment email sending.")
        return True
 
    full_name = f"{patient['first_name']} {patient['last_name']}"
   
    if appointment_type == "new":
        subject = "Appointment Confirmation"
        body = f"""Dear {full_name},
 
Your appointment has been scheduled for {kwargs['date']} at {kwargs['time']}.
 
Best regards,
Your Healthcare Team"""
 
    elif appointment_type == "reschedule":
        subject = "Appointment Rescheduled"
        body = f"""Dear {full_name},
 
Your appointment has been rescheduled from {kwargs['old_date']} to {kwargs['new_date']} at {kwargs['new_time']}.
 
Best regards,
Your Healthcare Team"""
 
    try:
        if send_email(patient['email_address'], subject, body):
            return True
        return False
    except Exception:
        return False
 
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
 
    def get_default_prompt(self):
        return """You are an AI assistant specialized in analyzing patient data for surgery likelihood.
        Consider: pain levels, symptoms, medical history, mobility level, diagnoses, medications.
        Maintain context from previous responses when answering follow-up questions.
        If asked about specific patients mentioned in previous responses, refer back to those patients.
        Prioritize urgent cases based on: severity, pain levels, impact on daily life, risk factors."""
 
    def analyze_patients(self, patient_data, query):
        """Analyze patient data with context awareness."""
        try:
            system_prompt = self.get_optimized_prompt() or self.get_default_prompt()
            data_context = patient_data.to_string()
 
            # Build conversation history
            messages = [{"role": "system", "content": system_prompt}]
           
            # Add previous context
            for msg in st.session_state.conversation_context:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
 
            # Add current query
            messages.append({
                "role": "user",
                "content": f"Based on this patient data:\n{data_context}\n\nQuery: {query}"
            })
 
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )
 
            # Store context
            st.session_state.conversation_context.append({
                "role": "user",
                "content": query
            })
            st.session_state.conversation_context.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
 
            # Limit context history to last 10 messages to prevent token overflow
            if len(st.session_state.conversation_context) > 10:
                st.session_state.conversation_context = st.session_state.conversation_context[-10:]
 
            analysis_id = str(uuid.uuid4())
            conn = get_db()
            try:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO feedback (id, query, response, prompt_used, feedback_requested)
                    VALUES (?, ?, ?, ?, ?)
                """, (analysis_id, query, response.choices[0].message.content, system_prompt, False))
                conn.commit()
            finally:
                conn.close()
 
            st.session_state.feedback_manager.update_interaction_time()
            return {
                'response': response.choices[0].message.content,
                'analysis_id': analysis_id
            }
 
        except Exception as e:
            return {'response': f"Error in analysis: {str(e)}", 'analysis_id': None}
 
    def clear_context(self):
        if 'conversation_context' in st.session_state:
            st.session_state.conversation_context = []
 
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
 
# CSS for improved feedback UI
st.markdown("""
    <style>
        .feedback-container { padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }
        .feedback-button { margin: 5px; }
        .feedback-prompt { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
 
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.analyzer = PatientAnalyzer()
 
# Sidebar setup
with st.sidebar:
    st.title("Elective Surgery AI Assistance")
    uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type="csv")
   
    if uploaded_file:
        try:
            patient_data = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
 
# Main chat interface
st.title("🏥Elective Surgery Chat Assistant")
 
# Display chat messages with enhanced feedback options
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "analysis_id" in message:
            with st.container():
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("👍", key=f"up_{message['analysis_id']}"):
                        st.session_state.analyzer.add_feedback(message["analysis_id"], rating=1)
                        st.session_state.feedback_manager.acknowledge_feedback()
                        st.success("Thank you for your feedback! Your input helps improve our system.")
                with col2:
                    if st.button("👎", key=f"down_{message['analysis_id']}"):
                        st.session_state.analyzer.add_feedback(message["analysis_id"], rating=0)
                        st.session_state.feedback_manager.acknowledge_feedback()
                        st.success("Thank you for your feedback! Your input helps improve our system.")
                with col3:
                    if st.button("Add Details", key=f"detail_{message['analysis_id']}"):
                        counselor_feedback = st.text_area(
                            "Additional Feedback",
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
 
# Automatic feedback request after 1 minute of inactivity
if st.session_state.feedback_manager.should_request_feedback():
    st.warning("⏰ It's been a minute since our last interaction. Would you mind rating my previous response? Your feedback is valuable!")
 
# Chat input with error handling
try:
    if prompt := st.chat_input("Ask about patients or schedule appointments..."):
        st.session_state.feedback_manager.update_interaction_time()
       
        # Process commands
        if prompt.lower().startswith("send email"):
            if uploaded_file is not None:
                result = send_email_by_name(patient_data, prompt)
                st.session_state.messages.append({"role": "assistant", "content": result})
                with st.chat_message("assistant"):
                    st.markdown(result)
            else:
                st.error("Please upload patient data first.")
        else:
            # Regular chat processing
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
 
            with st.chat_message("assistant"):
                if uploaded_file is not None:
                    with st.spinner("Analyzing..."):
                        result = st.session_state.analyzer.analyze_patients(patient_data, prompt)
                        if isinstance(result, dict):
                            st.markdown(result['response'])
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result['response'],
                                "analysis_id": result['analysis_id']
                            })
                        else:
                            st.markdown(result)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result
                            })
                        st.session_state.feedback_manager.update_interaction_time()
                else:
                    st.error("Please upload patient data first.")
 
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
   
# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.analyzer.clear_context()
        st.rerun()
   
    if st.button("Export Chat History", key="export_chat"):
        # Create a DataFrame from messages
        chat_df = pd.DataFrame(st.session_state.messages)
       
        # Convert to CSV
        csv = chat_df.to_csv(index=False)
       
        # Create download button
        st.download_button(
            label="Download Chat History",
            data=csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
 
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Patient Analysis System v1.0 |
    </div>
    """,
    unsafe_allow_html=True
)