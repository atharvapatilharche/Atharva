import streamlit as st
from typing import Literal, Tuple, Dict, Optional
import os
import time
import json
import requests
import PyPDF2
from datetime import datetime, timedelta
import pytz

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.email import EmailTools
from phi.tools.zoom import ZoomTool
from phi.utils.log import logger
from streamlit_pdf_viewer import pdf_viewer

# Custom Zoom Tool class with credentials passed
class CustomZoomTool(ZoomTool):
    def __init__(self, *, account_id: Optional[str] = None, 
                 client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None):
        super().__init__(account_id=account_id, client_id=client_id, client_secret=client_secret)
        self.token_url = "https://zoom.us/oauth/token"
        self.access_token = None
        self.token_expires_at = 0

# Role requirements
ROLE_REQUIREMENTS = {
    "ai_ml_engineer": """
        Required Skills:
        - Python, PyTorch/TensorFlow
        - Machine Learning algorithms
        - Deep Learning
        - MLOps
        - RAG, LLM, Prompt Engineering
    """,
    "frontend_engineer": """
        Required Skills:
        - JavaScript, React.js
        - HTML/CSS
        - UI/UX Design
        - Responsive Design
        - State Management (e.g., Redux)
    """,
    "backend_engineer": """
        Required Skills:
        - Python, Django/Flask
        - REST APIs
        - Database Management
        - Cloud Services
        - Performance Optimization
    """
}

# Initialize session state
def init_session_state():
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "candidate_email" not in st.session_state:
        st.session_state.candidate_email = "candidate@example.com"
    if "email_sender" not in st.session_state:
        st.session_state.email_sender = "recruiter@example.com"

# Extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Create resume analyzer agent
def create_resume_analyzer() -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        description="Expert technical recruiter",
        instructions=[
            "Analyze resume against requirements",
            "Consider project experience",
            "Value hands-on experience",
            "Return JSON response"
        ]
    )

# Create email agent
def create_email_agent() -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[EmailTools(
            receiver_email=st.session_state.candidate_email,
            sender_email=st.session_state.email_sender
        )],
        instructions=[
            "Draft professional emails",
            "Maintain friendly tone",
            "Use standardized signature"
        ]
    )

# Create scheduler agent with Zoom credentials
def create_scheduler_agent() -> Agent:
    zoom_tool = CustomZoomTool(
        account_id="IPjy-aNsREyiuJvmg75pRA",  # Set your Zoom account ID
        client_id="XLBZsBsOSj246Fp4K9RnDw",  # Set your Zoom client ID
        client_secret="oSWjiU1rkoM6m3RzIwzL8rIMAFqN82Eu"  # Set your Zoom client secret
    )
    return Agent(
        name="Interview Scheduler",
        model=OpenAIChat(id="gpt-4o"),
        tools=[zoom_tool],
        instructions=[
            "Schedule during business hours",
            "Create proper meeting details",
            "Handle scheduling errors"
        ]
    )

# Analyze resume
def analyze_resume(resume_text: str, role: str, analyzer: Agent) -> Tuple[bool, str]:
    try:
        response = analyzer.run(f"""
        Analyze against requirements:
        {ROLE_REQUIREMENTS[role]}
        Resume: {resume_text}
        Return JSON with decision and feedback
        """)
        return response["selected"], response["feedback"]
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        return False, "Error in analysis. Please try again."

# Send selection email
def send_selection_email(email_agent: Agent, to_email: str, role: str):
    email_agent.run(f"""
    Send congratulatory email for {role} position
    Include next steps and interview info
    """)

# Send rejection email
def send_rejection_email(email_agent: Agent, to_email: str, feedback: str):
    email_agent.run(f"""
    Send constructive feedback
    Include learning resources
    Be empathetic and encouraging
    """)

# Schedule interview
def schedule_interview(scheduler: Agent, email: str, role: str):
    try:
        meeting_response = scheduler.run(f"""
        Schedule 60-minute technical interview
        Title: '{role} Technical Interview'
        Use IST timezone
        Include meeting details
        """)
        st.write(f"Interview scheduled successfully: {meeting_response}")
    except Exception as e:
        logger.error(f"Error scheduling interview: {e}")
        st.error("Error scheduling interview. Please try again.")

# Main function
def main():
    st.title("AI Recruitment System")
    init_session_state()
    
    with st.sidebar:
        st.header("Configuration")
        # Add sidebar configurations if needed

    # Define variables and agents
    role = st.selectbox("Select Role", list(ROLE_REQUIREMENTS.keys()))
    resume_analyzer = create_resume_analyzer()
    scheduler_agent = create_scheduler_agent()

    # File upload for resume
    resume_file = st.file_uploader("Upload Resume", type=["pdf"])
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        st.session_state.resume_text = resume_text

    # Analyze resume button
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing..."):
            is_selected, feedback = analyze_resume(
                st.session_state.resume_text,
                role,
                resume_analyzer
            )
            st.write(f"Selected: {is_selected}")
            st.write(f"Feedback: {feedback}")

    # Schedule interview button
    if st.button("Schedule Interview"):
        with st.spinner("Scheduling..."):
            schedule_interview(
                scheduler_agent,
                st.session_state.candidate_email,
                role
            )

# Run the app
if __name__ == "__main__":
    main()
