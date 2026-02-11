import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import uuid
from datetime import datetime
from openai import OpenAI
import json
import pandas as pd
from typing import Optional, List, Dict, Any

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType

# CrewAI imports
from crewai import Agent, Task, Crew
from crewai_tools import tool

# Page configuration
st.set_page_config(
    page_title="GtoPdb AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-dark: #0f172a;
        --background-light: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: #334155;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #6366f1;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-left: 4px solid #8b5cf6;
    }
    
    .message-header {
        font-weight: 600;
        color: #6366f1;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .message-content {
        color: #f1f5f9;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* SQL query display */
    .sql-query {
        background: rgba(99, 102, 241, 0.1);
        padding: 1rem;
        border-radius: 12px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        border: 1px solid rgba(99, 102, 241, 0.3);
        overflow-x: auto;
    }
    
    .sql-query code {
        color: #a5b4fc;
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .chat-history-item {
        background: #1e293b;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #334155;
    }
    
    .chat-history-item:hover {
        background: #334155;
        border-color: #6366f1;
        transform: translateX(5px);
    }
    
    .chat-history-item-active {
        background: #334155;
        border-color: #6366f1;
        border-width: 2px;
    }
    
    /* Metrics and stats */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1e293b !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead tr th {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .dataframe tbody tr {
        background-color: #1e293b !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #334155 !important;
    }
    
    .dataframe tbody tr td {
        color: #f1f5f9 !important;
        padding: 0.75rem !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #6366f1 !important;
    }
    
    /* Select boxes */
    .stSelectbox {
        background-color: #1e293b !important;
        border-radius: 12px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #334155;
        border-radius: 8px;
        color: #94a3b8;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8b5cf6;
    }
    
    /* Agent badge */
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .langchain-badge {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
    }
    
    .crewai-badge {
        background: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border: 1px solid #8b5cf6;
    }
</style>
""", unsafe_allow_html=True)


# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection using Streamlit secrets"""
    try:
        conn = psycopg2.connect(
            host=st.secrets["connections"]["postgresql"]["host"],
            port=st.secrets["connections"]["postgresql"]["port"],
            database=st.secrets["connections"]["postgresql"]["database"],
            user=st.secrets["connections"]["postgresql"]["username"],
            password=st.secrets["connections"]["postgresql"]["password"],
            sslmode="require"
        )
        return conn
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return None


def init_database():
    """Initialize database tables for users and chat sessions"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()

        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create chat_sessions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(100) PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                title VARCHAR(255) DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create chat_messages table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                agent_type VARCHAR(20),
                sql_query TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_master_password(password: str) -> bool:
    """Verify master password"""
    return password == st.secrets.get("MASTER_PASSWORD", "")


def register_user(username: str, password: str) -> Optional[int]:
    """Register new user"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            password_hash = hash_password(password)
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING user_id",
                (username, password_hash)
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return user_id
        except psycopg2.IntegrityError:
            conn.rollback()
            return None
        except Exception as e:
            st.error(f"Registration error: {e}")
            return None
    return None


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        password_hash = hash_password(password)
        cur.execute(
            "SELECT user_id, username FROM users WHERE username = %s AND password_hash = %s",
            (username, password_hash)
        )
        user = cur.fetchone()
        cur.close()
        return dict(user) if user else None
    return None


def create_chat_session(user_id: int, title: str = "New Chat") -> str:
    """Create a new chat session"""
    conn = get_db_connection()
    if conn:
        session_id = str(uuid.uuid4())
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (session_id, user_id, title) VALUES (%s, %s, %s)",
            (session_id, user_id, title)
        )
        conn.commit()
        cur.close()
        return session_id
    return ""


def update_session_title(session_id: str, title: str):
    """Update chat session title"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE chat_sessions SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (title, session_id)
        )
        conn.commit()
        cur.close()


def get_user_sessions(user_id: int) -> List[Dict]:
    """Get all chat sessions for a user"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT session_id, title, created_at, updated_at 
               FROM chat_sessions 
               WHERE user_id = %s 
               ORDER BY updated_at DESC""",
            (user_id,)
        )
        sessions = cur.fetchall()
        cur.close()
        return [dict(s) for s in sessions]
    return []


def save_chat_message(session_id: str, role: str, content: str, agent_type: Optional[str] = None, sql_query: Optional[str] = None):
    """Save chat message to database"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO chat_messages (session_id, role, content, agent_type, sql_query)
               VALUES (%s, %s, %s, %s, %s)""",
            (session_id, role, content, agent_type, sql_query)
        )
        # Update session timestamp
        cur.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (session_id,)
        )
        conn.commit()
        cur.close()


def load_chat_messages(session_id: str) -> List[Dict]:
    """Load chat messages for a session"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT role, content, agent_type, sql_query, timestamp
               FROM chat_messages
               WHERE session_id = %s
               ORDER BY timestamp ASC""",
            (session_id,)
        )
        messages = cur.fetchall()
        cur.close()
        return [dict(m) for m in messages]
    return []


def delete_chat_session(session_id: str):
    """Delete a chat session"""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        conn.commit()
        cur.close()


def generate_chat_title(first_message: str) -> str:
    """Generate a title for the chat using OpenAI"""
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, concise title (max 6 words) for this chat based on the user's first message. Only return the title, nothing else."},
                {"role": "user", "content": first_message}
            ],
            max_tokens=20,
            temperature=0.7
        )
        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"').strip("'")
        return title[:50]  # Limit length
    except Exception as e:
        st.error(f"Error generating title: {e}")
        return "New Chat"


@st.cache_resource
def get_langchain_agent():
    """Create LangChain SQL agent"""
    try:
        db_uri = f"postgresql://{st.secrets['connections']['postgresql']['username']}:{st.secrets['connections']['postgresql']['password']}@{st.secrets['connections']['postgresql']['host']}:{st.secrets['connections']['postgresql']['port']}/{st.secrets['connections']['postgresql']['database']}"
        
        db = SQLDatabase.from_uri(db_uri)
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        return agent, db
    except Exception as e:
        st.error(f"Error creating LangChain agent: {e}")
        return None, None


def query_with_langchain(question: str, chat_history: List[Dict]) -> tuple[str, Optional[str]]:
    """Query using LangChain agent"""
    try:
        agent, db = get_langchain_agent()
        if not agent:
            return "Error: Could not initialize LangChain agent", None

        # Build context from chat history
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

        full_question = f"""Previous conversation:
{context}

Current question: {question}

Please answer the question using the GtoPdb database. If you need to query the database, show the SQL query used."""

        response = agent.run(full_question)

        # Try to extract SQL query from response
        sql_query = None
        if "SELECT" in response.upper():
            lines = response.split("\n")
            for line in lines:
                if "SELECT" in line.upper():
                    # Try to extract the full query
                    sql_start = response.upper().find("SELECT")
                    sql_end = response.find(";", sql_start)
                    if sql_end == -1:
                        sql_end = len(response)
                    sql_query = response[sql_start:sql_end].strip()
                    break

        return response, sql_query
    except Exception as e:
        return f"Error: {str(e)}", None


# CrewAI SQL Tool
@tool("query_gtopdb")
def query_gtopdb_tool(query: str) -> str:
    """
    Execute SQL queries on the GtoPdb database.
    
    Args:
        query: SQL query to execute
        
    Returns:
        Query results as a JSON string
    """
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query)
            results = cur.fetchall()
            cur.close()
            return json.dumps([dict(r) for r in results], default=str)
        except Exception as e:
            return f"Query error: {str(e)}"
    return "Database connection error"


def query_with_crewai(question: str, chat_history: List[Dict]) -> tuple[str, Optional[str]]:
    """Query using CrewAI multi-agent system"""
    try:
        # Create SQL Expert Agent
        sql_expert = Agent(
            role='SQL Database Expert',
            goal='Generate accurate SQL queries for the GtoPdb database',
            backstory="""You are an expert in the GtoPdb (Guide to Pharmacology) database schema.
            You understand tables like ligand, interaction, object, species, reference, etc.
            You can write complex SQL queries to extract pharmacological data.""",
            verbose=True,
            allow_delegation=False,
            tools=[query_gtopdb_tool]
        )

        # Create Data Analyst Agent
        analyst = Agent(
            role='Pharmacology Data Analyst',
            goal='Interpret and explain pharmacological data',
            backstory="""You are a pharmacology expert who can interpret drug-target interactions,
            ligand properties, and biological assay data. You provide clear explanations.""",
            verbose=True,
            allow_delegation=False
        )

        # Build context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

        # Create tasks
        sql_task = Task(
            description=f"""Based on the question: "{question}"

Previous context: {context}

Generate an appropriate SQL query for the GtoPdb database and execute it using the query_gtopdb tool.
Explain what the query does and what data it retrieves.""",
            agent=sql_expert,
            expected_output="SQL query with explanation and results"
        )

        analysis_task = Task(
            description=f"""Review the SQL query results and provide a comprehensive answer to: "{question}"

Explain the biological/pharmacological significance of the results in clear, understandable terms.""",
            agent=analyst,
            expected_output="Clear explanation of the data and its significance"
        )

        # Create and run crew
        crew = Crew(
            agents=[sql_expert, analyst],
            tasks=[sql_task, analysis_task],
            verbose=True
        )

        result = crew.kickoff()

        # Extract SQL query if present
        sql_query = None
        result_str = str(result)
        if "SELECT" in result_str.upper():
            sql_start = result_str.upper().find("SELECT")
            sql_end = result_str.find(";", sql_start)
            if sql_end == -1:
                # Try to find end of query
                lines = result_str[sql_start:].split("\n")
                for i, line in enumerate(lines):
                    if line.strip() and not any(keyword in line.upper() for keyword in ["SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER", "HAVING", "LIMIT"]):
                        sql_end = sql_start + result_str[sql_start:].find(line)
                        break
            if sql_end > sql_start:
                sql_query = result_str[sql_start:sql_end].strip()

        return result_str, sql_query

    except Exception as e:
        return f"Error: {str(e)}", None


def execute_sql_query(sql_query: str) -> Any:
    """Execute SQL query and return results"""
    conn = get_db_connection()
    if conn and sql_query:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql_query)
            results = cur.fetchall()
            cur.close()
            return [dict(r) for r in results]
        except Exception as e:
            return f"Query error: {str(e)}"
    return None


# Initialize session state
if 'master_authenticated' not in st.session_state:
    st.session_state.master_authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_type' not in st.session_state:
    st.session_state.agent_type = "LangChain"

# Initialize database
init_database()

# Master password check
if not st.session_state.master_authenticated:
    st.markdown("""
    <div class="main-header">
        <h1>🧬 GtoPdb AI Assistant</h1>
        <p>Multi-Agent Pharmacology Database Chatbot - Secure Access</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="warning-box"><strong>⚠️ Access Control:</strong> Enter the master password to access the system.</div>', unsafe_allow_html=True)

    master_password = st.text_input("Master Password", type="password", key="master_pass")

    if st.button("🔓 Unlock System", use_container_width=True):
        if verify_master_password(master_password):
            st.session_state.master_authenticated = True
            st.rerun()
        else:
            st.error("❌ Invalid master password")

    st.stop()

# User authentication
if not st.session_state.user:
    st.markdown("""
    <div class="main-header">
        <h1>🧬 GtoPdb AI Assistant</h1>
        <p>Multi-Agent Pharmacology Database Chatbot</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    with tab1:
        st.markdown("### Welcome Back!")
        st.markdown('<div class="info-box">Sign in to continue your pharmacology research journey.</div>', unsafe_allow_html=True)
        
        login_username = st.text_input("Username", key="login_user", placeholder="Enter your username")
        login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")

        if st.button("🚀 Login", key="login_btn", use_container_width=True):
            if not login_username or not login_password:
                st.warning("Please enter both username and password")
            else:
                with st.spinner("Authenticating..."):
                    user = authenticate_user(login_username, login_password)
                    if user:
                        st.session_state.user = user
                        st.success(f"✅ Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")

    with tab2:
        st.markdown("### Create Your Account")
        st.markdown('<div class="warning-box"><strong>⚠️ Security Notice:</strong> This is a test environment. DO NOT use a password you use elsewhere. Use simple passwords like "123" or "test123".</div>', unsafe_allow_html=True)
        
        reg_username = st.text_input("Username", key="reg_user", placeholder="Choose a username (min 3 characters)")
        reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Choose a password (min 6 characters)")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm", placeholder="Confirm your password")

        if st.button("📝 Create Account", key="reg_btn", use_container_width=True):
            if not reg_username or not reg_password:
                st.warning("Please fill in all fields")
            elif reg_password != reg_password_confirm:
                st.error("❌ Passwords don't match")
            elif len(reg_password) < 6:
                st.error("❌ Password must be at least 6 characters")
            elif len(reg_username) < 3:
                st.error("❌ Username must be at least 3 characters")
            else:
                with st.spinner("Creating account..."):
                    user_id = register_user(reg_username, reg_password)
                    if user_id:
                        st.success("✅ Account created successfully! Please login.")
                    else:
                        st.error("❌ Username already exists. Please choose another.")

    st.stop()

# Main application
st.markdown("""
<div class="main-header">
    <h1>🧬 GtoPdb AI Assistant</h1>
    <p>Your Intelligent Guide to Pharmacology</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2 style="color: white; margin: 0;">⚙️ Control Panel</h2></div>', unsafe_allow_html=True)
    
    # User info
    st.markdown(f'<div class="info-box"><strong>👤 User:</strong> {st.session_state.user["username"]}</div>', unsafe_allow_html=True)

    # Agent selection
    st.markdown("### 🤖 AI Agent")
    st.session_state.agent_type = st.selectbox(
        "Select Agent",
        ["LangChain", "CrewAI"],
        help="LangChain: Fast SQL agent | CrewAI: Multi-agent system with PostgreSQL tools"
    )
    
    agent_badge_class = "langchain-badge" if st.session_state.agent_type == "LangChain" else "crewai-badge"
    st.markdown(f'<div class="agent-badge {agent_badge_class}">Using {st.session_state.agent_type}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # New chat button
    if st.button("✨ New Chat", use_container_width=True, key="new_chat"):
        # Create new session
        new_session_id = create_chat_session(st.session_state.user['user_id'])
        st.session_state.current_session_id = new_session_id
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💬 Chat History")
    
    # Load user sessions
    sessions = get_user_sessions(st.session_state.user['user_id'])
    
    if sessions:
        for session in sessions:
            is_active = session['session_id'] == st.session_state.current_session_id
            button_class = "chat-history-item-active" if is_active else "chat-history-item"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{'📌 ' if is_active else '💬 '}{session['title'][:30]}...",
                    key=f"session_{session['session_id']}",
                    use_container_width=True
                ):
                    st.session_state.current_session_id = session['session_id']
                    st.session_state.messages = load_chat_messages(session['session_id'])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete chat"):
                    delete_chat_session(session['session_id'])
                    if st.session_state.current_session_id == session['session_id']:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.markdown('<div class="info-box">No chat history yet. Start a new chat!</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.session_state.master_authenticated = False
        st.rerun()

    st.markdown("---")
    
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    <div style="font-size: 0.9rem; color: #94a3b8;">
    • Ask about ligands & targets<br>
    • Request data comparisons<br>
    • Query by drug names<br>
    • Explore interactions<br>
    • View pharmacological data
    </div>
    """, unsafe_allow_html=True)

# Initialize session if needed
if not st.session_state.current_session_id:
    # Create initial session
    st.session_state.current_session_id = create_chat_session(st.session_state.user['user_id'])
    st.session_state.messages = []

# Load messages if not loaded
if not st.session_state.messages and st.session_state.current_session_id:
    st.session_state.messages = load_chat_messages(st.session_state.current_session_id)

# Main chat interface
if not st.session_state.messages:
    st.markdown("""
    <div class="info-box" style="text-align: center; padding: 3rem;">
        <h2 style="color: #6366f1; margin-bottom: 1rem;">👋 Welcome to GtoPdb AI Assistant!</h2>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Ask me anything about pharmacology, drug-target interactions, ligands, or the GtoPdb database.
        </p>
        <p style="color: #6b7280; margin-top: 1rem;">
            <strong>Example queries:</strong><br>
            • "Show me all approved drugs targeting GPCR receptors"<br>
            • "What are the interactions for dopamine?"<br>
            • "List the top 10 ligands by affinity"
        </p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        agent_badge = f'<span class="agent-badge {message.get("agent_type", "").lower()}-badge">{message.get("agent_type", "Assistant")}</span>' if message.get("agent_type") else ""
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">🤖 Assistant {agent_badge}</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show SQL query if available
        if message.get("sql_query"):
            with st.expander("📊 View SQL Query & Results"):
                st.code(message["sql_query"], language="sql")
                
                # Execute and show results
                results = execute_sql_query(message["sql_query"])
                if results:
                    if isinstance(results, list) and len(results) > 0:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        st.markdown(f'<div class="success-box">✅ Retrieved {len(results)} rows</div>', unsafe_allow_html=True)
                    elif isinstance(results, str):
                        st.error(results)
                    else:
                        st.info("No results found")

# Chat input
if prompt := st.chat_input("💬 Ask about the GtoPdb database..."):
    # Check if this is the first message in the session
    is_first_message = len(st.session_state.messages) == 0
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(
        st.session_state.current_session_id,
        "user",
        prompt
    )

    # Display user message immediately
    st.markdown(f"""
    <div class="chat-message user-message">
        <div class="message-header">👤 You</div>
        <div class="message-content">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Generate title for first message
    if is_first_message:
        with st.spinner("Generating chat title..."):
            title = generate_chat_title(prompt)
            update_session_title(st.session_state.current_session_id, title)

    # Generate response
    with st.spinner(f"🤖 {st.session_state.agent_type} agent is thinking..."):
        # Query based on selected agent
        if st.session_state.agent_type == "LangChain":
            response, sql_query = query_with_langchain(prompt, st.session_state.messages)
        else:
            response, sql_query = query_with_crewai(prompt, st.session_state.messages)

        # Display response
        agent_badge = f'<span class="agent-badge {st.session_state.agent_type.lower()}-badge">{st.session_state.agent_type}</span>'
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">🤖 Assistant {agent_badge}</div>
            <div class="message-content">{response}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show SQL query if available
        if sql_query:
            with st.expander("📊 View SQL Query & Results"):
                st.code(sql_query, language="sql")
                
                # Execute and show results
                results = execute_sql_query(sql_query)
                if results:
                    if isinstance(results, list) and len(results) > 0:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        st.markdown(f'<div class="success-box">✅ Retrieved {len(results)} rows</div>', unsafe_allow_html=True)
                    elif isinstance(results, str):
                        st.error(results)
                    else:
                        st.info("No results found")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "agent_type": st.session_state.agent_type,
        "sql_query": sql_query
    })
    save_chat_message(
        st.session_state.current_session_id,
        "assistant",
        response,
        st.session_state.agent_type,
        sql_query
    )

    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; padding: 2rem;">
    <p><strong>🧬 GtoPdb AI Assistant</strong> | Powered by OpenAI, LangChain & CrewAI</p>
    <p style="font-size: 0.9rem;">IUPHAR/BPS Guide to Pharmacology Database</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        <a href="https://www.guidetopharmacology.org" target="_blank" style="color: #6366f1;">Visit GtoPdb</a> | 
        <a href="https://www.guidetopharmacology.org/about.jsp" target="_blank" style="color: #6366f1;">About</a>
    </p>
</div>
""", unsafe_allow_html=True)
