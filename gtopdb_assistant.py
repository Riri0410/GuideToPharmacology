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
import tiktoken

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType

# CrewAI imports
from crewai import Agent, Task, Crew
from crewai_tools import tool

# Token limits
MAX_USER_INPUT_TOKENS = 500
MAX_CHAT_HISTORY_MESSAGES = 3
MAX_QUERY_RESULTS = 50

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to rough estimation
        return len(text.split()) * 1.3

# Page configuration
st.set_page_config(
    page_title="GtoPdb AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS - Dark ChatGPT/Claude-like Theme
st.markdown("""
<style>
    /* Main theme colors - Darker palette */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-darkest: #0a0a0a;
        --background-dark: #1a1a1a;
        --background-medium: #2a2a2a;
        --background-light: #3a3a3a;
        --text-primary: #ececec;
        --text-secondary: #9ca3af;
        --border-color: #404040;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    /* Global styles */
    .stApp {
        background: #0a0a0a;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e1e1e 0%, #2a2a2a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #404040;
    }
    
    .main-header h1 {
        color: #ececec;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #ececec !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: #2a2a2a !important;
        color: #ececec !important;
        font-weight: 500 !important;
        border: 1px solid #404040 !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stButton button:hover {
        background: #3a3a3a !important;
        border-color: #6366f1 !important;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        animation: slideIn 0.2s ease;
        border: 1px solid #404040;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: #2a2a2a;
        border-left: 3px solid #6366f1;
    }
    
    .assistant-message {
        background: #1a1a1a;
        border-left: 3px solid #8b5cf6;
    }
    
    .message-header {
        font-weight: 600;
        color: #9ca3af;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .message-content {
        color: #ececec;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* SQL query display */
    .sql-query {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        border: 1px solid #404040;
        overflow-x: auto;
    }
    
    .sql-query code {
        color: #a5b4fc;
        font-size: 0.85rem;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0a0a0a !important;
        border-right: 1px solid #404040 !important;
    }
    
    .sidebar-header {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #404040;
    }
    
    .chat-history-item {
        background: #1a1a1a;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #404040;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .chat-history-item:hover {
        background: #2a2a2a;
        border-color: #6366f1;
    }
    
    .chat-history-item-active {
        background: #2a2a2a;
        border-color: #6366f1;
        border-width: 2px;
    }
    
    /* Metrics and stats */
    .metric-card {
        background: #1a1a1a;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #404040;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #6366f1;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #404040 !important;
    }
    
    .dataframe thead tr th {
        background-color: #2a2a2a !important;
        color: #ececec !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid #404040 !important;
    }
    
    .dataframe tbody tr {
        background-color: #1a1a1a !important;
        border-bottom: 1px solid #404040 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #2a2a2a !important;
    }
    
    .dataframe tbody tr td {
        color: #ececec !important;
        padding: 0.75rem !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ececec;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 3px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ececec;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ececec;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
        border: 1px solid #404040 !important;
        color: #ececec !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #6366f1 !important;
    }
    
    /* Select boxes */
    .stSelectbox {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1a1a !important;
        border-color: #404040 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        border-radius: 8px;
        color: #9ca3af;
        padding: 0.6rem 1.25rem;
        font-weight: 500;
        border: 1px solid #404040;
    }
    
    .stTabs [aria-selected="true"] {
        background: #6366f1;
        color: white;
        border-color: #6366f1;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #404040;
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6366f1;
    }
    
    /* Agent badge */
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .langchain-badge {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid #3b82f6;
    }
    
    .crewai-badge {
        background: rgba(139, 92, 246, 0.15);
        color: #a78bfa;
        border: 1px solid #8b5cf6;
    }
    
    /* Fix button spacing in sidebar */
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.5rem;
    }
    
    /* Welcome message */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background: #1a1a1a;
        border-radius: 12px;
        border: 1px solid #404040;
        margin: 2rem 0;
    }
    
    .welcome-container h2 {
        color: #6366f1;
        margin-bottom: 1rem;
    }
    
    .welcome-container p {
        color: #9ca3af;
        line-height: 1.8;
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
    """Create LangChain SQL agent with comprehensive database schema knowledge"""
    try:
        db_uri = f"postgresql://{st.secrets['connections']['postgresql']['username']}:{st.secrets['connections']['postgresql']['password']}@{st.secrets['connections']['postgresql']['host']}:{st.secrets['connections']['postgresql']['port']}/{st.secrets['connections']['postgresql']['database']}"
        
        # Only include essential tables to reduce context size
        essential_tables = [
            'ligand', 'ligand_physchem', 'ligand_structure', 'interaction', 
            'object', 'species', 'reference', 'ligand2family', 'object2family',
            'ligand2synonym', 'interaction_affinity_refs'
        ]
        
        db = SQLDatabase.from_uri(
            db_uri,
            include_tables=essential_tables,
            sample_rows_in_table_info=1
        )
        
        # Use GPT-4o (128k context) instead of GPT-4 (8k context)
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=st.secrets["OPENAI_API_KEY"],
            max_tokens=2000
        )

        # Create custom system prompt with schema knowledge
        system_prefix = f"""You are a GtoPdb (Guide to Pharmacology) database expert.

CRITICAL DATABASE SCHEMA KNOWLEDGE:

1. LIGAND PROPERTIES:
   - Basic info: ligand table (ligand_id, name, approved, type)
   - Molecular weight: ligand_physchem.molecular_weight
   - Other physicochemical properties: ligand_physchem (h_bond_acceptors, h_bond_donors, lipinski_s_rule_of_five, etc.)
   - Chemical structure: ligand_structure (smiles, inchi, inchikey)
   - Join: ligand LEFT JOIN ligand_physchem USING(ligand_id)

2. DRUG-TARGET INTERACTIONS:
   - interaction table links ligand_id to object_id (target)
   - Affinity: Use COALESCE(affinity_median, affinity_high, affinity_low) for affinity values
   - Species: interaction.species_id links to species table
   - References: interaction_affinity_refs links to reference table

3. TARGETS (OBJECTS):
   - object table has target info (object_id, name, type)
   - Gene info: structural_info table (object_id links to object)
   - Target families: object2family links to family table

4. COMMON QUERIES:
   - Ligands by molecular weight: SELECT l.*, lp.molecular_weight FROM ligand l JOIN ligand_physchem lp ON l.ligand_id=lp.ligand_id WHERE lp.molecular_weight > [value]
   - Approved drugs: WHERE ligand.approved = true
   - Drug-target interactions: JOIN interaction ON ligand.ligand_id = interaction.ligand_id JOIN object ON interaction.object_id = object.object_id

RULES:
- ALWAYS LIMIT results to {MAX_QUERY_RESULTS} rows
- Use specific column names, NOT SELECT *
- JOIN ligand_physchem when querying molecular properties
- Show the SQL query in your response
- Keep explanations concise"""

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            agent_executor_kwargs={"handle_parsing_errors": True},
            prefix=system_prefix
        )

        return agent, db
    except Exception as e:
        st.error(f"Error creating LangChain agent: {e}")
        return None, None


def query_with_langchain(question: str, chat_history: List[Dict]) -> tuple[str, Optional[str]]:
    """Query using LangChain agent with token management"""
    try:
        agent, db = get_langchain_agent()
        if not agent:
            return "Error: Could not initialize LangChain agent", None

        # Limit chat history to last N messages to save tokens
        recent_history = chat_history[-MAX_CHAT_HISTORY_MESSAGES:] if len(chat_history) > MAX_CHAT_HISTORY_MESSAGES else chat_history
        context = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in recent_history])  # Truncate each message

        full_question = f"""Previous conversation (recent):
{context}

Current question: {question}

Important: 
1. LIMIT query results to {MAX_QUERY_RESULTS} rows maximum
2. Only select essential columns, not all columns with SELECT *
3. Keep the response concise
4. Show the SQL query you used"""

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
                    # Add LIMIT if not present
                    if "LIMIT" not in sql_query.upper():
                        sql_query += f" LIMIT {MAX_QUERY_RESULTS}"
                    break

        return response, sql_query
    except Exception as e:
        return f"Error: {str(e)}", None


# CrewAI SQL Tool
@tool("query_gtopdb")
def query_gtopdb_tool(query: str) -> str:
    """
    Execute SQL queries on the GtoPdb database with result limits.
    
    Args:
        query: SQL query to execute (will be limited to 50 rows)
        
    Returns:
        Query results as a JSON string
    """
    conn = get_db_connection()
    if conn:
        try:
            # Add LIMIT if not present
            if "LIMIT" not in query.upper():
                query = query.rstrip(";") + f" LIMIT {MAX_QUERY_RESULTS}"
            
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query)
            results = cur.fetchall()
            cur.close()
            
            # Limit results
            limited_results = results[:MAX_QUERY_RESULTS]
            return json.dumps([dict(r) for r in limited_results], default=str)
        except Exception as e:
            return f"Query error: {str(e)}"
    return "Database connection error"


def query_with_crewai(question: str, chat_history: List[Dict]) -> tuple[str, Optional[str]]:
    """Query using CrewAI multi-agent system with comprehensive database knowledge"""
    try:
        # Create SQL Expert Agent with detailed schema knowledge
        sql_expert = Agent(
            role='GtoPdb SQL Database Expert',
            goal='Generate accurate SQL queries for the GtoPdb pharmacology database',
            backstory=f"""You are an expert in the GtoPdb (Guide to Pharmacology) database schema.

KEY SCHEMA KNOWLEDGE:

1. LIGAND PROPERTIES (most important!):
   - ligand table: basic info (ligand_id, name, type, approved)
   - ligand_physchem table: molecular_weight, h_bond_acceptors, h_bond_donors, lipinski_s_rule_of_five
   - ligand_structure table: smiles, inchi, inchikey
   - ALWAYS JOIN ligand_physchem when querying molecular weight or physicochemical properties!
   
   Example: SELECT l.name, lp.molecular_weight FROM ligand l 
            JOIN ligand_physchem lp ON l.ligand_id = lp.ligand_id 
            WHERE lp.molecular_weight > 500 LIMIT {MAX_QUERY_RESULTS}

2. DRUG-TARGET INTERACTIONS:
   - interaction table links ligand_id to object_id (target protein)
   - Affinity: COALESCE(affinity_median, affinity_high, affinity_low) AS affinity
   - interaction.species_id → species table
   - interaction_affinity_refs → reference table

3. TARGETS (OBJECTS):
   - object table: target proteins (object_id, name, type like 'GPCR', 'Enzyme', etc.)
   - structural_info: gene names and chromosomal info
   - object2family → family: protein families

4. APPROVED DRUGS:
   - WHERE ligand.approved = true

RULES:
- LIMIT ALL queries to {MAX_QUERY_RESULTS} rows
- Use specific columns, avoid SELECT *
- JOIN tables properly (especially ligand_physchem for molecular properties)
- Return concise results""",
            verbose=True,
            allow_delegation=False,
            tools=[query_gtopdb_tool]
        )

        # Create Data Analyst Agent
        analyst = Agent(
            role='Pharmacology Data Analyst',
            goal='Interpret GtoPdb pharmacological data clearly and concisely',
            backstory="""You are a pharmacology expert who interprets drug-target interactions,
            ligand properties, and biological data. You provide clear, concise explanations.""",
            verbose=True,
            allow_delegation=False
        )

        # Limit chat history to save tokens
        recent_history = chat_history[-MAX_CHAT_HISTORY_MESSAGES:] if len(chat_history) > MAX_CHAT_HISTORY_MESSAGES else chat_history
        context = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in recent_history])

        # Create tasks
        sql_task = Task(
            description=f"""Question: "{question}"

Recent context: {context}

Generate and execute a SQL query for GtoPdb:
1. If asking about molecular weight, JOIN ligand_physchem table
2. If asking about drug-target interactions, JOIN interaction and object tables
3. LIMIT to {MAX_QUERY_RESULTS} rows
4. Select only essential columns
5. Execute using query_gtopdb tool
6. Explain what the query does briefly""",
            agent=sql_expert,
            expected_output="SQL query with brief explanation and results"
        )

        analysis_task = Task(
            description=f"""Review the SQL results and answer: "{question}"

Provide a concise, clear answer focusing on key findings.""",
            agent=analyst,
            expected_output="Clear, concise answer"
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
                # Add LIMIT if not present
                if "LIMIT" not in sql_query.upper():
                    sql_query += f" LIMIT {MAX_QUERY_RESULTS}"

        return result_str, sql_query

    except Exception as e:
        return f"Error: {str(e)}", None


def execute_sql_query(sql_query: str) -> Any:
    """Execute SQL query and return limited results"""
    conn = get_db_connection()
    if conn and sql_query:
        try:
            # Add LIMIT if not present
            if "LIMIT" not in sql_query.upper():
                sql_query = sql_query.rstrip(";") + f" LIMIT {MAX_QUERY_RESULTS}"
            
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql_query)
            results = cur.fetchall()
            cur.close()
            
            # Ensure we don't return more than MAX_QUERY_RESULTS
            limited_results = results[:MAX_QUERY_RESULTS]
            return [dict(r) for r in limited_results]
        except Exception as e:
            return f"Query error: {str(e)}"
    return None


# Initialize session state
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

# User authentication (no master password gate - only for registration)
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
        st.markdown('<div class="warning-box"><strong>🔒 Registration Requires Master Password:</strong> Only authorized users with the master password can create accounts. Contact your administrator for the master password.</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box"><strong>⚠️ Security Notice:</strong> This is a test environment. DO NOT use a password you use elsewhere. Use simple passwords like "123" or "test123".</div>', unsafe_allow_html=True)
        
        reg_master_password = st.text_input("Master Password (Required)", type="password", key="reg_master_pass", placeholder="Enter master password to create account")
        reg_username = st.text_input("Username", key="reg_user", placeholder="Choose a username (min 3 characters)")
        reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Choose a password (min 6 characters)")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm", placeholder="Confirm your password")

        if st.button("📝 Create Account", key="reg_btn", use_container_width=True):
            if not reg_master_password:
                st.error("❌ Master password is required to create accounts")
            elif not verify_master_password(reg_master_password):
                st.error("❌ Invalid master password. Contact your administrator.")
            elif not reg_username or not reg_password:
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
            
            # Use container for better layout control
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Truncate title to prevent overflow
                    display_title = session['title'][:35] + "..." if len(session['title']) > 35 else session['title']
                    button_label = f"{'📌 ' if is_active else '💬 '}{display_title}"
                    
                    if st.button(
                        button_label,
                        key=f"session_{session['session_id']}",
                        use_container_width=True
                    ):
                        st.session_state.current_session_id = session['session_id']
                        st.session_state.messages = load_chat_messages(session['session_id'])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete"):
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

# Initialize session if needed (but don't create empty chats)
if not st.session_state.current_session_id:
    # Don't create session yet - wait for first message
    pass
elif st.session_state.current_session_id and not st.session_state.messages:
    # Load messages for existing session
    st.session_state.messages = load_chat_messages(st.session_state.current_session_id)

# Main chat interface
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <h2>👋 Welcome to GtoPdb AI Assistant</h2>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">
            Ask me anything about pharmacology, drug-target interactions, ligands, or the GtoPdb database.
        </p>
        <div style="text-align: left; max-width: 600px; margin: 0 auto;">
            <p style="color: #6366f1; font-weight: 600; margin-bottom: 0.5rem;">Example queries:</p>
            <p style="margin-left: 1rem;">
                • "Show me ligands with molecular weight greater than 500"<br>
                • "What are the approved drugs targeting GPCR receptors?"<br>
                • "Find interactions for dopamine"<br>
                • "List the top 10 ligands by affinity"
            </p>
        </div>
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
    # Validate token count
    token_count = count_tokens(prompt)
    
    if token_count > MAX_USER_INPUT_TOKENS:
        st.error(f"❌ Your message is too long ({token_count} tokens). Please keep it under {MAX_USER_INPUT_TOKENS} tokens (approximately {int(MAX_USER_INPUT_TOKENS / 1.3)} words).")
        st.stop()
    
    # Create session if this is the very first message ever
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = create_chat_session(st.session_state.user['user_id'])
        st.session_state.messages = []
    
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
