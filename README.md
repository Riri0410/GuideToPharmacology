# GtoPdb AI Assistant

A modern, multi-agent Streamlit application for querying the IUPHAR/BPS Guide to Pharmacology database using AI-powered natural language processing.

## Features

✨ **Modern, Eye-Catching UI**
- Gradient backgrounds and animations
- Responsive design
- Dark theme optimized for long sessions
- Real-time chat interface

🤖 **Dual AI Agent System**
- **LangChain Agent**: Fast SQL query generation and execution
- **CrewAI Multi-Agent**: Advanced multi-agent system with PostgreSQL tools
- Toggle between agents on-the-fly

💬 **Advanced Chat Management**
- Auto-titling of chat sessions using GPT-3.5
- Persistent chat history per user
- View and continue previous conversations
- Delete unwanted chats

🔐 **Security & Access Control**
- Master password protection
- User registration and authentication
- Password hashing (SHA-256)
- Session management

📊 **Data Visualization**
- SQL query display and execution
- Interactive data tables
- Result visualization
- Query result download

👥 **Multi-User Support**
- Concurrent user sessions (4-5 users)
- Individual chat histories
- Isolated user data

## Prerequisites

- Python 3.12
- PostgreSQL database (Neon or other)
- OpenAI API key
- Streamlit Cloud account (for deployment)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd gtopdb-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Secrets

Create a `.streamlit/secrets.toml` file based on the template:

```bash
cp secrets.toml.template .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your credentials:

```toml
# Master password for application access
MASTER_PASSWORD = "your_secure_master_password"

# OpenAI API Key
OPENAI_API_KEY = "sk-..."

# PostgreSQL Database Connection
[connections.postgresql]
host = "your-db-host.neon.tech"
port = 5432
database = "neondb"
username = "your_username"
password = "your_password"
```

### 4. Run Locally

```bash
streamlit run gtopdb_assistant.py
```

The application will be available at `http://localhost:8501`

## Deployment to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Choose `main` branch
6. Set main file path: `gtopdb_assistant.py`
7. Click "Deploy"

### 3. Configure Secrets in Streamlit Cloud

1. Go to your app settings
2. Click on "Secrets"
3. Paste your secrets configuration:

```toml
MASTER_PASSWORD = "your_secure_master_password"
OPENAI_API_KEY = "sk-..."

[connections.postgresql]
host = "your-db-host.neon.tech"
port = 5432
database = "neondb"
username = "your_username"
password = "your_password"
```

4. Save and the app will automatically redeploy

## Database Schema

The application automatically creates the following tables:

### `users`
- `user_id` (SERIAL PRIMARY KEY)
- `username` (VARCHAR(50) UNIQUE)
- `password_hash` (VARCHAR(255))
- `created_at` (TIMESTAMP)

### `chat_sessions`
- `session_id` (VARCHAR(100) PRIMARY KEY)
- `user_id` (INTEGER FOREIGN KEY)
- `title` (VARCHAR(255))
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### `chat_messages`
- `message_id` (SERIAL PRIMARY KEY)
- `session_id` (VARCHAR(100) FOREIGN KEY)
- `role` (VARCHAR(20))
- `content` (TEXT)
- `agent_type` (VARCHAR(20))
- `sql_query` (TEXT)
- `timestamp` (TIMESTAMP)

## Usage Guide

### First Time Setup

1. **Access the Application**
   - Enter the master password (configured in secrets)

2. **Create an Account**
   - Click "Register" tab
   - Enter username and password
   - **⚠️ IMPORTANT**: Use a simple password (e.g., "123" or "test123") as this is a test environment

3. **Login**
   - Enter your credentials
   - Click "Login"

### Using the Chat Interface

1. **Select AI Agent**
   - Use the sidebar dropdown to choose between LangChain and CrewAI
   - LangChain: Faster, direct SQL queries
   - CrewAI: Multi-agent system with more comprehensive analysis

2. **Ask Questions**
   - Type your question in the chat input
   - Press Enter or click send
   - The AI will generate SQL queries and provide results

3. **Example Queries**
   ```
   - "Show me all approved drugs targeting GPCR receptors"
   - "What are the interactions for dopamine?"
   - "List the top 10 ligands by affinity"
   - "Find all ligands that interact with serotonin receptors"
   - "What are the properties of ibuprofen?"
   ```

4. **View Results**
   - Click "View SQL Query & Results" to see the generated SQL
   - Results are displayed in an interactive table
   - Download results as needed

### Managing Chats

- **New Chat**: Click "✨ New Chat" in the sidebar
- **View History**: Click on any previous chat to continue
- **Delete Chat**: Click the 🗑️ icon next to any chat
- **Auto-Titling**: First message automatically generates a descriptive title

## Understanding the GtoPdb Database

The Guide to Pharmacology database contains:

- **Ligands**: Drugs, compounds, metabolites, peptides
- **Targets**: Receptors, enzymes, transporters, ion channels
- **Interactions**: Binding data, affinities, activities
- **References**: Scientific literature citations

### Key Tables

- `ligand`: Drug and compound information
- `object`: Target proteins (receptors, enzymes, etc.)
- `interaction`: Drug-target interactions and binding data
- `species`: Species information
- `reference`: Literature citations

### Sample Queries

The AI agents understand the database schema and can:
- Join multiple tables
- Filter by specific criteria
- Aggregate data
- Sort and limit results
- Handle complex pharmacological queries

## Troubleshooting

### Database Connection Issues

**Problem**: Cannot connect to database
**Solution**: 
- Check your database credentials in secrets.toml
- Ensure database allows connections from Streamlit Cloud IPs
- Verify database is running and accessible

### OpenAI API Errors

**Problem**: API errors or rate limits
**Solution**:
- Verify your OpenAI API key is valid
- Check your API usage and billing
- Ensure you have sufficient credits

### Chat History Not Loading

**Problem**: Previous chats don't appear
**Solution**:
- Ensure database tables are created (run app once to initialize)
- Check database permissions
- Verify user_id is correctly set

### Agent Errors

**Problem**: LangChain or CrewAI errors
**Solution**:
- Check all dependencies are installed
- Verify database connection
- Review error messages in logs
- Try switching to the other agent

## Security Notes

⚠️ **IMPORTANT SECURITY WARNINGS**:

1. **Master Password**: Change the default master password in production
2. **User Passwords**: SHA-256 hashing is used, but this is a demo - DO NOT use real passwords
3. **Database Access**: Limit database user permissions to only necessary tables
4. **API Keys**: Keep your OpenAI API key secure and never commit to git
5. **SSL/TLS**: Ensure database connections use SSL (sslmode=require)

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Auth    │
    │  Layer   │
    └────┬─────┘
         │
    ┌────▼──────────┐
    │   AI Agents   │
    ├───────────────┤
    │  LangChain    │
    │  CrewAI       │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │  PostgreSQL   │
    │  (Neon)       │
    └───────────────┘
```

## Performance Optimization

- **Caching**: Database connections and agents are cached
- **Session Management**: Efficient session state handling
- **Query Optimization**: SQL queries are optimized by AI agents
- **Lazy Loading**: Messages loaded only when needed

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

- **IUPHAR/BPS**: Guide to Pharmacology database
- **OpenAI**: GPT-4 and GPT-3.5 models
- **LangChain**: SQL agent framework
- **CrewAI**: Multi-agent orchestration
- **Streamlit**: Web application framework

## Support

For issues and questions:
- Check the troubleshooting section
- Review Streamlit logs
- Check database logs
- Contact support at your organization

## Version History

### v1.0.0 (2024)
- Initial release
- LangChain integration
- CrewAI integration
- Multi-user support
- Chat history management
- Auto-titling
- Modern UI

## Future Enhancements

- [ ] Export chat conversations
- [ ] Advanced visualization options
- [ ] Query templates and examples
- [ ] Admin dashboard
- [ ] API access
- [ ] Voice input
- [ ] Multi-language support
- [ ] Advanced analytics

---

**Made with ❤️ for Pharmacology Research**
