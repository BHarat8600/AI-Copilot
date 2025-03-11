# SQL-Powered Flask Chatbot

## Overview
This project is a Flask-based chatbot that integrates with an SQL Server database to generate and execute SQL queries using natural language input. The application leverages LangChain and Groq's AI models to translate user queries into SQL, execute them, and provide results in either tabular or natural language format.

## Features
- Connects to an SQL Server database using `pyodbc`.
- Fetches database schema dynamically.
- Uses LangChain and Groq's LLM models to generate SQL queries from user input.
- Executes queries and returns results in tabular format or natural language.
- Flask API with endpoints for user queries.

## Technologies Used
- **Flask**: Backend framework for handling API requests.
- **SQL Server (pyodbc)**: Database connection and querying.
- **Pandas**: Data handling and transformation.
- **LangChain**: Prompt templates and AI model execution.
- **Groq LLM**: AI model for generating SQL queries and responses.
- **Dotenv**: Environment variable management.
- **Tabulate**: Formatting tabular data.
- **Regular Expressions (re)**: Text processing.

## Installation
### Prerequisites
- Python 3.8+
- SQL Server with appropriate credentials
- Groq API key
  Refer Following Link TO Get Free Groq Key [Gorq_Key](https://console.groq.com/playground)

