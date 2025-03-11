from flask import Flask,render_template, request, jsonify
import pyodbc
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain_groq import ChatGroq
from tabulate import tabulate
import os
from dotenv import load_dotenv
import re

app = Flask(__name__)

# Load environment variables from .env file (if using dotenv)
load_dotenv()

def connect_to_sql_server():
    try:
        conn = pyodbc.connect(
            f"Driver={{ODBC Driver 17 for SQL Server}};"
            f"Server={os.getenv('SQL_SERVER')};"
            f"Database={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USER')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
        )
        print("✅ Connection to SQL Server successful!")
        return conn
    except Exception as e:
        print("❌ Error connecting to SQL Server:", str(e))
        return None



# Fetch schema from database
def fetch_schema(conn):
    try:
        query = """
        SELECT 
            TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM 
            INFORMATION_SCHEMA.COLUMNS
        ORDER BY 
            TABLE_NAME, ORDINAL_POSITION;
        """
        schema_df = pd.read_sql_query(query, conn)
        schema_dict = {}
        for _, row in schema_df.iterrows():
            table_name = row["TABLE_NAME"]
            if table_name not in schema_dict:
                schema_dict[table_name] = []
            schema_dict[table_name].append(f"{row['COLUMN_NAME']} ({row['DATA_TYPE']})")
        return schema_dict
    except Exception as e:
        print("Error fetching schema:", str(e))
        return {}


def remove_think_text(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Load the SQL prompt template
prompt_file_path = os.path.join(os.path.dirname(__file__), "sql_prompt2.txt")

def load_prompt(file_path=prompt_file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

templets = load_prompt()

gorq_KEY = os.getenv("gorq_KEY")


# Generate SQL query
def generate_sql_query(user_input, schema_dict):
    schema_str = "\n".join(
        [f"Table: {table}\nColumns:\n- " + "\n- ".join(columns) for table, columns in schema_dict.items()]
    )

    template = templets
    try:
        # Make sure the prompt template includes both 'user_input' and 'schema_str'
        prompt = PromptTemplate(input_variables=["user_input", "schema_str"], template=template)

        llm = ChatGroq(
            model="deepseek-r1-distill-qwen-32b",
            temperature=0.7,
            groq_api_key= gorq_KEY  # Replace with your API key
        )

        # Pass both 'user_input' and 'schema_str' into the LLM pipeline
        runnable = RunnableMap({"output": prompt | llm})
        result = runnable.invoke({"user_input": user_input, "schema_str": schema_str})

        sql_query = remove_think_text(
        result["output"].content.strip()
        .replace("```sql", "")
        .replace("```", "")
        .strip()
        )

        return sql_query

    except Exception as e:
        return f"Error generating SQL query: {str(e)}"


# Execute SQL query
def execute_sql_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# Generate natural language explanation

def generate_natural_language_explanation(query_result, user_input):
    """
    Generates a natural language explanation or formats a table based on query results.
    """
    
    try:
        # Convert the query result to a string and calculate token count
        query_result_str = query_result.to_string(index=False)  # Assuming query_result is a pandas DataFrame
        token_count = len(query_result_str.split())

        # Format query result as a table using tabulate
        query_result_table = tabulate(query_result, headers='keys', tablefmt='grid')

        # Check token count and return large data as a table
        if token_count > 100:
            return {
                "isTable": True,
                "response": query_result.to_dict(orient="records")  # JSON-like format for the frontend
            }

        # Summarized explanation using LLM (if token count is within limit)
        llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=1.5,
        groq_api_key= gorq_KEY  # Replace with your actual API key
        )        
    
        template = """
           You are an AI assistant designed to provide concise and clear answers to users based on their natural language questions. Use the query result provided below to craft your reply.

            Query Result:
            {query_result}

            User Input:
            {user_input}

            Guidelines for your response:

            1. Directly answer the user's question based on the query result, ensuring your reply is natural and user-friendly.
            Assume the query result is already filtered to match the user's request.
            2. Use the tone and format illustrated in this example:
            User Question: Who is the manager of Alok Sahoo?
            Response: Alok Sahoo's manager is Ajeet Anjan Thakur.
            3. If the query result is empty or lacks the requested information, politely indicate the absence of data without implying the query was incorrect.
            Provide your reply based solely on the query result and user input.
        """
        prompt = PromptTemplate(input_variables=["query_result", "user_input"], template=template)
        runnable = RunnableMap({"output": prompt | llm})
        result = runnable.invoke({
            "query_result": query_result_table,
            "user_input": user_input
        })

        return {
            "isTable": False,
            "response": result["output"].content.strip()
        }

    except Exception as e:
        return {
            "isTable": False,
            "response": f"Error generating explanation: {str(e)}"
        }


# Flask routes
@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question")

    # Use your existing connection method
    conn = connect_to_sql_server()  # Ensure this method returns a valid connection
    if not conn:
        return jsonify({"isTable": False, "response": "Error connecting to the database."})

    # Fetch the schema dictionary
    schema_dict = fetch_schema(conn)
    if not schema_dict:
        return jsonify({"isTable": False, "response": "Unable to fetch database schema."})

    # Generate SQL query
    sql_query = generate_sql_query(user_input, schema_dict)
    if "Error" in sql_query:
        return jsonify({"isTable": False, "response": sql_query})
    print(sql_query)

   

    try:
        query_result1 = pd.read_sql_query(sql_query, conn)  # Fetch the query result into a DataFrame
    except Exception as e:
        return jsonify({"isTable": False, "response": f"Error executing SQL query: {str(e)}"})

    # Drop duplicates
    query_result = query_result1.drop_duplicates()

    # Handle empty or invalid query results
    if not isinstance(query_result, pd.DataFrame) or query_result.empty:
        return jsonify({"isTable": False, "response": "No results found."})

    # Replace NaT/NaN with None for JSON serialization
    query_result_cleaned = query_result.where(query_result.notnull(), None)

    # Generate natural language explanation
    explanation = generate_natural_language_explanation(query_result_cleaned, user_input)

    # Ensure the response is JSON serializable
    try:
        json_response = explanation if isinstance(explanation, dict) else explanation.to_dict(orient="records")
        return jsonify(json_response)
    except Exception as e:
        return jsonify({"isTable": False, "response": f"Error generating response: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
