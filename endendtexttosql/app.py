import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os 
import sqlite3    
load_dotenv()
## configure our api key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def get_response(question,prompt):
     model=genai.GenerativeModel("gemini-pro")
     response=model.generate_content([prompt[0],question])
     return response.text


def read_sql_query(sql,db):
     conn=sqlite3.connect(db)
     conn.cursor()
     cur=conn.execute(sql)
     rows=cur.fetchall()
     conn.commit()
     conn.close()
     for row in rows:
          print(row)
     return rows


prompt = ['''You are an expert in converting English questions to SQL query!
The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION, MARKS.

For example,
Example 1 - Insert the following records into the STUDENT table:
    ('John Doe', '10', 'A', 85),
    ('Jane Smith', '11', 'B', 90),
    ('Emily Davis', '10', 'A', 88),
    ('Michael Brown', '12', 'C', 76),
    ('Jessica White', '11', 'B', 92).
The SQL command will be something like this:
    INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS) VALUES 
    ('John Doe', '10', 'A', 85), 
    ('Jane Smith', '11', 'B', 90),
    ('Emily Davis', '10', 'A', 88),
    ('Michael Brown', '12', 'C', 76),
    ('Jessica White', '11', 'B', 92);

Example 2 - How many students are in Class 10?
The SQL command will be something like this:
    SELECT COUNT(*) FROM STUDENT WHERE CLASS = "10";

Also, the SQL code should not have ``` in the beginning or end, and the word 'sql' should not appear in the prompt.
'''
]


 
st.set_page_config(page_title="I can retrieve all the")
st.header("Student Data")

questions=st.text_input("Input :",key="input")

submit=st.button("Ask the question")

if submit:
      response=get_response(questions,prompt)
      st.write(response)
      data=read_sql_query(response,"student.db")
      st.subheader("Results")
      for row in data:
           
           print(row)
           st.header(row)
