import sqlite3

# Connect to SQLite
connection = sqlite3.connect("student.db")

# Create cursor object
cursor = connection.cursor()

# Create table
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT (
    NAME VARCHAR(25),
    CLASS VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS INT
);
"""

cursor.execute(table_info)

# Inserting 5 records
records = [
    ('John Doe', '10', 'A', 85),
    ('Jane Smith', '11', 'B', 90),
    ('Emily Davis', '10', 'A', 88),
    ('Michael Brown', '12', 'C', 76),
    ('Jessica White', '11', 'B', 92)
]

# Inserting multiple records into the 'students' table
records = [
    ('John Doe', '10', 'A', 85),
    ('Jane Smith', '11', 'B', 90),
    ('Emily Davis', '10', 'A', 88),
    ('Michael Brown', '12', 'C', 76),
    ('Jessica White', '11', 'B', 92)
]

# Execute the insertion using executemany()
data=cursor.executemany('''INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS) VALUES (?, ?, ?, ?)''', records)

# Commit the changes to the database
connection.commit()


for row in data:
    print(row)
# Commit the changes  
connection.commit()

# Close the connection
connection.close()
 