import sqlite3
import time
import os
import atexit
from datetime import datetime
import re

def extract_db_name(image_name):
    pattern = r'ID\d+_MID\d+'
    match = re.search(pattern, image_name)
    
    if match:
        return match.group(0)
    else:
        return None

############################################################################################################################## do not forget to close the db connection
class DatabaseManager:
    def __init__(self, db_dir="db"):
        self.db_name = None
        self.sb_dir = db_dir
        self.conn = None
        self.cursor = None
        atexit.register(self.close)  # Register the 'close' method to be called at exit

    def connect(self, timestamp, db_name= None):
        # Convert the string timestamp to a struct_time object
        timestamp_struct = time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
        # Extract the year from the timestamp
        year = time.strftime('%Y', timestamp_struct)
    
        # Construct the database name
                
        if db_name is None:
            self.db_name = f"MDE_{year}"
        else:
            self.db_name = f"{db_name}_MDE_{year}"
            
        db_path = os.path.join(self.sb_dir, f"{self.db_name}.db")
    
        if not os.path.exists(db_path):
            os.makedirs(self.sb_dir, exist_ok=True)
            print(f"Creating new database: {db_path}")
    
        # Connect to the database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()


    def store_data(self, timestamp, extracted_parameters, table_name, db_name=None):
        data = {'ts': timestamp}
        # Add the 'extracted_parameters' to 'data'  
        data.update(extracted_parameters)

        self.connect(timestamp, db_name)
        try:
            # Try to insert data directly into the table
            self.insert_data(table_name, data)
            print("Data inserted successfully.")
        except sqlite3.OperationalError:
            # If the table doesn't exist, create it and then insert data
            self.create_table(table_name, data)
            self.insert_data(table_name, data)
            print("Data inserted successfully.")
        except sqlite3.IntegrityError:
            print("Data already exists. Updating...")
            self.update_data(table_name, data)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.conn.commit()
           

   
    def create_table(self, table_name, data_dict):
        data_dict = {key.lower(): value for key, value in data_dict.items()}  # Convert keys to lowercase
    
        # Check if there are columns other than 'ts'
        if len(data_dict) == 1 and 'ts' in data_dict:
            columns_with_primary_key = "ts TEXT PRIMARY KEY"
        else:
            columns = ', '.join([f"{key} TEXT" for key in data_dict.keys() if key != 'ts'])  # Exclude 'ts' from columns
            columns_with_primary_key = f"ts TEXT PRIMARY KEY, {columns}"
    
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_with_primary_key})")
        self.conn.commit()

    def insert_data(self, table_name, data_dict):
        """
        This function inserts data into a SQLite table. If a column does not exist, it adds the column to the table.

        Parameters:
        table_name (str): The name of the table where data will be inserted.
        data_dict (dict): A dictionary where the keys are column names and the values are the data to be inserted.
        """

        # Convert keys to lowercase
        data_dict = {key.lower(): value for key, value in data_dict.items()}  

        # Prepare the column names, placeholders for the values, and the values themselves
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join(['?' for _ in data_dict.values()])
        values = tuple(data_dict.values())

        while True:
            try:
                # Try to insert the data into the table
                self.cursor.execute(f"INSERT INTO {table_name} (ts, {columns}) VALUES (?, {placeholders})", (data_dict['ts'],) + values)
                # If successful, break the loop
                break
            except sqlite3.OperationalError as e:
                if 'no column named' in str(e):
                    # Extract the missing column name from the error message
                    missing_column = str(e).split(' ')[-1]
        
                    # Add the missing column to the table
                    print(f"missing_column: {missing_column} ")
                    self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {missing_column} TEXT")
                else:
                    # Re-raise the error if it's not related to a missing column
                    raise e

        # Commit the changes to the database
        self.conn.commit()



    def update_data(self, table_name, data_dict):
        data_dict = {key.lower(): value for key, value in data_dict.items()}  # Convert keys to lowercase
        columns = ', '.join([f"{key} = ?" for key in data_dict.keys() if key != 'ts'])
        values = tuple(data_dict[key] for key in data_dict.keys() if key != 'ts')
    
        try:
            self.cursor.execute(f"UPDATE {table_name} SET {columns} WHERE ts = ?", values + (data_dict['ts'],))
            self.conn.commit()
            print("Data updated successfully.")
        except Exception as e:
            print(f"Error updating data: {e}")

    

    def close(self):
        if self.conn:
            self.conn.close()
