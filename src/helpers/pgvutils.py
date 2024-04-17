from sqlalchemy import create_engine, inspect
import psycopg2
import urllib.parse

def list_database_tables(connection_string: str):
    """
    List all table names in the database specified by the connection string.
    
    Args:
    - connection_string (str): Database connection string.
    
    Returns:
    - None
    """
    engine = create_engine(connection_string)

    inspector = inspect(engine)

    table_names = inspector.get_table_names()
    print("Available tables:")
    for table_name in table_names:
        print(table_name)

def drop_all_data_in_postgres(connection_string: str):
    """
    Drops all tables and data in the specified PostgreSQL database.
    
    Args:
        connection_string (str): The connection string to the PostgreSQL database.
    """
    conn = None  # Initialize the conn variable
    try:
        # Parse the connection string
        params = urllib.parse.urlparse(connection_string)
        conn = psycopg2.connect(
            dbname=params.path[1:],
            user=params.username,
            password=params.password,
            host=params.hostname,
            port=params.port
        )
        cur = conn.cursor()  # You were missing this line to initialize the cursor
        cur.execute("""
            DO $$ DECLARE
                r RECORD;
            BEGIN
                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = current_schema()) LOOP
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
                END LOOP;
            END $$;
        """)
        conn.commit()
        print("All data has been dropped successfully.")
    except Exception as e:
        print(f"Failed to drop data: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

# drop_all_data_in_postgres("postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres")

# connection_string = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres"
# list_database_tables(connection_string)
