import psycopg2
from psycopg2 import sql


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config

    # ---------------------------------------------------
    # Create database connection
    # ---------------------------------------------------

    def get_connection(self):

        try:
            return psycopg2.connect(**self.db_config)

        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

    # ---------------------------------------------------
    # Get schema details
    # ---------------------------------------------------

    def schema_details(self, schema_name):

        connection = None

        try:

            connection = self.get_connection()

            if connection is None:
                return "Database connection is not available."

            schema_info_context = (
                f"Database Schema: {schema_name}\n"
            )

            with connection.cursor() as cursor:

                # -----------------------------------------
                # Get tables
                # -----------------------------------------

                cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                    """,
                    (schema_name,)
                )

                tables_list = cursor.fetchall()

                # -----------------------------------------
                # Process tables
                # -----------------------------------------

                for table in tables_list:

                    table_name = table[0]

                    schema_info_context += (
                        f"\nTable: {table_name}\n"
                    )

                    # -------------------------------------
                    # Get columns
                    # -------------------------------------

                    cursor.execute(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = %s
                        AND table_name = %s
                        ORDER BY ordinal_position;
                        """,
                        (schema_name, table_name)
                    )

                    columns_list = cursor.fetchall()

                    for column in columns_list:

                        column_name = column[0]
                        data_type = column[1]

                        schema_info_context += (
                            f"  Column: {column_name}, "
                            f"Data Type: {data_type}\n"
                        )

                    # -------------------------------------
                    # Get sample data
                    # -------------------------------------

                    sample_query = sql.SQL(
                        "SELECT * FROM {}.{} LIMIT 5"
                    ).format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name)
                    )

                    cursor.execute(sample_query)

                    sample_data = cursor.fetchall()

                    schema_info_context += (
                        "  Sample Data:\n"
                    )

                    for row in sample_data:

                        schema_info_context += (
                            f"    {row}\n"
                        )

            return schema_info_context

        except Exception as e:

            print(
                f"Error fetching schema details: {e}"
            )

            return (
                f"Error fetching schema details: {e}"
            )

        finally:

            if connection is not None:
                connection.close()

    # ---------------------------------------------------
    # Execute SQL
    # ---------------------------------------------------

    def execute_sql(self, query):

        connection = None

        try:

            connection = self.get_connection()

            if connection is None:
                return None

            with connection.cursor() as cursor:

                cursor.execute(query)

                result = cursor.fetchall()

            connection.commit()

            return str(result)

        except Exception as e:

            if connection is not None:
                connection.rollback()

            print(
                f"Error executing query: {e}"
            )

            return None

        finally:

            if connection is not None:
                connection.close()