"""
Simple Database Creation Script
Creates database tables step by step
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create database tables step by step"""
    print("🗄️ Creating database tables...")
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'retail_analytics'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("📋 Connected to database")
        
        # Create tables in correct order
        table_creations = [
            # Entry points
            """
            CREATE TABLE IF NOT EXISTS entry_points (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL,
                is_entrance BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Departments
            """
            CREATE TABLE IF NOT EXISTS departments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Customers
            """
            CREATE TABLE IF NOT EXISTS customers (
                id SERIAL PRIMARY KEY,
                logical_id INTEGER UNIQUE NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_visits INTEGER DEFAULT 0,
                encoding_data BYTEA,
                is_active BOOLEAN DEFAULT TRUE
            )
            """,
            
            # Customer sessions
            """
            CREATE TABLE IF NOT EXISTS customer_sessions (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                entry_point_id INTEGER REFERENCES entry_points(id),
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                duration_seconds INTEGER,
                first_department_id INTEGER REFERENCES departments(id),
                total_departments_visited INTEGER DEFAULT 0,
                is_completed BOOLEAN DEFAULT FALSE
            )
            """,
            
            # Department visits
            """
            CREATE TABLE IF NOT EXISTS department_visits (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                session_id INTEGER REFERENCES customer_sessions(id),
                department_id INTEGER REFERENCES departments(id),
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                dwell_time_seconds INTEGER,
                visit_number INTEGER DEFAULT 1,
                is_first_department BOOLEAN DEFAULT FALSE
            )
            """,
            
            # Customer service interactions
            """
            CREATE TABLE IF NOT EXISTS customer_service_interactions (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                session_id INTEGER REFERENCES customer_sessions(id),
                department_id INTEGER REFERENCES departments(id),
                interaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_attended BOOLEAN DEFAULT FALSE,
                employee_id VARCHAR(50),
                interaction_type VARCHAR(50) DEFAULT 'general'
            )
            """,
            
            # Customer wait times
            """
            CREATE TABLE IF NOT EXISTS customer_wait_times (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                session_id INTEGER REFERENCES customer_sessions(id),
                department_id INTEGER REFERENCES departments(id),
                wait_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                wait_end_time TIMESTAMP,
                wait_duration_seconds INTEGER,
                is_served BOOLEAN DEFAULT FALSE
            )
            """
        ]
        
        # Create each table
        for i, create_sql in enumerate(table_creations):
            try:
                cursor.execute(create_sql)
                table_name = create_sql.split("CREATE TABLE IF NOT EXISTS ")[1].split(" ")[0]
                print(f"✅ Created table: {table_name}")
            except Exception as e:
                print(f"❌ Error creating table: {e}")
                return False
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_customers_logical_id ON customers(logical_id)",
            "CREATE INDEX IF NOT EXISTS idx_customers_active ON customers(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_customer_sessions_customer ON customer_sessions(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_customer_sessions_entry_time ON customer_sessions(entry_time)",
            "CREATE INDEX IF NOT EXISTS idx_department_visits_customer ON department_visits(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_department_visits_department ON department_visits(department_id)",
            "CREATE INDEX IF NOT EXISTS idx_department_visits_entry_time ON department_visits(entry_time)",
            "CREATE INDEX IF NOT EXISTS idx_service_interactions_customer ON customer_service_interactions(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_service_interactions_attended ON customer_service_interactions(is_attended)",
            "CREATE INDEX IF NOT EXISTS idx_wait_times_customer ON customer_wait_times(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_wait_times_department ON customer_wait_times(department_id)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"✅ Created index")
            except Exception as e:
                print(f"⚠️ Index warning: {e}")
        
        # Insert sample data
        sample_data = [
            # Entry points
            "INSERT INTO entry_points (name, x1, y1, x2, y2, is_entrance) VALUES ('Main Entrance', 50, 400, 150, 480, TRUE) ON CONFLICT DO NOTHING",
            "INSERT INTO entry_points (name, x1, y1, x2, y2, is_entrance) VALUES ('Side Entrance', 850, 400, 950, 480, TRUE) ON CONFLICT DO NOTHING",
            
            # Departments
            "INSERT INTO departments (name, x1, y1, x2, y2, description) VALUES ('Electronics', 50, 200, 250, 380, 'TVs, phones, computers') ON CONFLICT DO NOTHING",
            "INSERT INTO departments (name, x1, y1, x2, y2, description) VALUES ('Clothing', 300, 200, 500, 380, 'Men and women clothing') ON CONFLICT DO NOTHING",
            "INSERT INTO departments (name, x1, y1, x2, y2, description) VALUES ('Groceries', 550, 200, 750, 380, 'Food and beverages') ON CONFLICT DO NOTHING",
            "INSERT INTO departments (name, x1, y1, x2, y2, description) VALUES ('Furniture', 800, 200, 950, 380, 'Home furniture and decor') ON CONFLICT DO NOTHING",
            "INSERT INTO departments (name, x1, y1, x2, y2, description) VALUES ('Checkout', 400, 400, 600, 480, 'Payment counters') ON CONFLICT DO NOTHING"
        ]
        
        for insert_sql in sample_data:
            try:
                cursor.execute(insert_sql)
                print(f"✅ Inserted sample data")
            except Exception as e:
                print(f"⚠️ Sample data warning: {e}")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"\n📊 Database created with {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        conn.close()
        print("\n🎉 Database creation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False

if __name__ == "__main__":
    if create_database():
        print("\n✅ Ready to run tracking system!")
        print("Run: python track-2.py")
    else:
        print("\n❌ Please check the error messages above")
