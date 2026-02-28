"""
Database Reset Script
Completely drops and recreates all database tables
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reset_database():
    """Reset database completely"""
    print("🗄️ Resetting database...")
    
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
        
        # Drop all tables in correct order (respecting foreign keys)
        tables_to_drop = [
            'customer_wait_times',
            'customer_service_interactions', 
            'department_visits',
            'customer_sessions',
            'customers',
            'departments',
            'entry_points'
        ]
        
        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"✅ Dropped table: {table}")
            except Exception as e:
                print(f"⚠️ Could not drop {table}: {e}")
        
        # Read and execute schema
        if os.path.exists('database_schema.sql'):
            with open('database_schema.sql', 'r') as f:
                schema_sql = f.read()
            
            print("📝 Creating new schema...")
            
            # Execute schema statements
            statements = schema_sql.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        print(f"❌ Error executing: {statement[:50]}...")
                        print(f"   Error: {e}")
                        return False
            
            print("✅ Database schema created successfully")
            
            # Verify tables were created
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            print(f"\n📊 Created {len(tables)} tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
        else:
            print("❌ database_schema.sql not found")
            return False
        
        conn.close()
        print("\n🎉 Database reset completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False

if __name__ == "__main__":
    if reset_database():
        print("\n✅ Ready to run tracking system!")
    else:
        print("\n❌ Please check the error messages above")
