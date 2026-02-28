
"""
Retail Analytics Setup Script
Automated setup and configuration for the retail analytics system
"""

import os
import sys
import subprocess
import psycopg2
from getpass import getpass
import json
import sqlparse  # ✅ Added for safe SQL parsing

def print_banner():
    """Print setup banner"""
    print("=" * 70)
    print("🏪 RETAIL ANALYTICS TRACKING SYSTEM - SETUP")
    print("=" * 70)
    print("This script will help you set up the retail analytics system")
    print("including database configuration and dependencies.")
    print("=" * 70)

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_database():
    """Setup PostgreSQL database"""
    print("🗄️ Setting up PostgreSQL database...")

    # Get database credentials
    host = input("Enter database host (default: localhost): ").strip() or "localhost"
    port = input("Enter database port (default: 5432): ").strip() or "5432"
    dbname = input("Enter database name (default: retail_analytics): ").strip() or "retail_analytics"
    user = input("Enter database user (default: postgres): ").strip() or "postgres"
    password = getpass("Enter database password: ")

    # Test connection
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=dbname
        )
        conn.close()
        print("✅ Database connection successful")
    except psycopg2.OperationalError as e:
        if "does not exist" in str(e):
            print(f"📝 Database '{dbname}' does not exist. Creating...")
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database="postgres"
                )
                conn.autocommit = True
                cursor = conn.cursor()
                cursor.execute(f"CREATE DATABASE {dbname}")
                conn.close()
                print(f"✅ Database '{dbname}' created successfully")
            except Exception as create_error:
                print(f"❌ Failed to create database: {create_error}")
                return False
        else:
            print(f"❌ Database connection failed: {e}")
            return False

    # Create schema
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=dbname
        )
        cursor = conn.cursor()

        schema_path = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        statements = sqlparse.split(schema_sql)
        for statement in statements:
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                except psycopg2.Error as e:
                    print(f"❌ Failed executing statement:\n{statement}\nError: {e}")
                    conn.rollback()
                    conn.close()
                    return False

        conn.commit()
        conn.close()
        print("✅ Database schema created successfully")

    except Exception as e:
        print(f"❌ Failed to create database schema: {e}")
        return False

    # Create .env file
    env_content = f"""# Database Configuration
DB_HOST={host}
DB_NAME={dbname}
DB_USER={user}
DB_PASSWORD={password}
DB_PORT={port}

# Video Configuration
VIDEO_PATH=test_video.mp4
OUTPUT_PATH=outputs/retail_analytics_output.mp4
SHOW_DISPLAY=true

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
DASHBOARD_DEBUG=false

# Tracking Configuration
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.5
MAX_HUMAN_SPEED=300
ENCODING_THRESHOLD=0.65
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✅ Environment configuration saved to .env")
    return True

def download_models():
    """Download YOLO models"""
    print("🤖 Downloading YOLO models...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8m.pt')
        print("✅ YOLO model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['outputs', 'templates', 'static']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    return True

def validate_setup():
    """Validate the setup"""
    print("🔍 Validating setup...")

    required_files = [
        'track-2.py',
        'retail_tracker.py',
        'database_manager.py',
        'store_config.py',
        'analytics_dashboard.py',
        'database_schema.sql',
        'bytetrack_retail.yaml',
        '.env'
    ]

    missing_files = [file for file in required_files if not os.path.exists(file)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    try:
        from database_manager import RetailDatabaseManager
        db = RetailDatabaseManager()
        if db.connect():
            print("✅ Database connection test successful")
            db.disconnect()
        else:
            print("❌ Database connection test failed")
            return False
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import psycopg2
        import flask
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    print("✅ Setup validation completed successfully")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("1. Place your video file as 'test_video.mp4' in the project directory")
    print("2. Configure store layout in 'store_config.py' if needed")
    print("3. Run the tracking system:")
    print("   python track-2.py")
    print("4. Start the analytics dashboard:")
    print("   python analytics_dashboard.py")
    print("5. Open http://localhost:5000 in your browser")
    print("\n📚 For more information, see README.md")
    print("=" * 70)

def main():
    """Main setup function"""
    print_banner()

    if not check_python_version():
        sys.exit(1)

    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)

    if not create_directories():
        print("❌ Setup failed during directory creation")
        sys.exit(1)

    if not setup_database():
        print("❌ Setup failed during database configuration")
        sys.exit(1)

    if not download_models():
        print("❌ Setup failed during model download")
        sys.exit(1)

    if not validate_setup():
        print("❌ Setup validation failed")
        sys.exit(1)

    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)