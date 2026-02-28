"""
Retail Analytics Database Manager
Handles all database operations for the retail tracking system
"""

import psycopg2
import psycopg2.extras
import pickle
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RetailDatabaseManager:
    def __init__(self, db_config: Dict = None):
        """
        Initialize database connection with better error handling
        
        Args:
            db_config: Database configuration dictionary
        """
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'retail_analytics'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password'),
                'port': os.getenv('DB_PORT', '5432')
            }
        
        self.db_config = db_config
        self.connection = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """Establish database connection with retry logic and better error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add connection timeout and other safety parameters
                self.connection = psycopg2.connect(
                    **self.db_config,
                    connect_timeout=10,
                    application_name='retail_analytics'
                )
                self.connection.autocommit = True
                self.is_connected = True
                
                # Test the connection with a simple query
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                
                print(f"✅ Database connected successfully (attempt {attempt + 1})")
                return True
            except psycopg2.OperationalError as e:
                print(f"❌ Database connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if "password authentication failed" in str(e).lower():
                    print("🔐 Authentication failed - please check database credentials")
                    break
                elif "connection refused" in str(e).lower():
                    print("🔌 Connection refused - is PostgreSQL server running?")
                elif "database" in str(e).lower() and "does not exist" in str(e).lower():
                    print("🗄️ Database does not exist - please create the database first")
                    break
                    
                if attempt == max_retries - 1:
                    print("⚠️ Continuing without database - some features will be disabled")
                    self.is_connected = False
                    return False
                time.sleep(2)  # Wait before retry
            except Exception as e:
                print(f"❌ Unexpected database error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("⚠️ Continuing without database - some features will be disabled")
                    self.is_connected = False
                    return False
                time.sleep(2)
        
        return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("🔌 Database disconnected")
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """
        Execute a database query with graceful fallback
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, else None
        """
        if not self.is_connected:
            if not self.connect():
                print("⚠️ Database not available - query skipped")
                return None
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            # Try to reconnect
            self.is_connected = False
            return None
    
    # Customer Management
    def get_or_create_customer(self, logical_id: int, encoding_data: np.ndarray = None) -> int:
        """
        Get existing customer or create new one with fallback
        
        Args:
            logical_id: Customer logical ID from tracking system
            encoding_data: Customer encoding data
            
        Returns:
            Customer database ID or -1 if database unavailable
        """
        if not self.is_connected:
            return -1  # Fallback ID
            
        # Try to get existing customer
        query = "SELECT id FROM customers WHERE logical_id = %s"
        result = self.execute_query(query, (logical_id,))
        
        if result and len(result) > 0:
            customer_id = result[0]['id']
            # Update last seen time
            update_query = "UPDATE customers SET last_seen = CURRENT_TIMESTAMP WHERE id = %s"
            self.execute_query(update_query, (customer_id,), fetch=False)
            return customer_id
        
        # Create new customer
        encoding_bytes = pickle.dumps(encoding_data) if encoding_data is not None else None
        insert_query = """
        INSERT INTO customers (logical_id, encoding_data) 
        VALUES (%s, %s) RETURNING id
        """
        result = self.execute_query(insert_query, (logical_id, encoding_bytes))
        return result[0]['id'] if result else -1
    
    # Session Management
    def start_customer_session(self, customer_id: int, entry_point_id: int) -> int:
        """
        Start a new customer session
        
        Args:
            customer_id: Customer database ID
            entry_point_id: Entry point ID
            
        Returns:
            Session ID
        """
        query = """
        INSERT INTO customer_sessions (customer_id, entry_point_id)
        VALUES (%s, %s) RETURNING id
        """
        result = self.execute_query(query, (customer_id, entry_point_id))
        return result[0]['id'] if result else None
    
    def end_customer_session(self, session_id: int):
        """End a customer session"""
        query = """
        UPDATE customer_sessions 
        SET exit_time = CURRENT_TIMESTAMP,
            duration_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entry_time))::INTEGER,
            is_completed = TRUE
        WHERE id = %s
        """
        self.execute_query(query, (session_id,), fetch=False)
    
    # Department Visit Management
    def start_department_visit(self, customer_id: int, session_id: int, department_id: int) -> int:
        """
        Start tracking a department visit
        
        Args:
            customer_id: Customer database ID
            session_id: Session ID
            department_id: Department ID
            
        Returns:
            Department visit ID
        """
        # Check if this is first department for this session
        query = "SELECT COUNT(*) as count FROM department_visits WHERE session_id = %s"
        result = self.execute_query(query, (session_id,))
        is_first = result[0]['count'] == 0 if result else True
        
        # Get visit number for this customer
        visit_query = """
        SELECT COALESCE(MAX(visit_number), 0) + 1 as visit_num 
        FROM department_visits 
        WHERE customer_id = %s
        """
        visit_result = self.execute_query(visit_query, (customer_id,))
        visit_number = visit_result[0]['visit_num'] if visit_result else 1
        
        # Insert department visit
        insert_query = """
        INSERT INTO department_visits 
        (customer_id, session_id, department_id, visit_number, is_first_department)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """
        result = self.execute_query(insert_query, (customer_id, session_id, department_id, visit_number, is_first))
        
        # Update session first department if needed
        if is_first:
            update_session = "UPDATE customer_sessions SET first_department_id = %s WHERE id = %s"
            self.execute_query(update_session, (department_id, session_id), fetch=False)
        
        # Update total departments visited
        update_total = """
        UPDATE customer_sessions 
        SET total_departments_visited = (
            SELECT COUNT(DISTINCT department_id) FROM department_visits WHERE session_id = %s
        )
        WHERE id = %s
        """
        self.execute_query(update_total, (session_id, session_id), fetch=False)
        
        return result[0]['id'] if result else None
    
    def end_department_visit(self, visit_id: int):
        """End a department visit and calculate dwell time"""
        query = """
        UPDATE department_visits 
        SET exit_time = CURRENT_TIMESTAMP,
            dwell_time_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - entry_time))::INTEGER
        WHERE id = %s
        """
        self.execute_query(query, (visit_id,), fetch=False)
    
    # Service Interaction Management
    def record_service_interaction(self, customer_id: int, session_id: int, department_id: int, 
                                 is_attended: bool = False, employee_id: str = None):
        """Record a customer service interaction"""
        query = """
        INSERT INTO customer_service_interactions 
        (customer_id, session_id, department_id, is_attended, employee_id)
        VALUES (%s, %s, %s, %s, %s)
        """
        self.execute_query(query, (customer_id, session_id, department_id, is_attended, employee_id), fetch=False)
    
    def start_wait_time(self, customer_id: int, session_id: int, department_id: int) -> int:
        """Start tracking customer wait time"""
        query = """
        INSERT INTO customer_wait_times (customer_id, session_id, department_id)
        VALUES (%s, %s, %s) RETURNING id
        """
        result = self.execute_query(query, (customer_id, session_id, department_id))
        return result[0]['id'] if result else None
    
    def end_wait_time(self, wait_id: int, is_served: bool = True):
        """End wait time tracking"""
        query = """
        UPDATE customer_wait_times 
        SET wait_end_time = CURRENT_TIMESTAMP,
            wait_duration_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - wait_start_time))::INTEGER,
            is_served = %s
        WHERE id = %s
        """
        self.execute_query(query, (is_served, wait_id), fetch=False)
    
    # Zone Detection
    def get_department_at_position(self, x: int, y: int) -> Optional[Dict]:
        """Get department at given position"""
        query = """
        SELECT * FROM departments 
        WHERE x1 <= %s AND x2 >= %s AND y1 <= %s AND y2 >= %s
        """
        result = self.execute_query(query, (x, x, y, y))
        return dict(result[0]) if result else None
    
    def get_entry_point_at_position(self, x: int, y: int) -> Optional[Dict]:
        """Get entry point at given position"""
        query = """
        SELECT * FROM entry_points 
        WHERE x1 <= %s AND x2 >= %s AND y1 <= %s AND y2 >= %s AND is_entrance = TRUE
        """
        result = self.execute_query(query, (x, x, y, y))
        return dict(result[0]) if result else None
    
    # Analytics Queries
    def get_footfall_stats(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get footfall statistics"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        query = """
        SELECT 
            DATE(entry_time) as date,
            COUNT(*) as footfall_count,
            entry_points.name as entry_point_name
        FROM customer_sessions 
        JOIN entry_points ON customer_sessions.entry_point_id = entry_points.id
        WHERE entry_time BETWEEN %s AND %s
        GROUP BY DATE(entry_time), entry_points.name
        ORDER BY date DESC
        """
        return self.execute_query(query, (start_date, end_date))
    
    def get_department_analytics(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get department analytics"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        query = """
        SELECT 
            d.name as department_name,
            COUNT(DISTINCT dv.customer_id) as unique_customers,
            COUNT(*) as total_visits,
            AVG(dv.dwell_time_seconds) as avg_dwell_time,
            COUNT(CASE WHEN dv.is_first_department THEN 1 END) as first_visits
        FROM departments d
        LEFT JOIN department_visits dv ON d.id = dv.department_id
        LEFT JOIN customer_sessions cs ON dv.session_id = cs.id
        WHERE dv.entry_time BETWEEN %s AND %s OR dv.entry_time IS NULL
        GROUP BY d.name, d.id
        ORDER BY total_visits DESC
        """
        return self.execute_query(query, (start_date, end_date))
    
    def get_service_analytics(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get service analytics"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        query = """
        SELECT 
            d.name as department_name,
            COUNT(*) as total_interactions,
            COUNT(CASE WHEN csi.is_attended THEN 1 END) as attended_count,
            COUNT(CASE WHEN NOT csi.is_attended THEN 1 END) as unattended_count,
            AVG(cwt.wait_duration_seconds) as avg_wait_time
        FROM departments d
        LEFT JOIN customer_service_interactions csi ON d.id = csi.department_id
        LEFT JOIN customer_wait_times cwt ON csi.customer_id = cwt.customer_id AND d.id = cwt.department_id
        WHERE csi.interaction_time BETWEEN %s AND %s OR csi.interaction_time IS NULL
        GROUP BY d.name, d.id
        ORDER BY total_interactions DESC
        """
        return self.execute_query(query, (start_date, end_date))
    
    def get_customer_journey(self, customer_id: int) -> List[Dict]:
        """Get customer journey details"""
        query = """
        SELECT 
            dv.visit_number,
            d.name as department_name,
            dv.entry_time,
            dv.exit_time,
            dv.dwell_time_seconds,
            dv.is_first_department
        FROM department_visits dv
        JOIN departments d ON dv.department_id = d.id
        WHERE dv.customer_id = %s
        ORDER BY dv.visit_number
        """
        return self.execute_query(query, (customer_id,))
    
    # Utility Methods
    def get_all_departments(self) -> List[Dict]:
        """Get all departments"""
        return self.execute_query("SELECT * FROM departments ORDER BY name")
    
    def get_all_entry_points(self) -> List[Dict]:
        """Get all entry points"""
        return self.execute_query("SELECT * FROM entry_points WHERE is_entrance = TRUE ORDER BY name")
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up data older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        queries = [
            "DELETE FROM customer_wait_times WHERE wait_start_time < %s",
            "DELETE FROM customer_service_interactions WHERE interaction_time < %s",
            "DELETE FROM department_visits WHERE entry_time < %s",
            "DELETE FROM customer_sessions WHERE entry_time < %s",
            "DELETE FROM customers WHERE last_seen < %s"
        ]
        
        for query in queries:
            self.execute_query(query, (cutoff_date,), fetch=False)
        
        print(f"🧹 Cleaned up data older than {days} days")
