-- Retail Analytics Database Schema
-- PostgreSQL schema for customer tracking and analytics

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS customer_wait_times CASCADE;
DROP TABLE IF EXISTS customer_service_interactions CASCADE;
DROP TABLE IF EXISTS department_visits CASCADE;
DROP TABLE IF EXISTS customer_sessions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS departments CASCADE;
DROP TABLE IF EXISTS entry_points CASCADE;

-- Entry points (store entrances/exits)
CREATE TABLE entry_points (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    is_entrance BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Departments (store sections)
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customers (unique individuals)
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    logical_id INTEGER UNIQUE NOT NULL,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_visits INTEGER DEFAULT 0,
    encoding_data BYTEA,
    is_active BOOLEAN DEFAULT TRUE
);

-- Customer sessions (individual store visits)
CREATE TABLE customer_sessions (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    entry_point_id INTEGER REFERENCES entry_points(id),
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP,
    duration_seconds INTEGER,
    first_department_id INTEGER REFERENCES departments(id),
    total_departments_visited INTEGER DEFAULT 0,
    is_completed BOOLEAN DEFAULT FALSE
);

-- Department visits (each time a customer enters a department)
CREATE TABLE department_visits (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    session_id INTEGER REFERENCES customer_sessions(id),
    department_id INTEGER REFERENCES departments(id),
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP,
    dwell_time_seconds INTEGER,
    visit_number INTEGER DEFAULT 1, -- Order of visits for this customer
    is_first_department BOOLEAN DEFAULT FALSE
);

-- Customer service interactions
CREATE TABLE customer_service_interactions (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    session_id INTEGER REFERENCES customer_sessions(id),
    department_id INTEGER REFERENCES departments(id),
    interaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_attended BOOLEAN DEFAULT FALSE,
    employee_id VARCHAR(50), -- Can be null if unattended
    interaction_type VARCHAR(50) DEFAULT 'general' -- 'general', 'purchase', 'inquiry', etc.
);

-- Customer wait times
CREATE TABLE customer_wait_times (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    session_id INTEGER REFERENCES customer_sessions(id),
    department_id INTEGER REFERENCES departments(id),
    wait_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    wait_end_time TIMESTAMP,
    wait_duration_seconds INTEGER,
    is_served BOOLEAN DEFAULT FALSE
);

-- Indexes for performance
CREATE INDEX idx_customers_logical_id ON customers(logical_id);
CREATE INDEX idx_customers_active ON customers(is_active);
CREATE INDEX idx_customer_sessions_customer ON customer_sessions(customer_id);
CREATE INDEX idx_customer_sessions_entry_time ON customer_sessions(entry_time);
CREATE INDEX idx_department_visits_customer ON department_visits(customer_id);
CREATE INDEX idx_department_visits_department ON department_visits(department_id);
CREATE INDEX idx_department_visits_entry_time ON department_visits(entry_time);
CREATE INDEX idx_service_interactions_customer ON customer_service_interactions(customer_id);
CREATE INDEX idx_service_interactions_attended ON customer_service_interactions(is_attended);
CREATE INDEX idx_wait_times_customer ON customer_wait_times(customer_id);
CREATE INDEX idx_wait_times_department ON customer_wait_times(department_id);

-- Insert sample data
INSERT INTO entry_points (name, x1, y1, x2, y2, is_entrance) VALUES
('Main Entrance', 50, 400, 150, 480, TRUE),
('Side Entrance', 850, 400, 950, 480, TRUE),
('Emergency Exit', 450, 50, 550, 130, FALSE);

INSERT INTO departments (name, x1, y1, x2, y2, description) VALUES
('Electronics', 50, 200, 250, 380, 'TVs, phones, computers'),
('Clothing', 300, 200, 500, 380, 'Men and women clothing'),
('Groceries', 550, 200, 750, 380, 'Food and beverages'),
('Furniture', 800, 200, 950, 380, 'Home furniture and decor'),
('Checkout', 400, 400, 600, 480, 'Payment counters');

-- Views for common analytics queries
CREATE VIEW customer_footfall AS
SELECT 
    DATE(entry_time) as date,
    COUNT(*) as footfall_count,
    entry_points.name as entry_point_name
FROM customer_sessions 
JOIN entry_points ON customer_sessions.entry_point_id = entry_points.id
WHERE entry_time >= CURRENT_DATE
GROUP BY DATE(entry_time), entry_points.name;

CREATE VIEW department_analytics AS
SELECT 
    d.name as department_name,
    COUNT(DISTINCT dv.customer_id) as unique_customers,
    COUNT(*) as total_visits,
    AVG(dv.dwell_time_seconds) as avg_dwell_time,
    COUNT(CASE WHEN dv.is_first_department THEN 1 END) as first_visits
FROM departments d
LEFT JOIN department_visits dv ON d.id = dv.department_id
GROUP BY d.name, d.id;

CREATE VIEW service_analytics AS
SELECT 
    d.name as department_name,
    COUNT(*) as total_interactions,
    COUNT(CASE WHEN csi.is_attended THEN 1 END) as attended_count,
    COUNT(CASE WHEN NOT csi.is_attended THEN 1 END) as unattended_count,
    AVG(cwt.wait_duration_seconds) as avg_wait_time
FROM departments d
LEFT JOIN customer_service_interactions csi ON d.id = csi.department_id
LEFT JOIN customer_wait_times cwt ON csi.customer_id = cwt.customer_id AND d.id = cwt.department_id
GROUP BY d.name, d.id;
