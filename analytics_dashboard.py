"""
Retail Analytics Dashboard
Web-based dashboard for viewing retail analytics data
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict

from database_manager import RetailDatabaseManager
from store_config import store_config

app = Flask(__name__)

# Set matplotlib to use Agg backend for non-interactive plots
plt.switch_backend('Agg')

class AnalyticsDashboard:
    """Analytics dashboard for retail data visualization"""
    
    def __init__(self):
        self.db_manager = RetailDatabaseManager()
        if not self.db_manager.connect():
            raise Exception("Failed to connect to database")
        
        # Create templates directory
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # Create HTML template
        self.create_html_template()
        
        print("📊 Analytics Dashboard initialized")
    
    def create_html_template(self):
        """Create HTML template for the dashboard"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .date-filter {
            margin: 20px 0;
            text-align: center;
        }
        .date-filter input {
            padding: 8px;
            margin: 0 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .date-filter button {
            padding: 8px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .date-filter button:hover {
            background: #5a6fd8;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .table th, .table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .table th {
            background-color: #667eea;
            color: white;
        }
        .table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            z-index: 1000;
        }
        .refresh-btn:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏪 Retail Analytics Dashboard</h1>
        <p>Real-time customer behavior analytics and insights</p>
    </div>

    <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>

    <div class="date-filter">
        <label>From:</label>
        <input type="date" id="startDate" value="{{ start_date }}">
        <label>To:</label>
        <input type="date" id="endDate" value="{{ end_date }}">
        <button onclick="updateDateRange()">Update</button>
    </div>

    <div class="dashboard">
        <!-- Footfall Metrics -->
        <div class="card">
            <h3>🚪 Footfall Analytics</h3>
            <div class="metric">
                <span>Total Visitors</span>
                <span class="metric-value" id="totalFootfall">-</span>
            </div>
            <div class="metric">
                <span>Active Sessions</span>
                <span class="metric-value" id="activeSessions">-</span>
            </div>
            <div class="metric">
                <span>Avg. Visit Duration</span>
                <span class="metric-value" id="avgDuration">-</span>
            </div>
            <div class="chart-container">
                <canvas id="footfallChart"></canvas>
            </div>
        </div>

        <!-- Department Analytics -->
        <div class="card">
            <h3>🏬 Department Performance</h3>
            <div class="chart-container">
                <canvas id="departmentChart"></canvas>
            </div>
            <table class="table" id="deptTable">
                <thead>
                    <tr>
                        <th>Department</th>
                        <th>Visitors</th>
                        <th>Visits</th>
                        <th>Avg. Dwell</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <!-- Service Analytics -->
        <div class="card">
            <h3>👥 Service Analytics</h3>
            <div class="metric">
                <span>Attended</span>
                <span class="metric-value" id="attendedCount">-</span>
            </div>
            <div class="metric">
                <span>Unattended</span>
                <span class="metric-value" id="unattendedCount">-</span>
            </div>
            <div class="metric">
                <span>Avg. Wait Time</span>
                <span class="metric-value" id="avgWaitTime">-</span>
            </div>
            <div class="chart-container">
                <canvas id="serviceChart"></canvas>
            </div>
        </div>

        <!-- Customer Journey -->
        <div class="card">
            <h3>🗺️ Customer Journey</h3>
            <div class="chart-container">
                <canvas id="journeyChart"></canvas>
            </div>
            <div id="journeyDetails"></div>
        </div>

        <!-- Peak Hours -->
        <div class="card">
            <h3>⏰ Peak Hours Analysis</h3>
            <div class="chart-container">
                <canvas id="peakHoursChart"></canvas>
            </div>
        </div>

        <!-- First Department Analysis -->
        <div class="card">
            <h3>🎯 First Department Analysis</h3>
            <div class="chart-container">
                <canvas id="firstDeptChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let charts = {};

        function initCharts() {
            // Footfall Chart
            const footfallCtx = document.getElementById('footfallChart').getContext('2d');
            charts.footfall = new Chart(footfallCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Daily Footfall',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Department Chart
            const deptCtx = document.getElementById('departmentChart').getContext('2d');
            charts.department = new Chart(deptCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Total Visits',
                        data: [],
                        backgroundColor: '#28a745'
                    }, {
                        label: 'Unique Visitors',
                        data: [],
                        backgroundColor: '#ffc107'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Service Chart
            const serviceCtx = document.getElementById('serviceChart').getContext('2d');
            charts.service = new Chart(serviceCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Attended', 'Unattended'],
                    datasets: [{
                        data: [0, 0],
                        backgroundColor: ['#28a745', '#dc3545']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });

            // Journey Chart
            const journeyCtx = document.getElementById('journeyChart').getContext('2d');
            charts.journey = new Chart(journeyCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Visit Count',
                        data: [],
                        backgroundColor: '#17a2b8'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Peak Hours Chart
            const peakCtx = document.getElementById('peakHoursChart').getContext('2d');
            charts.peakHours = new Chart(peakCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Visitors per Hour',
                        data: [],
                        backgroundColor: '#6f42c1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // First Department Chart
            const firstDeptCtx = document.getElementById('firstDeptChart').getContext('2d');
            charts.firstDept = new Chart(firstDeptCtx, {
                type: 'pie',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        function refreshData() {
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            
            fetch(`/api/analytics?start_date=${startDate}&end_date=${endDate}`)
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                })
                .catch(error => console.error('Error:', error));
        }

        function updateDashboard(data) {
            // Update metrics
            document.getElementById('totalFootfall').textContent = data.footfall.total || 0;
            document.getElementById('activeSessions').textContent = data.footfall.active || 0;
            document.getElementById('avgDuration').textContent = data.footfall.avg_duration || '0m';

            document.getElementById('attendedCount').textContent = data.service.attended || 0;
            document.getElementById('unattendedCount').textContent = data.service.unattended || 0;
            document.getElementById('avgWaitTime').textContent = data.service.avg_wait || '0s';

            // Update footfall chart
            charts.footfall.data.labels = data.footfall.daily.map(d => d.date);
            charts.footfall.data.datasets[0].data = data.footfall.daily.map(d => d.count);
            charts.footfall.update();

            // Update department chart
            charts.department.data.labels = data.departments.map(d => d.name);
            charts.department.data.datasets[0].data = data.departments.map(d => d.total_visits);
            charts.department.data.datasets[1].data = data.departments.map(d => d.unique_customers);
            charts.department.update();

            // Update department table
            const deptTable = document.getElementById('deptTable').getElementsByTagName('tbody')[0];
            deptTable.innerHTML = '';
            data.departments.forEach(dept => {
                const row = deptTable.insertRow();
                row.innerHTML = `
                    <td>${dept.name}</td>
                    <td>${dept.unique_customers}</td>
                    <td>${dept.total_visits}</td>
                    <td>${dept.avg_dwell_time}s</td>
                `;
            });

            // Update service chart
            charts.service.data.datasets[0].data = [data.service.attended, data.service.unattended];
            charts.service.update();

            // Update journey chart
            if (data.journey) {
                charts.journey.data.labels = data.journey.map(j => j.department);
                charts.journey.data.datasets[0].data = data.journey.map(j => j.count);
                charts.journey.update();
            }

            // Update peak hours chart
            if (data.peak_hours) {
                charts.peakHours.data.labels = data.peak_hours.map(h => h.hour);
                charts.peakHours.data.datasets[0].data = data.peak_hours.map(h => h.visitors);
                charts.peakHours.update();
            }

            // Update first department chart
            if (data.first_department) {
                charts.firstDept.data.labels = data.first_department.map(d => d.department);
                charts.firstDept.data.datasets[0].data = data.first_department.map(d => d.count);
                charts.firstDept.update();
            }
        }

        function updateDateRange() {
            refreshData();
        }

        // Initialize on load
        window.onload = function() {
            initCharts();
            refreshData();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        };
    </script>
</body>
</html>
        """
        
        with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
            f.write(template_content)
    
    def get_analytics_data(self, start_date: str = None, end_date: str = None) -> dict:
        """Get comprehensive analytics data with graceful fallback"""
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=7)
        
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        
        # Initialize with default values
        analytics_data = {
            'footfall': {
                'total': 0,
                'active': 0,
                'avg_duration': '0m',
                'daily': []
            },
            'departments': [],
            'service': {
                'attended': 0,
                'unattended': 0,
                'avg_wait': '0s'
            },
            'peak_hours': [],
            'first_department': []
        }
        
        try:
            # Get footfall data
            footfall_data = self.db_manager.get_footfall_stats(start_date, end_date)
            footfall_daily = []
            total_footfall = 0
            
            if footfall_data:
                for record in footfall_data:
                    footfall_daily.append({
                        'date': record['date'].strftime('%Y-%m-%d'),
                        'count': record['footfall_count']
                    })
                    total_footfall += record['footfall_count']
            
            # Get department analytics
            dept_data = self.db_manager.get_department_analytics(start_date, end_date)
            departments = []
            
            if dept_data:
                for record in dept_data:
                    departments.append({
                        'name': record['department_name'],
                        'unique_customers': record['unique_customers'] or 0,
                        'total_visits': record['total_visits'] or 0,
                        'avg_dwell_time': int(record['avg_dwell_time'] or 0)
                    })
            
            # Get service analytics
            service_data = self.db_manager.get_service_analytics(start_date, end_date)
            attended = 0
            unattended = 0
            avg_wait = 0
            
            if service_data:
                for record in service_data:
                    attended += record['attended_count'] or 0
                    unattended += record['unattended_count'] or 0
                    if record['avg_wait_time']:
                        avg_wait += record['avg_wait_time']
            
            # Calculate additional metrics
            active_sessions = self.get_active_sessions_count()
            avg_duration = self.get_avg_session_duration(start_date, end_date)
            
            # Get peak hours data
            peak_hours = self.get_peak_hours_data(start_date, end_date)
            
            # Get first department data
            first_department = self.get_first_department_data(start_date, end_date)
            
            analytics_data = {
                'footfall': {
                    'total': total_footfall,
                    'active': active_sessions,
                    'avg_duration': f"{int(avg_duration // 60)}m",
                    'daily': footfall_daily
                },
                'departments': departments,
                'service': {
                    'attended': attended,
                    'unattended': unattended,
                    'avg_wait': f"{int(avg_wait)}s"
                },
                'peak_hours': peak_hours,
                'first_department': first_department
            }
            
        except Exception as e:
            print(f"⚠️ Error getting analytics data: {e}")
            # Return default values if database is unavailable
            pass
        
        return analytics_data
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions with fallback"""
        if not self.db_manager.is_connected:
            return 0
        
        query = "SELECT COUNT(*) as count FROM customer_sessions WHERE exit_time IS NULL"
        result = self.db_manager.execute_query(query)
        return result[0]['count'] if result else 0
    
    def get_avg_session_duration(self, start_date: datetime, end_date: datetime) -> float:
        """Get average session duration with fallback"""
        if not self.db_manager.is_connected:
            return 0.0
            
        query = """
        SELECT AVG(duration_seconds) as avg_duration 
        FROM customer_sessions 
        WHERE entry_time BETWEEN %s AND %s 
        AND duration_seconds IS NOT NULL
        """
        result = self.db_manager.execute_query(query, (start_date, end_date))
        return result[0]['avg_duration'] if result and result[0]['avg_duration'] else 0
    
    def get_peak_hours_data(self, start_date: datetime, end_date: datetime) -> List[dict]:
        """Get peak hours data with fallback"""
        if not self.db_manager.is_connected:
            return []
            
        query = """
        SELECT 
            EXTRACT(HOUR FROM entry_time) as hour,
            COUNT(*) as visitors
        FROM customer_sessions 
        WHERE entry_time BETWEEN %s AND %s
        GROUP BY EXTRACT(HOUR FROM entry_time)
        ORDER BY hour
        """
        result = self.db_manager.execute_query(query, (start_date, end_date))
        
        peak_hours = []
        if result:
            for record in result:
                peak_hours.append({
                    'hour': f"{int(record['hour']):02d}:00",
                    'visitors': record['visitors']
                })
        
        return peak_hours
    
    def get_first_department_data(self, start_date: datetime, end_date: datetime) -> List[dict]:
        """Get first department visited data with fallback"""
        if not self.db_manager.is_connected:
            return []
            
        query = """
        SELECT 
            d.name as department,
            COUNT(*) as count
        FROM customer_sessions cs
        JOIN departments d ON cs.first_department_id = d.id
        WHERE cs.entry_time BETWEEN %s AND %s
        AND cs.first_department_id IS NOT NULL
        GROUP BY d.name
        ORDER BY count DESC
        """
        result = self.db_manager.execute_query(query, (start_date, end_date))
        
        first_dept = []
        if result:
            for record in result:
                first_dept.append({
                    'department': record['department'],
                    'count': record['count']
                })
        
        return first_dept

# Initialize dashboard
dashboard = AnalyticsDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', 
                         start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                         end_date=datetime.now().strftime('%Y-%m-%d'))

@app.route('/api/analytics')
def get_analytics():
    """API endpoint for analytics data"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    data = dashboard.get_analytics_data(start_date, end_date)
    return jsonify(data)

@app.route('/api/departments')
def get_departments():
    """Get department list"""
    departments = store_config.get_all_departments()
    return jsonify(departments)

@app.route('/api/entry-points')
def get_entry_points():
    """Get entry points list"""
    entry_points = store_config.get_all_entry_points()
    return jsonify(entry_points)

if __name__ == '__main__':
    print("🌐 Starting Analytics Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
