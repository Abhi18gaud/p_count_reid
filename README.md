# 🏪 Retail Analytics Tracking System

A comprehensive computer vision-based retail analytics system that tracks customer behavior, department visits, and service interactions in real-time with advanced employee detection and management.

## 📊 Features

### Core Analytics
- ✅ **Footfall Entry Counting**: Track number of customers entering the store
- ✅ **Department Dwell Time**: Measure time customers spend in each department
- ✅ **Department Visit Counting**: Count total visits per department
- ✅ **Unique Customer Counting**: Track unique customers per department
- ✅ **First Department Analysis**: Identify first departments visited by customers
- ✅ **Service Interaction Tracking**: Monitor attended vs unattended customers
- ✅ **Wait Time Analysis**: Track customer wait times for service

### 🚀 Employee Detection & Management System
- 🧠 **OSNet Face Recognition**: Advanced employee face detection using `osnet_x1_0_msmt17.pth`
- 📸 **Image Upload Registration**: Register employees from photos or webcam
- 💾 **12-Hour Body Feature Storage**: Capture and store body features after face detection
- 🚫 **No Repeated Face Scans**: Use body features for identification within 12 hours
- 👥 **Employee Name Display**: Show employee names instead of generic IDs
- 🎯 **Employee Exclusion**: Automatically exclude employees from customer analytics
- ⏰ **Active Shift Management**: Track employee shifts with automatic expiration
- 🔄 **Real-time Employee Updates**: Add/remove employees dynamically

### Technical Features
- 🎥 Real-time video processing with YOLOv8
- 🧠 Advanced person re-identification with encoding similarity
- 🗄️ PostgreSQL database for data persistence
- 🌐 Web-based analytics dashboard
- 📊 Real-time visualization and reporting
- 🔧 Configurable store layout and zones
- 🤖 OpenCV 4.13.0 compatible image processing
- 🚀 GPU acceleration with CUDA support

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- CUDA-compatible GPU (optional but recommended)

### Setup Steps

1. **Clone and navigate to project**
   ```bash
   cd p_count
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup PostgreSQL database**
   ```sql
   CREATE DATABASE retail_analytics;
   CREATE USER postgres WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE retail_analytics TO postgres;
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Download YOLO model** (will be downloaded automatically)
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
   ```

## 🚀 Usage

### Employee Management System

1. **Register New Employee**
   ```bash
   python register_employee.py
   ```
   
   Options available:
   - Register from webcam capture
   - Register from image upload (JPG/PNG)
   - List all registered employees
   - Remove employees
   - View active shifts

2. **Employee Detection Protocol**
   - **First Detection**: OSNet face recognition identifies employee
   - **Body Feature Capture**: System captures and stores body features for 12 hours
   - **Subsequent Detection**: Body features used for identification (no face scan needed)
   - **Display**: Employee names shown instead of generic IDs
   - **Exclusion**: Employees automatically excluded from customer analytics

### Running the Tracking System

1. **Basic tracking with video**
   ```bash
   python track-2.py
   ```

2. **Custom video file**
   ```bash
   python track-2.py --video your_video.mp4
   ```

3. **Headless mode (no display)**
   ```bash
   python track-2.py --no-display
   ```

### Starting the Analytics Dashboard

1. **Run dashboard separately**
   ```bash
   python analytics_dashboard.py
   ```

2. **Access dashboard**
   Open http://localhost:5000 in your browser

### Database Operations

1. **Setup database schema**
   ```bash
   psql -d retail_analytics -f database_schema.sql
   ```

2. **View analytics data**
   ```python
   from database_manager import RetailDatabaseManager
   
   db = RetailDatabaseManager()
   db.connect()
   
   # Get footfall stats
   footfall = db.get_footfall_stats()
   
   # Get department analytics
   departments = db.get_department_analytics()
   
   db.disconnect()
   ```

## 📁 Project Structure

```
p_count/
├── track-2.py                 # Main tracking script
├── retail_tracker.py          # Enhanced tracking engine
├── database_manager.py        # Database operations
├── store_config.py            # Store layout configuration
├── analytics_dashboard.py     # Web dashboard
├── employee_exclusion.py      # Employee detection & management system
├── register_employee.py       # Employee registration utility
├── professional_reid.py       # OSNet Re-ID model handler
├── database_schema.sql        # PostgreSQL schema
├── bytetrack_retail.yaml      # ByteTrack configuration
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
├── outputs/                  # Video output directory
├── templates/                # HTML templates for dashboard
└── static/                   # Static assets for dashboard
```

## ⚙️ Configuration

### Employee Detection Configuration

Employee detection settings in `employee_exclusion.py`:

```python
# Face recognition threshold
self.face_similarity_threshold = 0.6

# Body feature matching threshold  
self.body_similarity_threshold = 0.5

# Shift duration (12 hours)
self.shift_duration = 12 * 60 * 60  # seconds

# OSNet model path
self.osnet_model_path = "osnet_x1_0_msmt17.pth"
```

### Store Layout Configuration

Edit `store_config.py` to define your store layout:

```python
self.departments = {
    'electronics': {
        'name': 'Electronics',
        'x1': 50, 'y1': 200, 'x2': 250, 'y2': 380,
        'color': (255, 0, 0),
        'has_service': True,
        'service_desk': {'x': 150, 'y': 290}
    },
    # Add more departments...
}
```

### Tracking Parameters

Adjust tracking parameters in `store_config.py`:

```python
self.tracking_params = {
    'min_dwell_time': 5,        # seconds to count as visit
    'session_timeout': 600,     # session timeout in seconds
    'max_idle_time': 120,       # idle time before timeout
    'reentry_cooldown': 30      # cooldown before re-entry
}
```

### Database Configuration

Update `.env` file with your database settings:

```env
DB_HOST=localhost
DB_NAME=retail_analytics
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432
```

## 📊 Analytics Dashboard Features

The web dashboard provides:

- **Real-time Metrics**: Footfall, active sessions, service interactions
- **Department Performance**: Visit counts, dwell times, unique visitors
- **Service Analytics**: Attended vs unattended ratios, wait times
- **Customer Journey**: Department visit patterns
- **Peak Hours Analysis**: Busiest times of day
- **First Department Analysis**: Customer entry point preferences

### Dashboard API Endpoints

- `GET /` - Main dashboard
- `GET /api/analytics` - Analytics data with date filtering
- `GET /api/departments` - Department configuration
- `GET /api/entry-points` - Entry point configuration

## 🔧 Advanced Usage

### Custom Department Zones

1. **Define new departments** in `store_config.py`
2. **Set coordinates** based on your video dimensions
3. **Configure service desks** for departments with staff
4. **Run tracking** to collect data

### Export Analytics Data

```python
from database_manager import RetailDatabaseManager
import pandas as pd

db = RetailDatabaseManager()
db.connect()

# Export to CSV
footfall_data = db.get_footfall_stats()
df = pd.DataFrame(footfall_data)
df.to_csv('footfall_report.csv', index=False)

db.disconnect()
```

### Real-time Monitoring

For production deployment:

1. **Use a video stream** instead of file
2. **Set up database replication** for scalability
3. **Configure dashboard** for remote access
4. **Set up monitoring** for system health

## 🐛 Troubleshooting

### Common Issues

1. **Database connection failed**
   - Check PostgreSQL is running
   - Verify credentials in `.env`
   - Ensure database exists

2. **Video not found**
   - Check video file path
   - Ensure video format is supported
   - Verify file permissions

3. **GPU not detected**
   - Install CUDA toolkit
   - Update PyTorch with CUDA support
   - Check GPU compatibility

4. **Dashboard not loading**
   - Check Flask installation
   - Verify port availability
   - Check firewall settings

### Employee Detection Issues

1. **OSNet model not found**
   ```bash
   # Download OSNet model
   wget https://github.com/KaiyangZhou/pytorch-reid/raw/master/reid-models/osnet_x1_0_msmt17.pth
   ```

2. **Employee registration failed**
   - Check image quality and lighting
   - Ensure face is clearly visible
   - Try different image format (JPG/PNG)

3. **Body feature extraction errors**
   - Verify OpenCV 4.13.0 compatibility
   - Check image preprocessing pipeline
   - Ensure proper histogram calculation syntax

4. **Employee not detected**
   - Adjust face similarity threshold (0.5-0.7)
   - Check body feature matching threshold (0.4-0.6)
   - Verify employee shift is still active (12-hour window)

5. **False positive detections**
   - Increase similarity thresholds
   - Add more training data for employees
   - Check lighting and camera angles

### Performance Optimization

1. **GPU Acceleration**: Use CUDA-compatible GPU
2. **Video Resolution**: Lower resolution for faster processing
3. **Database Indexing**: Ensure proper indexes on large tables
4. **Batch Processing**: Process multiple frames in batches

## 📈 Analytics Metrics Explained

### Footfall Metrics
- **Total Visitors**: Count of unique customers entering
- **Active Sessions**: Currently ongoing customer visits
- **Average Duration**: Mean time spent in store

### Department Metrics
- **Unique Customers**: Distinct visitors per department
- **Total Visits**: All department entries
- **Average Dwell Time**: Mean time spent per visit
- **First Visits**: Customers who visited department first

### Service Metrics
- **Attended Count**: Customers who received staff assistance
- **Unattended Count**: Customers who left without assistance
- **Average Wait Time**: Mean time waiting for service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:

1. Check the troubleshooting section
2. Review the configuration documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Built with ❤️ using Computer Vision, PostgreSQL, and Flask**
