# VoyagerTrader Admin Interface

A React-based admin interface for monitoring and controlling the VoyagerTrader autonomous trading system.

## Features

- **System Status Monitoring**: Real-time status of the trading system
- **Performance Analytics**: Charts and metrics showing trading performance
- **Skill Library Management**: View and manage learned trading skills
- **System Logs**: Monitor system logs with filtering and auto-refresh
- **System Control**: Start, stop, and restart the trading system
- **Secure Authentication**: JWT-based authentication for admin access

## Quick Start

### Prerequisites

- Node.js (v14 or higher)
- Python 3.12+
- VoyagerTrader backend dependencies

### Installation

1. Install React dependencies:
```bash
cd admin-ui
npm install
```

2. Install Python backend dependencies:
```bash
cd ..
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Use the startup script (Recommended)
```bash
python start_admin.py
```
This will start the backend API server. Then in a new terminal:
```bash
cd admin-ui
npm start
```

#### Option 2: Manual startup
1. Start the backend API server:
```bash
python -m uvicorn src.voyager_trader.admin_api:app --reload --host 0.0.0.0 --port 8001
```

2. In a new terminal, start the React development server:
```bash
cd admin-ui
npm start
```

### Access the Interface

- **Admin Interface**: http://localhost:3001
- **API Documentation**: http://localhost:8001/docs
- **API Health Check**: http://localhost:8001/api/health

### Admin Credentials

- **Username**: admin
- **Password**: Configurable via environment variables
  - Set `VOYAGER_ADMIN_PASSWORD` for custom password
  - Default: `admin123` (⚠️ change for production!)

**Security Note**: Always change default credentials in production environments.

## API Endpoints

The backend provides the following REST API endpoints:

- `POST /api/auth/login` - User authentication
- `GET /api/status` - System status
- `POST /api/system/control` - Control system (start/stop/restart)
- `GET /api/skills` - Skill library data
- `GET /api/performance` - Performance metrics
- `GET /api/logs` - System logs
- `GET /api/health` - Health check

## Development

### Project Structure

```
admin-ui/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Dashboard.js       # Main dashboard
│   │   ├── Login.js          # Authentication
│   │   ├── SystemStatus.js   # System overview
│   │   ├── PerformanceChart.js # Performance analytics
│   │   ├── SkillsTable.js    # Skills management
│   │   ├── LogsPanel.js      # System logs
│   │   └── ProtectedRoute.js # Route protection
│   ├── services/
│   │   └── api.js            # API service
│   ├── App.js                # Main app component
│   └── index.js              # App entry point
├── package.json
└── README.md
```

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Customization

1. **Styling**: The app uses Tailwind CSS for styling. Customize colors and themes in `tailwind.config.js`.

2. **API Configuration**: Update API endpoints in `src/services/api.js`.

3. **Authentication**: Modify authentication logic in the backend `admin_api.py`.

4. **Components**: Add new dashboard components by creating files in `src/components/`.

## Security Considerations

- Change default credentials in production
- Use environment variables for sensitive configuration
- Implement proper HTTPS in production
- Configure CORS properly for your domain
- Use secure JWT secret keys

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure the backend CORS configuration allows your frontend domain.

2. **Authentication Issues**: Check that JWT tokens are being stored and sent correctly.

3. **Connection Refused**: Ensure the backend server is running on port 8001.

4. **Missing Dependencies**: Run `npm install` and `pip install -r requirements.txt`.

### Development Mode

For development, the React app proxies API requests to `http://localhost:8001`. This is configured in `package.json`:

```json
"proxy": "http://localhost:8001"
```

## Production Deployment

1. Build the React app:
```bash
npm run build
```

2. Serve the built files using a web server (nginx, Apache, etc.)

3. Configure the backend to serve the React build files for a single-page application.

4. Set up proper environment variables and configuration files.

## License

This project is part of the VoyagerTrader system and follows the same license terms.