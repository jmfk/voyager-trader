"""
Admin API for VoyagerTrader system.

Provides REST endpoints for monitoring and controlling the trading system.
"""

import logging
import os
import re
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

try:
    from passlib.context import CryptContext

    PASSLIB_AVAILABLE = True
except ImportError:
    try:
        import bcrypt

        PASSLIB_AVAILABLE = False
    except ImportError:
        raise ImportError(
            "Either passlib or bcrypt must be installed for password hashing"
        )

from .core import VoyagerTrader, TradingConfig
from .skills import SkillLibrary
from .curriculum import AutomaticCurriculum


# Security configuration
def get_secret_key() -> str:
    """Get JWT secret key from environment or generate a secure one."""
    secret = os.getenv("VOYAGER_JWT_SECRET")
    if not secret:
        # Generate a secure random secret for development
        secret = secrets.token_urlsafe(32)
        logging.warning(
            "Using auto-generated JWT secret. Set VOYAGER_JWT_SECRET environment variable for production."
        )
    return secret


SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("VOYAGER_JWT_EXPIRE_MINUTES", "30"))

if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None
security = HTTPBearer()

# Simple user database (in production, use proper database)
# Password: admin123
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$yGQFLGs6cdErg/KjO0T1/OLHi0kqy.iLCoQDDAHHwFH2beyAWX/du",
        "role": "admin",
    }
}


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class UserLogin(BaseModel):
    username: str
    password: str


class SystemStatus(BaseModel):
    is_running: bool
    current_task: Optional[str]
    skills_learned: int
    performance: Dict[str, Any]
    uptime: str


class SkillInfo(BaseModel):
    name: str
    description: str
    success_rate: float
    usage_count: int
    created_at: str


class SystemCommand(BaseModel):
    action: str  # start, stop, restart


# FastAPI app
app = FastAPI(
    title="VoyagerTrader Admin API",
    description="Admin interface for the VoyagerTrader autonomous trading system",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trader instance
trader_instance: Optional[VoyagerTrader] = None
start_time = datetime.now()


def get_trader() -> VoyagerTrader:
    """Get or create the trader instance."""
    global trader_instance
    if trader_instance is None:
        trader_instance = VoyagerTrader()
    return trader_instance


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    if PASSLIB_AVAILABLE and pwd_context:
        return pwd_context.verify(plain_password, hashed_password)
    else:
        # Fallback to direct bcrypt verification
        import bcrypt

        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )


def authenticate_user(username: str, password: str):
    """Authenticate user credentials."""
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user


# API Endpoints
@app.post("/api/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Authenticate user and return JWT token."""
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/status", response_model=SystemStatus)
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """Get current system status."""
    trader = get_trader()
    status = trader.get_status()

    uptime = datetime.now() - start_time
    uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"

    return SystemStatus(
        is_running=status["is_running"],
        current_task=status["current_task"],
        skills_learned=status["skills_learned"],
        performance=status["performance"],
        uptime=uptime_str,
    )


@app.post("/api/system/control")
async def control_system(
    command: SystemCommand, current_user: dict = Depends(get_current_user)
):
    """Control the trading system (start/stop/restart)."""
    trader = get_trader()

    if command.action == "start":
        if trader.is_running:
            raise HTTPException(status_code=400, detail="System is already running")
        trader.start()
        return {"message": "System started successfully"}

    elif command.action == "stop":
        if not trader.is_running:
            raise HTTPException(status_code=400, detail="System is already stopped")
        trader.stop()
        return {"message": "System stopped successfully"}

    elif command.action == "restart":
        trader.stop()
        trader.start()
        return {"message": "System restarted successfully"}

    else:
        raise HTTPException(status_code=400, detail="Invalid action")


@app.get("/api/skills", response_model=List[SkillInfo])
async def get_skills(current_user: dict = Depends(get_current_user)):
    """Get list of all skills in the skill library."""
    trader = get_trader()
    skills_data = []

    # Mock skill data (replace with actual skill library data)
    for skill_name, skill_obj in trader.skill_library.skills.items():
        skills_data.append(
            SkillInfo(
                name=skill_name,
                description=getattr(
                    skill_obj, "description", f"Trading skill: {skill_name}"
                ),
                success_rate=getattr(skill_obj, "success_rate", 0.75),
                usage_count=getattr(skill_obj, "usage_count", 0),
                created_at=getattr(skill_obj, "created_at", datetime.now().isoformat()),
            )
        )

    return skills_data


@app.get("/api/performance")
async def get_performance_metrics(current_user: dict = Depends(get_current_user)):
    """Get detailed performance metrics."""
    trader = get_trader()

    try:
        # Get actual performance metrics from the trader
        performance_data = trader.get_performance_metrics()

        # Format the data for the API response
        return {
            "total_trades": performance_data.get("total_trades", 0),
            "winning_trades": performance_data.get("winning_trades", 0),
            "losing_trades": performance_data.get("losing_trades", 0),
            "win_rate": performance_data.get("win_rate", 0.0),
            "total_return": performance_data.get("total_return", 0.0),
            "sharpe_ratio": performance_data.get("sharpe_ratio", 0.0),
            "max_drawdown": performance_data.get("max_drawdown", 0.0),
            "daily_returns": performance_data.get("daily_returns", []),
            "trade_history": performance_data.get("trade_history", []),
        }
    except Exception as e:
        logging.warning(f"Failed to get performance metrics: {e}")
        # Fallback to basic data if trader doesn't have performance metrics yet
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "daily_returns": [],
            "trade_history": [],
            "message": "Performance tracking not yet active - start trading to see metrics",
        }


@app.get("/api/logs")
async def get_system_logs(
    current_user: dict = Depends(get_current_user), limit: int = 100
):
    """Get recent system logs."""
    try:
        # Try to read actual log files
        log_entries = []

        # Read backend logs if available
        backend_log_path = "logs/backend.log"
        if os.path.exists(backend_log_path):
            with open(backend_log_path, "r") as f:
                lines = f.readlines()
                for line in reversed(lines[-limit:]):  # Get most recent entries
                    if line.strip():
                        # Parse log line - basic parsing for uvicorn logs
                        timestamp_match = re.match(
                            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line
                        )
                        if timestamp_match:
                            timestamp = timestamp_match.group(1)
                        else:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Determine log level
                        level = "INFO"
                        if "WARNING" in line or "WARN" in line:
                            level = "WARNING"
                        elif "ERROR" in line:
                            level = "ERROR"
                        elif "DEBUG" in line:
                            level = "DEBUG"

                        log_entries.append(
                            {
                                "timestamp": timestamp,
                                "level": level,
                                "message": line.strip(),
                            }
                        )

        # If no logs found, provide basic status
        if not log_entries:
            log_entries = [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "Admin API server is running",
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "No system logs available yet",
                },
            ]

        return {"logs": log_entries[:limit]}

    except Exception as e:
        logging.error(f"Failed to read log files: {e}")
        # Fallback log entries
        return {
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "WARNING",
                    "message": f"Unable to read log files: {str(e)}",
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "Admin API server is operational",
                },
            ]
        }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
