"""
Admin API for VoyagerTrader system.

Provides REST endpoints for monitoring and controlling the trading system.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

from .core import VoyagerTrader, TradingConfig
from .skills import SkillLibrary
from .curriculum import AutomaticCurriculum


# Security configuration
SECRET_KEY = "your-secret-key-here"  # Should be in config/environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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
    return pwd_context.verify(plain_password, hashed_password)


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

    # Mock performance data (replace with actual metrics)
    return {
        "total_trades": 150,
        "winning_trades": 95,
        "losing_trades": 55,
        "win_rate": 63.3,
        "total_return": 12.5,
        "sharpe_ratio": 1.8,
        "max_drawdown": -5.2,
        "daily_returns": [0.2, -0.1, 0.5, 0.3, -0.2, 0.8, 0.1],
        "trade_history": [
            {
                "timestamp": "2024-01-15T10:30:00",
                "symbol": "AAPL",
                "action": "BUY",
                "quantity": 100,
                "price": 185.50,
            },
            {
                "timestamp": "2024-01-15T11:45:00",
                "symbol": "GOOGL",
                "action": "SELL",
                "quantity": 50,
                "price": 142.30,
            },
            {
                "timestamp": "2024-01-15T14:20:00",
                "symbol": "TSLA",
                "action": "BUY",
                "quantity": 75,
                "price": 238.90,
            },
        ],
    }


@app.get("/api/logs")
async def get_system_logs(
    current_user: dict = Depends(get_current_user), limit: int = 100
):
    """Get recent system logs."""
    # Mock log data (replace with actual log reading)
    logs = [
        {
            "timestamp": "2024-01-15T10:30:00",
            "level": "INFO",
            "message": "System started",
        },
        {
            "timestamp": "2024-01-15T10:31:00",
            "level": "INFO",
            "message": "Loading skill library",
        },
        {
            "timestamp": "2024-01-15T10:32:00",
            "level": "INFO",
            "message": "Curriculum initialized",
        },
        {
            "timestamp": "2024-01-15T10:33:00",
            "level": "WARNING",
            "message": "Market volatility detected",
        },
        {
            "timestamp": "2024-01-15T10:34:00",
            "level": "INFO",
            "message": "Executing trade strategy",
        },
    ]

    return {"logs": logs[:limit]}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
