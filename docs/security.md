# Security Configuration

## JWT Authentication for Admin Interface

The VOYAGER-Trader admin interface uses JWT (JSON Web Tokens) for secure authentication. Proper JWT secret configuration is essential for production deployments.

### Quick Setup

```bash
# Generate and save persistent JWT secret
make jwt-setup

# Or generate a new secret manually
make jwt-generate
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VOYAGER_JWT_SECRET` | Cryptographic secret for JWT signing | Auto-generated | Recommended |
| `VOYAGER_JWT_EXPIRE_MINUTES` | Token expiration time in minutes | 30 | Optional |

### Configuration Options

#### 1. Auto-Generated Secret (Development)
```bash
# No configuration needed - secret auto-generates
# ⚠️ Warning: New secret generated on each restart
```

#### 2. Persistent Secret (Recommended)
```bash
# Generate and save to .env file
python scripts/generate_jwt_secret.py --env

# Or set manually
export VOYAGER_JWT_SECRET="your-secure-secret-here"
export VOYAGER_JWT_EXPIRE_MINUTES="30"
```

#### 3. Production Deployment
```bash
# Use a strong, persistent secret
export VOYAGER_JWT_SECRET="$(openssl rand -base64 32)"
export VOYAGER_JWT_EXPIRE_MINUTES="60"

# Or use the generator
make jwt-setup
```

### Security Best Practices

1. **Use Persistent Secrets**: Set `VOYAGER_JWT_SECRET` to avoid user logouts on restart
2. **Strong Secrets**: Use cryptographically secure random strings (32+ characters)
3. **Appropriate Expiration**: Set `VOYAGER_JWT_EXPIRE_MINUTES` based on your security needs
4. **Environment Variables**: Never hardcode secrets in source code
5. **Secret Rotation**: Periodically generate new JWT secrets

### Troubleshooting

#### Users Getting Logged Out on Restart
```bash
# Problem: Auto-generated secret changes on restart
# Solution: Set persistent secret
make jwt-setup
source .env
```

#### Invalid JWT Errors
```bash
# Check if JWT secret is properly set
echo $VOYAGER_JWT_SECRET

# Regenerate secret if corrupted
make jwt-setup
```

#### Token Expiration Issues
```bash
# Adjust expiration time (in minutes)
export VOYAGER_JWT_EXPIRE_MINUTES="60"  # 1 hour
export VOYAGER_JWT_EXPIRE_MINUTES="480" # 8 hours
```

### Admin Credentials

Default admin credentials:
- **Username**: admin
- **Password**: admin123

⚠️ **Security Warning**: Change default credentials in production deployments.

### API Endpoints

- **Login**: `POST /api/auth/login`
- **Protected Routes**: All `/api/*` endpoints except `/api/health` and `/api/auth/login`
- **Token Format**: Bearer token in Authorization header

### Example Usage

```bash
# Login and get token
curl -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use token for API calls
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  http://localhost:8001/api/status
```