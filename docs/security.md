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

Admin credentials can be configured via environment variables:
- **Username**: admin (default)
- **Password**: Set via `VOYAGER_ADMIN_PASSWORD` environment variable
- **Password Hash**: Set via `VOYAGER_ADMIN_PASSWORD_HASH` environment variable (takes precedence)

```bash
# Set custom admin password
export VOYAGER_ADMIN_PASSWORD="your-secure-password"

# Or set pre-hashed password (recommended for production)
export VOYAGER_ADMIN_PASSWORD_HASH="$2b$12$your-bcrypt-hash-here"
```

⚠️ **Security Warning**: Always change default credentials for production deployments.

## CORS Configuration

Cross-Origin Resource Sharing (CORS) controls which domains can access the admin API. Proper CORS configuration is essential for production security.

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VOYAGER_CORS_ORIGINS` | Comma-separated list of allowed origins | `http://localhost:3001,http://127.0.0.1:3001` | Production |

### Configuration Examples

#### 1. Development (Default)
```bash
# No configuration needed - uses localhost defaults
# Default: http://localhost:3001,http://127.0.0.1:3001
```

#### 2. Production Deployment
```bash
# Single domain
export VOYAGER_CORS_ORIGINS="https://admin.mycompany.com"

# Multiple domains
export VOYAGER_CORS_ORIGINS="https://admin.mycompany.com,https://trader.mycompany.com"

# Include staging and production
export VOYAGER_CORS_ORIGINS="https://admin-staging.mycompany.com,https://admin.mycompany.com"
```

#### 3. Development + Production
```bash
# Allow both local development and production
export VOYAGER_CORS_ORIGINS="http://localhost:3001,https://admin.mycompany.com"
```

### Security Best Practices

1. **Specific Origins**: Never use wildcard (`*`) in production
2. **HTTPS Only**: Use HTTPS origins for production environments
3. **Minimal Origins**: Only include necessary domains
4. **Regular Review**: Periodically audit allowed origins
5. **Environment Separation**: Different CORS settings for dev/staging/prod

### Common CORS Issues

#### Issue: Admin Interface Can't Connect
```bash
# Problem: CORS origin not allowed
# Solution: Add your domain to CORS origins
export VOYAGER_CORS_ORIGINS="https://your-admin-domain.com"
```

#### Issue: Multiple Domains Need Access
```bash
# Solution: Comma-separated list
export VOYAGER_CORS_ORIGINS="https://domain1.com,https://domain2.com,https://domain3.com"
```

## Rate Limiting

Rate limiting prevents API abuse by restricting the number of requests from each client. This is essential for preventing brute force attacks and maintaining system performance.

### Environment Variables

| Variable | Description | Default | Purpose |
|----------|-------------|---------|---------|
| `VOYAGER_RATE_LIMIT_LOGIN` | Login attempts per time period | `5/minute` | Prevent brute force attacks |
| `VOYAGER_RATE_LIMIT_API` | API requests per time period | `100/minute` | General API protection |
| `VOYAGER_RATE_LIMIT_HEALTH` | Health check requests per time period | `60/minute` | Monitoring endpoint protection |
| `VOYAGER_RATE_LIMIT_STORAGE` | Storage backend for rate limiting | `memory://` | Rate limit data storage |

### Rate Limit Formats

Rate limits use the format `number/period` where period can be:
- `second` - Per second
- `minute` - Per minute  
- `hour` - Per hour
- `day` - Per day

Examples: `5/minute`, `100/hour`, `1000/day`

### Configuration Examples

#### 1. Development (Default)
```bash
# No configuration needed - uses safe defaults
# LOGIN: 5/minute, API: 100/minute, HEALTH: 60/minute
```

#### 2. Production (Stricter)
```bash
# Stricter limits for production
export VOYAGER_RATE_LIMIT_LOGIN="3/minute"
export VOYAGER_RATE_LIMIT_API="60/minute" 
export VOYAGER_RATE_LIMIT_HEALTH="30/minute"
```

#### 3. High Traffic (Relaxed)
```bash
# Higher limits for high-traffic environments
export VOYAGER_RATE_LIMIT_LOGIN="10/minute"
export VOYAGER_RATE_LIMIT_API="500/minute"
export VOYAGER_RATE_LIMIT_HEALTH="120/minute"
```

#### 4. Redis Storage (Production)
```bash
# Use Redis for distributed rate limiting
export VOYAGER_RATE_LIMIT_STORAGE="redis://localhost:6379/0"

# Or Redis with authentication
export VOYAGER_RATE_LIMIT_STORAGE="redis://:password@localhost:6379/0"
```

### Storage Backends

#### Memory (Development)
- **URI**: `memory://`
- **Pros**: Simple, no external dependencies
- **Cons**: Not shared across processes/instances
- **Use case**: Development, single-instance deployments

#### Redis (Production)
- **URI**: `redis://host:port/db` or `redis://:password@host:port/db`
- **Pros**: Distributed, persistent, scalable
- **Cons**: Requires Redis server
- **Use case**: Production, multi-instance deployments

### Security Benefits

1. **Brute Force Protection**: Login rate limiting prevents password attacks
2. **DoS Prevention**: API rate limiting prevents resource exhaustion
3. **Fair Usage**: Ensures equal access for all legitimate users
4. **Cost Control**: Prevents excessive API usage and associated costs

### Testing Rate Limits

```bash
# Check current configuration
make rate-limit-check

# Test rate limiting functionality
make rate-limit-test

# Manual testing with curl
for i in {1..10}; do curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8001/api/health; done
```

### Common Rate Limit Issues

#### Issue: 429 Too Many Requests
```bash
# Problem: Client hitting rate limits
# Solution: Implement exponential backoff in client
# Or increase limits if legitimate traffic

export VOYAGER_RATE_LIMIT_API="200/minute"
```

#### Issue: Rate Limits Not Working
```bash
# Check that slowapi is installed
pip install slowapi

# Check rate limit configuration
make rate-limit-check

# Test rate limiting
make rate-limit-test
```

### Rate Limit Headers

When rate limiting is active, responses include headers:
- `X-RateLimit-Limit`: Total requests allowed
- `X-RateLimit-Remaining`: Requests remaining in current period
- `Retry-After`: Seconds to wait before retrying (when limited)

### API Endpoints

- **Login**: `POST /api/auth/login`
- **Protected Routes**: All `/api/*` endpoints except `/api/health` and `/api/auth/login`
- **Token Format**: Bearer token in Authorization header

### Example Usage

```bash
# Login and get token
curl -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# Use token for API calls
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  http://localhost:8001/api/status
```