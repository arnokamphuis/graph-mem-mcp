# Deployment Guide

Complete guide for deploying the Graph Memory MCP Server using Docker/Podman containers.

## Prerequisites

- Docker or Podman installed
- Basic understanding of container operations
- Port 8000 available on your system

## Quick Deployment

### Using Podman

```bash
# Build the container
podman build -t graph-mcp-server ./mcp_server

# Run with persistent storage
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  graph-mcp-server
```

### Using Docker

```bash
# Build the container
docker build -t graph-mcp-server ./mcp_server

# Run with persistent storage
docker run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  graph-mcp-server
```

## Container Configuration

### Dockerfile Details

The included Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory for persistence
RUN mkdir -p /data

# Copy application
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Required Dependencies

Create `requirements.txt` in the `mcp_server` directory:

```text
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
```

## Persistent Storage

### Volume Configuration

The container uses `/data` directory for persistence:

- **Volume Mount**: `-v graph-mcp-memory:/data`
- **Storage File**: `/data/memory_banks.json`
- **Automatic Backup**: Data saved after each mutation

### Data Location

```bash
# Check volume location
podman volume inspect graph-mcp-memory
# or
docker volume inspect graph-mcp-memory

# Manual backup
podman volume export graph-mcp-memory > backup.tar
# or
docker run --rm -v graph-mcp-memory:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /data .
```

## Container Management

### Basic Operations

```bash
# Start container
podman start graph-mcp-server

# Stop container
podman stop graph-mcp-server

# Restart container
podman restart graph-mcp-server

# Remove container (keeps data)
podman rm graph-mcp-server

# Remove container and volume (DESTROYS DATA)
podman rm graph-mcp-server
podman volume rm graph-mcp-memory
```

### Monitoring

```bash
# View logs
podman logs graph-mcp-server

# Follow logs in real-time
podman logs -f graph-mcp-server

# Check container status
podman ps

# Container resource usage
podman stats graph-mcp-server
```

## Production Deployment

### Security Considerations

1. **Network Security**
```bash
# Bind to localhost only
podman run -d \
  --name graph-mcp-server \
  -p 127.0.0.1:8000:8000 \
  -v graph-mcp-memory:/data \
  graph-mcp-server
```

2. **User Permissions**
```bash
# Run as non-root user
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  --user 1000:1000 \
  graph-mcp-server
```

3. **Resource Limits**
```bash
# Set memory and CPU limits
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  --memory 512m \
  --cpus 1.0 \
  graph-mcp-server
```

### High Availability

1. **Auto-restart Policy**
```bash
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  --restart always \
  graph-mcp-server
```

2. **Health Checks**
```bash
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  --health-cmd "curl -f http://localhost:8000/ || exit 1" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  graph-mcp-server
```

### Environment Configuration

```bash
# Custom configuration via environment variables
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  -e LOG_LEVEL=INFO \
  -e MAX_MEMORY_SIZE=1GB \
  graph-mcp-server
```

## Load Balancing

### Multiple Instances

For high traffic, run multiple instances:

```bash
# Instance 1
podman run -d \
  --name graph-mcp-server-1 \
  -p 8001:8000 \
  -v graph-mcp-memory-1:/data \
  graph-mcp-server

# Instance 2
podman run -d \
  --name graph-mcp-server-2 \
  -p 8002:8000 \
  -v graph-mcp-memory-2:/data \
  graph-mcp-server
```

### Nginx Configuration

Example Nginx load balancer configuration:

```nginx
upstream graph_mcp_servers {
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 8000;
    
    location / {
        proxy_pass http://graph_mcp_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Backup and Recovery

### Automated Backup

Create a backup script:

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
VOLUME_NAME="graph-mcp-memory"

# Create backup
podman volume export $VOLUME_NAME > "$BACKUP_DIR/mcp_backup_$DATE.tar"

# Keep only last 7 backups
find $BACKUP_DIR -name "mcp_backup_*.tar" -mtime +7 -delete

echo "Backup completed: mcp_backup_$DATE.tar"
```

### Restore from Backup

```bash
# Stop container
podman stop graph-mcp-server

# Remove existing volume
podman volume rm graph-mcp-memory

# Restore from backup
podman volume import graph-mcp-memory < backup.tar

# Start container
podman start graph-mcp-server
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Check what's using port 8000
netstat -tlnp | grep :8000
# or
lsof -i :8000

# Use different port
podman run -d --name graph-mcp-server -p 8080:8000 -v graph-mcp-memory:/data graph-mcp-server
```

2. **Permission Denied on Volume**
```bash
# Fix volume permissions
podman volume create graph-mcp-memory
podman run --rm -v graph-mcp-memory:/data alpine chmod 777 /data
```

3. **Container Won't Start**
```bash
# Check logs
podman logs graph-mcp-server

# Run interactively for debugging
podman run -it --rm -p 8000:8000 -v graph-mcp-memory:/data graph-mcp-server bash
```

4. **Memory Issues**
```bash
# Check container memory usage
podman stats graph-mcp-server

# Increase memory limit
podman run -d --name graph-mcp-server -p 8000:8000 -v graph-mcp-memory:/data --memory 1g graph-mcp-server
```

### Performance Tuning

1. **Optimize for CPU**
```bash
# Set CPU affinity
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  --cpuset-cpus "0,1" \
  graph-mcp-server
```

2. **Optimize for Memory**
```bash
# Use memory limits and swap
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  --memory 512m \
  --memory-swap 1g \
  graph-mcp-server
```

## Updates and Maintenance

### Update Process

1. **Backup Current Data**
```bash
podman volume export graph-mcp-memory > backup_before_update.tar
```

2. **Stop and Remove Container**
```bash
podman stop graph-mcp-server
podman rm graph-mcp-server
```

3. **Rebuild Image**
```bash
podman build -t graph-mcp-server ./mcp_server
```

4. **Start New Container**
```bash
podman run -d \
  --name graph-mcp-server \
  -p 8000:8000 \
  -v graph-mcp-memory:/data \
  graph-mcp-server
```

5. **Verify Operation**
```bash
curl http://localhost:8000/
curl http://localhost:8000/entities
```

### Maintenance Tasks

**Weekly:**
- Check logs for errors
- Verify backup integrity
- Monitor resource usage

**Monthly:**
- Update base image
- Review security patches
- Clean old logs

**Quarterly:**
- Performance review
- Capacity planning
- Security audit

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Deploy MCP Server

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build container
      run: podman build -t graph-mcp-server ./mcp_server
      
    - name: Deploy container
      run: |
        podman stop graph-mcp-server || true
        podman rm graph-mcp-server || true
        podman run -d \
          --name graph-mcp-server \
          -p 8000:8000 \
          -v graph-mcp-memory:/data \
          graph-mcp-server
          
    - name: Health check
      run: |
        sleep 10
        curl -f http://localhost:8000/ || exit 1
```

## Related Documentation

- [API Reference](./API.md) - Complete REST API documentation
- [MCP Integration](./MCP_INTEGRATION.md) - AI agent integration guide
- [VS Code Setup](./VS_CODE_SETUP.md) - Agent Chat configuration
