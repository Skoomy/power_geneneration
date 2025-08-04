# Power Generation Framework - Docker Setup

This document explains how to use Docker with the Power Generation Framework.

## Quick Start

1. **Build and start the application:**
   ```bash
   cd docker
   ./run.sh build
   ./run.sh start
   ```

2. **Access Jupyter Lab:**
   - Open http://localhost:8888 in your browser
   - Use token: `power_generation_token`

## Available Services

### Main Application (`power_generation_app`)
- **Purpose**: Primary service with Jupyter Lab for interactive development
- **Ports**: 8888 (Jupyter), 8080 (Web), 5000 (API)
- **Usage**: `./run.sh start`

### Development Environment (`power_generation_dev`)
- **Purpose**: Interactive development container
- **Usage**: 
  ```bash
  ./run.sh dev
  docker exec -it power_generation_dev bash
  ```

### Training Environment (`power_generation_training`)
- **Purpose**: Dedicated environment for model training
- **Usage**:
  ```bash
  ./run.sh train
  docker exec -it power_generation_training python your_training_script.py
  ```

### Full Stack
- **Purpose**: Complete environment with database and cache
- **Includes**: Main app + PostgreSQL + Redis
- **Usage**: `./run.sh full`

## Helper Script Commands

The `run.sh` script provides convenient commands:

```bash
# Build the Docker image
./run.sh build

# Start main application
./run.sh start

# Start development environment
./run.sh dev

# Start training environment
./run.sh train

# Start full stack (app + database + cache)
./run.sh full

# Stop all services
./run.sh stop

# Clean up (remove containers and volumes)
./run.sh clean

# Show logs
./run.sh logs [service_name]

# Show status
./run.sh status

# Execute command in container
./run.sh exec [service_name] [command]

# Show help
./run.sh help
```

## Service Profiles

Docker Compose uses profiles to organize different environments:

- **Default**: Main application only
- **`dev`**: Development tools and utilities
- **`training`**: Model training environment
- **`db`**: PostgreSQL database
- **`cache`**: Redis cache
- **`monitoring`**: Grafana dashboard

## Environment Configuration

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Modify settings in `.env` as needed:**
   - Database credentials
   - API keys
   - Model parameters
   - Paths and ports

## Volume Mounts

The containers mount several directories for persistent data:

- `../src` → `/app/src` - Source code (for development)
- `../data` → `/app/data` - Input data
- `../outputs` → `/app/outputs` - Results and outputs
- `../models` → `/app/models` - Trained models
- `../logs` → `/app/logs` - Application logs
- `../notebooks` → `/app/notebooks` - Jupyter notebooks

## GPU Support

To enable GPU support for PyTorch:

1. **Install NVIDIA Docker runtime**
2. **Uncomment GPU sections in docker-compose.yml**
3. **Set `CUDA_VISIBLE_DEVICES=0` in environment**

## Common Use Cases

### Interactive Development
```bash
# Start development environment
./run.sh dev

# Open a shell in the container
docker exec -it power_generation_dev bash

# Run Python interactively
docker exec -it power_generation_dev python
```

### Model Training
```bash
# Start training environment
./run.sh train

# Run a training script
docker exec -it power_generation_training python -m src.forecaster.trainer

# Monitor training progress
./run.sh logs power_generation_training
```

### Data Analysis
```bash
# Start main application with Jupyter
./run.sh start

# Access Jupyter Lab at http://localhost:8888
# Token: power_generation_token
```

### Production Deployment
```bash
# Start full stack with monitoring
./run.sh full
docker-compose --profile monitoring up -d

# Access services:
# - Application: http://localhost:8888
# - Database: localhost:5432
# - Cache: localhost:6379
# - Monitoring: http://localhost:3000
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
./run.sh logs power_generation_app

# Check container status
./run.sh status

# Rebuild image
./run.sh build
```

### Permission Issues
```bash
# Fix file permissions
docker exec -it power_generation_app chown -R $(id -u):$(id -g) /app/outputs
```

### Database Connection Issues
```bash
# Check database status
./run.sh logs postgres_db

# Connect to database
docker exec -it power_generation_db psql -U power_user -d power_generation
```

### Memory Issues
- Increase Docker memory limit in Docker Desktop
- Reduce batch sizes in training scripts
- Use CPU-only mode: set `CUDA_VISIBLE_DEVICES=-1`

## Network Configuration

All services run on the `power_gen_network` bridge network, allowing internal communication between containers.

## Security Notes

- Change default passwords in production
- Use environment files for sensitive data
- Consider using Docker secrets for production deployments
- Restrict network access as needed

## Performance Optimization

- Use `.dockerignore` to exclude unnecessary files
- Mount only necessary volumes
- Use multi-stage builds for smaller images
- Consider using alpine-based images for production
