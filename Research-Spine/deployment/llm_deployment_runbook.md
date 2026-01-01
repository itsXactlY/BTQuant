# LLM Integration Deployment Runbook

## Quick Start

### Prerequisites Check
```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Check system resources
free -h
df -h

# 3. Check network connectivity
ping -c 3 localhost
```

### One-Command Deployment
```bash
# Clone and setup
git clone <repository>
cd <repository>

# Install dependencies
pip install -r requirements.txt

# Start Ollama (if not running)
ollama serve &

# Pull model
ollama pull qwen2.5:7b

# Run deployment test
python test_llm_integration_comprehensive.py

# Start the system
python main.py
```

## Architecture Overview

### Components
- **EnhancedStrategyGenerator**: Main entry point
- **IntelligentRouter**: Dynamic component selection
- **MonitoringSystem**: Metrics and health tracking
- **LLMConfig**: Configuration management

### Data Flow
```
User Request → Intelligent Router → Component Selection → Strategy Generation → Validation → Response
```

## Installation

### 1. System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Network**: Localhost or LLM server access

### 2. Environment Setup

**Option A: Local Development**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export LLM_PROVIDER="ollama"
export LLM_BASE_URL="http://localhost:11434"
export LLM_PROFILE="development"
```

**Option B: Docker**
```bash
# Build image
docker build -t strategy-generator:latest .

# Run container
docker run -d \
  --name strategy-gen \
  -p 8000:8000 \
  -v $(pwd)/monitoring_exports:/app/monitoring_exports \
  -v $(pwd)/config/llm_configs:/app/config/llm_configs \
  -e LLM_BASE_URL=http://host.docker.internal:11434 \
  strategy-generator:latest
```

**Option C: Kubernetes**
```bash
# Apply configuration
kubectl apply -f deployment/k8s/

# Check status
kubectl get pods -l app=strategy-generator
kubectl logs -f deployment/strategy-generator
```

### 3. Model Setup

**Ollama Setup**
```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve

# Pull model (choose one)
ollama pull qwen2.5:7b      # Balanced performance
ollama pull llama2:7b       # Alternative
ollama pull mistral:7b      # Fast inference

# Verify installation
ollama list
curl http://localhost:11434/api/version
```

**Model Selection Guide**
| Model | Speed | Quality | Memory | Use Case |
|-------|-------|---------|--------|----------|
| qwen2.5:7b | Fast | Good | ~4GB | General purpose |
| llama2:7b | Medium | Good | ~4GB | Balanced |
| mistral:7b | Fast | Very Good | ~4GB | High performance |
| qwen2.5:14b | Slow | Excellent | ~8GB | Maximum quality |

## Configuration

### 1. Profile Setup

**Development** (`config/llm_configs/development.json`)
```json
{
  "provider": "ollama",
  "base_url": "http://localhost:11434",
  "model_name": "qwen2.5:7b",
  "timeout": 15,
  "max_retries": 2,
  "temperature": 0.8,
  "max_tokens": 1500,
  "generation_mode": "adaptive",
  "fallback_threshold": 0.5,
  "max_consecutive_failures": 3,
  "latency_threshold": 8.0,
  "min_novelty_score": 0.6,
  "min_validation_score": 0.7,
  "max_generation_time": 15.0,
  "enable_monitoring": true,
  "metrics_retention_days": 7,
  "circuit_breaker_enabled": true,
  "circuit_breaker_threshold": 5,
  "circuit_breaker_timeout": 180
}
```

**Production** (`config/llm_configs/production.json`)
```json
{
  "provider": "ollama",
  "base_url": "http://llm-service:11434",
  "model_name": "qwen2.5:7b",
  "timeout": 60,
  "max_retries": 5,
  "temperature": 0.7,
  "max_tokens": 2500,
  "generation_mode": "hybrid",
  "fallback_threshold": 0.7,
  "max_consecutive_failures": 2,
  "latency_threshold": 3.0,
  "min_novelty_score": 0.8,
  "min_validation_score": 0.9,
  "max_generation_time": 8.0,
  "enable_monitoring": true,
  "metrics_retention_days": 90,
  "circuit_breaker_enabled": true,
  "circuit_breaker_threshold": 3,
  "circuit_breaker_timeout": 600
}
```

### 2. Environment Variables

**Required**
```bash
export LLM_PROVIDER="ollama"
export LLM_BASE_URL="http://localhost:11434"
export LLM_PROFILE="production"
```

**Optional**
```bash
export LLM_MODEL="qwen2.5:7b"
export LLM_TIMEOUT="30"
export LLM_MAX_RETRIES="3"
export LLM_GENERATION_MODE="adaptive"
export LLM_FALLBACK_THRESHOLD="0.6"
export LLM_MIN_NOVELTY_SCORE="0.7"
export LLM_ENABLE_MONITORING="true"
```

## Deployment Procedures

### 1. Development Deployment

**Step 1: Setup**
```bash
# Clone repository
git clone <repo> strategy-generator
cd strategy-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup directories
mkdir -p monitoring_exports config/llm_configs logs
```

**Step 2: Configure**
```bash
# Copy default config
cp config/llm_configs/default.json config/llm_configs/development.json

# Edit as needed
nano config/llm_configs/development.json

# Set environment
export LLM_PROFILE="development"
export LLM_BASE_URL="http://localhost:11434"
```

**Step 3: Test**
```bash
# Run comprehensive tests
python test_llm_integration_comprehensive.py

# Check results
cat test_llm_integration_results.json
```

**Step 4: Run**
```bash
# Start system
python main.py

# Or run in background
nohup python main.py > logs/system.log 2>&1 &
```

### 2. Production Deployment

**Step 1: Infrastructure**
```bash
# Ensure LLM server is running
systemctl status ollama

# Check resources
htop
df -h

# Verify network
curl http://llm-service:11434/api/version
```

**Step 2: Configuration**
```bash
# Use production profile
export LLM_PROFILE="production"

# Verify config
python -c "from config.llm_config import get_llm_config; c = get_llm_config('production'); print(c.to_json())"
```

**Step 3: Health Check**
```bash
# Run health check script
python -c "
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
g = EnhancedStrategyGenerator('production')
status = g.get_system_status()
print('LLM Enabled:', status['llm_enabled'])
print('Health Score:', status['router_health']['llm_health_score'])
"
```

**Step 4: Start Service**
```bash
# Using systemd
sudo cp deployment/strategy-generator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable strategy-generator
sudo systemctl start strategy-generator

# Check logs
sudo journalctl -u strategy-generator -f
```

### 3. Docker Deployment

**Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p monitoring_exports config/llm_configs logs

# Set environment variables
ENV PYTHONPATH=/app
ENV LLM_PROFILE=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator; g = EnhancedStrategyGenerator(); print('OK')"

CMD ["python", "main.py"]
```

**Docker Compose**
```yaml
version: '3.8'

services:
  llm-server:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=24h
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

  strategy-generator:
    build: .
    depends_on:
      - llm-server
    ports:
      - "8000:8000"
    volumes:
      - ./monitoring_exports:/app/monitoring_exports
      - ./config/llm_configs:/app/config/llm_configs
      - ./logs:/app/logs
    environment:
      - LLM_BASE_URL=http://llm-server:11434
      - LLM_PROFILE=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

volumes:
  ollama_data:
```

**Deploy with Docker Compose**
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f strategy-generator

# Scale if needed
docker-compose up -d --scale strategy-generator=2

# Stop
docker-compose down
```

### 4. Kubernetes Deployment

**Namespace**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: strategy-generator
```

**ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: strategy-generator-config
  namespace: strategy-generator
data:
  production.json: |
    {
      "provider": "ollama",
      "base_url": "http://llm-service:11434",
      "model_name": "qwen2.5:7b",
      "timeout": 60,
      "max_retries": 5,
      "generation_mode": "hybrid",
      "fallback_threshold": 0.7,
      "min_novelty_score": 0.8,
      "min_validation_score": 0.9,
      "enable_monitoring": true
    }
```

**Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategy-generator
  namespace: strategy-generator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: strategy-generator
  template:
    metadata:
      labels:
        app: strategy-generator
    spec:
      containers:
      - name: generator
        image: strategy-generator:latest
        imagePullPolicy: Always
        env:
        - name: LLM_BASE_URL
          value: "http://llm-service:11434"
        - name: LLM_PROFILE
          value: "production"
        - name: PYTHONPATH
          value: "/app"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: config
          mountPath: /app/config/llm_configs
          readOnly: true
        - name: monitoring
          mountPath: /app/monitoring_exports
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator; g = EnhancedStrategyGenerator(); print('OK')"
          initialDelaySeconds: 60
          periodSeconds: 60
          timeoutSeconds: 10
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator; g = EnhancedStrategyGenerator(); status = g.get_system_status(); exit(0 if status['router_health']['llm_health_score'] > 0.5 else 1)"
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
      volumes:
      - name: config
        configMap:
          name: strategy-generator-config
      - name: monitoring
        persistentVolumeClaim:
          claimName: strategy-generator-monitoring
      - name: logs
        emptyDir: {}
```

**Service**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: strategy-generator-service
  namespace: strategy-generator
spec:
  selector:
    app: strategy-generator
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**Persistent Volume**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: strategy-generator-monitoring
  namespace: strategy-generator
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

**Deploy**
```bash
# Apply all configurations
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/pvc.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml

# Check status
kubectl get all -n strategy-generator

# View logs
kubectl logs -f deployment/strategy-generator -n strategy-generator

# Port forward for testing
kubectl port-forward svc/strategy-generator-service 8080:80 -n strategy-generator
```

## Monitoring & Operations

### 1. Health Checks

**Automated Health Check**
```bash
#!/bin/bash
# health_check.sh

echo "=== Strategy Generator Health Check ==="
echo "Timestamp: $(date)"

# Check LLM server
echo -n "LLM Server: "
curl -s http://localhost:11434/api/version > /dev/null && echo "✓ OK" || echo "✗ FAIL"

# Check system status
python -c "
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
try:
    g = EnhancedStrategyGenerator('production')
    status = g.get_system_status()
    health = status['router_health']
    print(f'Health Score: {health[\"llm_health_score\"]:.2f}')
    print(f'Circuit Breaker: {health[\"circuit_breaker_state\"]}')
    print(f'Success Rate: {health[\"success_rate\"]:.2%}')
except Exception as e:
    print(f'System Status: FAIL - {e}')
"

# Check monitoring exports
echo -n "Monitoring Exports: "
if [ -d "monitoring_exports" ]; then echo "✓ OK"; else echo "✗ FAIL"; fi

echo "=== End Health Check ==="
```

**Kubernetes Health Check**
```bash
# Check pod status
kubectl get pods -n strategy-generator

# Describe pod for issues
kubectl describe pod -n strategy-generator -l app=strategy-generator

# Check logs
kubectl logs -n strategy-generator -l app=strategy-generator --tail=100

# Check events
kubectl get events -n strategy-generator --sort-by='.lastTimestamp'
```

### 2. Metrics Monitoring

**View Real-time Metrics**
```python
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator

generator = EnhancedStrategyGenerator('production')

# Health status
health = generator.monitoring.get_health_status()
print(f"Health Score: {health.health_score:.2f}")
print(f"Success Rate: {health.success_rate:.2%}")
print(f"Latency: {health.avg_latency:.2f}s")

# Performance summary
summary = generator.monitoring.get_performance_summary()
print(f"Total Operations: {summary['stats']['total']}")
print(f"Total Cost: ${summary['total_cost']:.2f}")

# Recent metrics
recent = generator.monitoring.get_recent_metrics(last_n=10)
for metric in recent:
    print(f"{metric['operation']}: {metric['execution_time']:.2f}s - {'✓' if metric['success'] else '✗'}")
```

**Export Metrics**
```python
# Export to file
generator.monitoring.export_metrics("metrics_$(date +%Y%m%d).json")

# Export system report
generator.export_system_report("system_report_$(date +%Y%m%d).json")
```

### 3. Alerting

**Set up alerts for:**
- Health score < 0.7
- Success rate < 80%
- Average latency > 5 seconds
- Circuit breaker opens
- Cost exceeds threshold

**Example Alert Script**
```python
#!/usr/bin/env python3
# alert_check.py

from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
import sys

generator = EnhancedStrategyGenerator('production')
health = generator.monitoring.get_health_status()

alerts = []

if health.health_score < 0.7:
    alerts.append(f"Low health score: {health.health_score:.2f}")

if health.success_rate < 0.8:
    alerts.append(f"Low success rate: {health.success_rate:.2%}")

if health.avg_latency > 5.0:
    alerts.append(f"High latency: {health.avg_latency:.2f}s")

if health.circuit_breaker_state == "OPEN":
    alerts.append("Circuit breaker is OPEN")

if alerts:
    print("ALERTS:")
    for alert in alerts:
        print(f"  - {alert}")
    sys.exit(1)
else:
    print("All systems normal")
    sys.exit(0)
```

## Troubleshooting

### 1. LLM Connection Issues

**Symptoms**: Circuit breaker opens, all generations use templates

**Diagnosis**
```bash
# Check LLM server
curl http://localhost:11434/api/version

# Test model
echo '{"model": "qwen2.5:7b", "prompt": "test", "stream": false}' | \
  curl -X POST http://localhost:11434/api/generate -d @-

# Check logs
journalctl -u ollama -f
```

**Solutions**
```bash
# Restart Ollama
sudo systemctl restart ollama

# Re-pull model
ollama pull qwen2.5:7b

# Check firewall
sudo ufw status
sudo ufw allow 11434

# Verify model exists
ollama list | grep qwen2.5:7b
```

### 2. High Memory Usage

**Symptoms**: System slow, OOM errors

**Diagnosis**
```bash
# Check memory usage
htop
free -h

# Check Python memory
ps aux | grep python

# Check monitoring data size
du -sh monitoring_exports/
```

**Solutions**
```bash
# Reduce metrics retention
export LLM_METRICS_RETENTION_DAYS="7"

# Clear old metrics
rm monitoring_exports/metrics_*.json

# Use smaller model
ollama pull qwen2.5:7b  # Instead of larger models

# Restart with memory limits
docker run -m 4g strategy-generator:latest
```

### 3. Slow Generation

**Symptoms**: Strategies take too long to generate

**Diagnosis**
```python
# Check generation times
generator = EnhancedStrategyGenerator('production')
status = generator.get_system_status()
print(f"Average generation time: {status['generation_stats']['average_generation_time']:.2f}s")
```

**Solutions**
```bash
# Switch to template-only mode
export LLM_GENERATION_MODE="template_only"

# Reduce token limits
export LLM_MAX_TOKENS="1000"

# Use faster model
ollama pull mistral:7b

# Increase timeout threshold
export LLM_LATENCY_THRESHOLD="10.0"
```

### 4. Validation Failures

**Symptoms**: Many strategies fail validation

**Diagnosis**
```python
generator = EnhancedStrategyGenerator('production')
strategy = generator.generate_strategy("innovative")
is_valid, report = generator.validate_strategy(strategy)

print(f"Valid: {is_valid}")
print(f"Score: {report['overall_score']}")
print(f"Failed checks: {report['failed_checks']}")
```

**Solutions**
```python
# Lower validation threshold temporarily
generator.config.min_validation_score = 0.6

# Use feedback refinement
refined = generator.refine_strategy(strategy, performance_metrics)

# Switch to more conservative generation
generator.config.temperature = 0.5
```

### 5. Monitoring Issues

**Symptoms**: No metrics, monitoring errors

**Diagnosis**
```bash
# Check monitoring directory
ls -la monitoring_exports/

# Check permissions
python -c "
import os
try:
    os.makedirs('monitoring_exports', exist_ok=True)
    print('Directory writable: OK')
except Exception as e:
    print(f'Directory issue: {e}')
"
```

**Solutions**
```bash
# Fix permissions
chmod 755 monitoring_exports
chown $USER:$USER monitoring_exports

# Create directory if missing
mkdir -p monitoring_exports

# Disable monitoring temporarily
export LLM_ENABLE_MONITORING="false"
```

## Maintenance

### 1. Daily Tasks
```bash
# Check system health
./health_check.sh

# Review metrics
python -c "
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
g = EnhancedStrategyGenerator('production')
print(g.monitoring.get_metrics_summary(hours=24))
"

# Check logs for errors
grep -i "error\|fail" logs/system.log | tail -20
```

### 2. Weekly Tasks
```bash
# Export and backup metrics
tar -czf metrics_backup_$(date +%Y%m%d).tar.gz monitoring_exports/

# Review performance trends
python -c "
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
g = EnhancedStrategyGenerator('production')
report = g.monitoring.get_metrics_summary(hours=168)  # 7 days
print('Weekly Throughput:', report['throughput_per_hour'])
print('Average Cost:', report['total_cost'] / 7)
"

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete
```

### 3. Monthly Tasks
```bash
# Update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt

# Review and update configuration
# Check for new model versions
ollama list

# Performance review
python -c "
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
g = EnhancedStrategyGenerator('production')
status = g.get_system_status()
print('Monthly Performance Summary:')
print(f'  Total Generated: {status[\"generation_stats\"][\"total_generated\"]}')
print(f'  Avg Time: {status[\"generation_stats\"][\"average_generation_time\"]:.2f}s')
print(f'  LLM Usage: {status[\"generation_stats\"][\"llm_generated\"]}')
print(f'  Template Usage: {status[\"generation_stats\"][\"template_generated\"]}')
"
```

## Rollback Procedures

### 1. Quick Rollback
```bash
# Stop current service
sudo systemctl stop strategy-generator

# Restore previous version
git checkout <previous-commit>

# Restart
sudo systemctl start strategy-generator

# Verify
sudo systemctl status strategy-generator
```

### 2. Configuration Rollback
```bash
# Restore config from backup
cp config/llm_configs/production.json.backup config/llm_configs/production.json

# Restart service
sudo systemctl restart strategy-generator
```

### 3. Emergency Rollback
```bash
# Force fallback mode
python -c "
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
g = EnhancedStrategyGenerator('production')
g.router.force_fallback_mode()
print('Forced fallback mode')
"

# Or switch to template-only
export LLM_GENERATION_MODE="template_only"
sudo systemctl restart strategy-generator
```

## Support Contacts

- **System Administrator**: [Contact]
- **ML Engineer**: [Contact]
- **DevOps**: [Contact]
- **Emergency**: [Contact]

## Version Information

- **Version**: 1.0
- **Last Updated**: 2026-01-01
- **Status**: Production Ready
- **Documentation**: LLM_INTEGRATION_PRODUCTION_GUIDE.md

---

**End of Runbook**