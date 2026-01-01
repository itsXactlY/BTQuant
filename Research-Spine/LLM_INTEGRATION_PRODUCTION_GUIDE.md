# LLM Integration Production Guide

## Overview

This guide documents the complete LLM integration for the Autonomous Quantitative Research Agency, providing production-ready strategy generation with intelligent routing between LLM agents and legacy components.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EnhancedStrategyGenerator                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Intelligent Router                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │  │
│  │  │  LLM Agents  │  │  Templates   │  │  Fallback   │ │  │
│  │  │              │  │              │  │             │ │  │
│  │  │ • Strategy   │  │ • MA Crossover│  │ • Emergency │ │  │
│  │  │ • Feedback   │  │ • RSI Mean Rev│  │ • Conservative│ │
│  │  │ • Validation │  │ • Bollinger   │  │             │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Monitoring System                             │  │
│  │  • Metrics Collection                                 │  │
│  │  • Health Monitoring                                  │  │
│  │  • Circuit Breaker                                    │  │
│  │  • Alert System                                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Request** → Intelligent Router
2. **Decision** → Component Selection (LLM/Template/Fallback)
3. **Execution** → Strategy Generation
4. **Validation** → Quality Check
5. **Monitoring** → Metrics Collection
6. **Response** → Generated Strategy

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt

# Start Ollama server (if using Ollama)
ollama serve

# Pull required model
ollama pull qwen2.5:7b
```

### Configuration

#### 1. Environment Variables

```bash
# LLM Configuration
export LLM_PROVIDER="ollama"
export LLM_BASE_URL="http://localhost:11434"
export LLM_MODEL="qwen2.5:7b"
export LLM_TIMEOUT="30"
export LLM_MAX_RETRIES="3"

# Generation Mode
export LLM_GENERATION_MODE="adaptive"

# Quality Thresholds
export LLM_FALLBACK_THRESHOLD="0.6"
export LLM_MIN_NOVELTY_SCORE="0.7"
export LLM_MIN_VALIDATION_SCORE="0.8"

# Performance
export LLM_MAX_GENERATION_TIME="10.0"
export LLM_LATENCY_THRESHOLD="5.0"

# Monitoring
export LLM_ENABLE_MONITORING="true"
export LLM_METRICS_RETENTION_DAYS="30"
```

#### 2. Configuration Profiles

Create configuration profiles in `config/llm_configs/`:

**Default Profile** (`config/llm_configs/default.json`):
```json
{
  "provider": "ollama",
  "base_url": "http://localhost:11434",
  "model_name": "qwen2.5:7b",
  "timeout": 30,
  "max_retries": 3,
  "temperature": 0.8,
  "max_tokens": 2000,
  "generation_mode": "adaptive",
  "fallback_threshold": 0.6,
  "max_consecutive_failures": 3,
  "latency_threshold": 5.0,
  "min_novelty_score": 0.7,
  "min_validation_score": 0.8,
  "max_generation_time": 10.0,
  "enable_monitoring": true,
  "metrics_retention_days": 30,
  "circuit_breaker_enabled": true,
  "circuit_breaker_threshold": 5,
  "circuit_breaker_timeout": 300
}
```

**Production Profile** (`config/llm_configs/production.json`):
```json
{
  "provider": "ollama",
  "base_url": "http://llm-server:11434",
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

## Usage

### Basic Strategy Generation

```python
from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator

# Initialize with default configuration
generator = EnhancedStrategyGenerator(config_profile="default")

# Generate a single strategy
strategy = generator.generate_strategy(
    strategy_type="physics_based",
    market_context={
        "volatility": "high",
        "trend": "bullish"
    },
    performance_requirements={
        "min_novelty": 0.8
    }
)

print(f"Generated: {strategy['name']}")
print(f"Validation Score: {strategy['validation_report']['overall_score']:.2f}")
```

### Population Generation

```python
# Generate diverse population
strategies = generator.generate_strategy_population(
    population_size=15,
    strategy_types=['innovative', 'physics_based', 'biology_based', 'game_theory'],
    max_generation_time=120  # 2 minutes max
)

print(f"Generated {len(strategies)} strategies")
print(f"Average Novelty: {sum(s.get('metadata', {}).get('novelty_score', 0) for s in strategies) / len(strategies):.2f}")
```

### Strategy Refinement

```python
# After backtesting, refine based on performance
performance_metrics = {
    'sharpe_ratio': 1.2,
    'max_drawdown': 15.0,
    'win_rate': 55.0,
    'profit_factor': 1.5,
    'total_return': 25.0
}

refined_strategy = generator.refine_strategy(
    strategy=strategy,
    performance_metrics=performance_metrics,
    refinement_iterations=2
)
```

### Validation

```python
# Validate a strategy
is_valid, validation_report = generator.validate_strategy(strategy)

if is_valid:
    print("Strategy passed validation")
    print(f"Score: {validation_report['overall_score']:.2f}")
else:
    print("Strategy failed validation")
    print(f"Failed checks: {validation_report['failed_checks']}")
```

### System Status

```python
# Get comprehensive system status
status = generator.get_system_status()

print(f"LLM Operational: {status['llm_enabled']}")
print(f"Health Score: {status['router_health']['llm_health_score']:.2f}")
print(f"Generation Stats: {status['generation_stats']}")
```

## Intelligent Routing

### Decision Logic

The intelligent router selects components based on:

1. **LLM Health Score** (0.0-1.0)
   - Success rate
   - Latency
   - Consecutive failures

2. **Strategy Requirements**
   - Innovation level
   - Novelty requirements
   - Performance targets

3. **Context Factors**
   - Strategy type
   - Market conditions
   - Time constraints

### Routing Scenarios

| Scenario | Health Score | Strategy Type | Decision |
|----------|--------------|---------------|----------|
| High innovation needed | >0.8 | Physics-based | **LLM** |
| Moderate health | 0.6-0.8 | General | **Hybrid** |
| Low health | <0.6 | Any | **Template** |
| Critical failures | Any | Any | **Fallback** |
| Time critical | Any | Any | **Template** |

### Circuit Breaker

The circuit breaker protects against LLM failures:

- **CLOSED**: Normal operation
- **OPEN**: LLM suspended, using fallbacks
- **HALF_OPEN**: Testing recovery

**Automatic Recovery**: After 5 minutes, system attempts to reconnect.

## Monitoring & Metrics

### Real-time Monitoring

```python
# Get health status
health = generator.monitoring.get_health_status()
print(f"Health Score: {health.health_score:.2f}")
print(f"Success Rate: {health.success_rate:.2%}")
print(f"Circuit Breaker: {health.circuit_breaker_state}")
```

### Performance Summary

```python
# Get performance metrics
summary = generator.monitoring.get_performance_summary()
print(f"Average Latency: {summary['avg_latency']:.2f}s")
print(f"Success Rate: {summary['success_rate']:.2%}")
print(f"Total Cost: ${summary['total_cost']:.2f}")
```

### System Health Report

```python
# Comprehensive health report
report = generator.monitoring.get_system_health_report()

print("System Health Report:")
print(f"  Health Score: {report['health_status']['health_score']:.2f}")
print(f"  Recommendations: {report['recommendations']}")
```

### Metrics Export

```python
# Export metrics to file
generator.monitoring.export_metrics("metrics_export.json")

# Export system report
generator.export_system_report("system_report.json")
```

## Error Handling & Fallbacks

### Error Types

1. **LLM Connection Failure**
   - Automatic fallback to templates
   - Circuit breaker activation

2. **Generation Failure**
   - Retry with different parameters
   - Fallback to conservative strategies

3. **Validation Failure**
   - Automatic refinement attempts
   - Rejection of invalid strategies

4. **Timeout**
   - Switch to faster generation methods
   - Emergency fallback strategies

### Fallback Hierarchy

```
LLM Generation → Template Generation → Emergency Fallback → Conservative Strategy
```

### Manual Intervention

```python
# Force fallback mode
generator.router.force_fallback_mode()

# Attempt recovery
recovered = generator.router.recover_llm_mode()
print(f"Recovery successful: {recovered}")
```

## Performance Optimization

### Generation Time Optimization

```python
# Configure for speed
config = {
    "max_generation_time": 5.0,
    "timeout": 10,
    "max_retries": 1,
    "generation_mode": "template_only"  # Fastest
}

generator = EnhancedStrategyGenerator(config_profile="fast")
```

### Quality Optimization

```python
# Configure for quality
config = {
    "min_novelty_score": 0.9,
    "min_validation_score": 0.95,
    "temperature": 0.6,  # More focused
    "max_tokens": 3000
}

generator = EnhancedStrategyGenerator(config_profile="high_quality")
```

### Cost Optimization

```python
# Configure for cost
config = {
    "max_tokens": 1000,
    "temperature": 0.7,
    "max_retries": 2,
    "fallback_threshold": 0.8  # Use templates more often
}

generator = EnhancedStrategyGenerator(config_profile="cost_efficient")
```

## Deployment

### Docker Setup

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p monitoring_exports config/llm_configs

# Set environment variables
ENV PYTHONPATH=/app
ENV LLM_PROFILE=production

CMD ["python", "main.py"]
```

**docker-compose.yml**:
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

  strategy-generator:
    build: .
    depends_on:
      - llm-server
    volumes:
      - ./monitoring_exports:/app/monitoring_exports
      - ./config/llm_configs:/app/config/llm_configs
    environment:
      - LLM_BASE_URL=http://llm-server:11434
      - LLM_PROFILE=production
    restart: unless-stopped

volumes:
  ollama_data:
```

### Kubernetes Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategy-generator
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
        env:
        - name: LLM_BASE_URL
          value: "http://llm-service:11434"
        - name: LLM_PROFILE
          value: "production"
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: config
          mountPath: /app/config/llm_configs
        - name: monitoring
          mountPath: /app/monitoring_exports
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator; g = EnhancedStrategyGenerator(); print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 60
      volumes:
      - name: config
        configMap:
          name: strategy-generator-config
      - name: monitoring
        persistentVolumeClaim:
          claimName: strategy-generator-monitoring
```

### Production Configuration

**config/llm_configs/production.json**:
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

## Troubleshooting

### Common Issues

#### 1. LLM Connection Failed

**Symptoms**: Circuit breaker opens, all generations use templates

**Solution**:
```bash
# Check Ollama server
curl http://localhost:11434/api/version

# Restart Ollama
ollama serve

# Test model
ollama run qwen2.5:7b "test"

# Force recovery in Python
generator.router.recover_llm_mode()
```

#### 2. High Latency

**Symptoms**: Generation takes >5 seconds

**Solutions**:
- Reduce `max_tokens` in config
- Use smaller model
- Increase `latency_threshold`
- Switch to template-only mode

#### 3. Low Success Rate

**Symptoms**: Many strategies fail validation

**Solutions**:
- Increase `min_validation_score` threshold
- Use more conservative generation
- Enable feedback refinement
- Review prompt engineering

#### 4. Memory Issues

**Symptoms**: System runs out of memory

**Solutions**:
- Reduce `metrics_retention_days`
- Export metrics more frequently
- Use smaller model
- Increase system memory

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with verbose logging
generator = EnhancedStrategyGenerator(config_profile="default")

# Check detailed status
status = generator.get_system_status()
print(json.dumps(status, indent=2))
```

### Health Checks

```python
def health_check():
    """Comprehensive health check"""
    generator = EnhancedStrategyGenerator()
    status = generator.get_system_status()
    
    # Check LLM health
    if not status['llm_enabled']:
        return "LLM disabled", 503
    
    health = status['router_health']
    if health['llm_health_score'] < 0.5:
        return "LLM unhealthy", 503
    
    # Check circuit breaker
    if health['circuit_breaker_state'] == 'OPEN':
        return "Circuit breaker open", 503
    
    return "Healthy", 200
```

## Best Practices

### 1. Configuration Management

- Use environment-specific profiles
- Keep sensitive data in environment variables
- Version control configuration files
- Test configuration changes

### 2. Monitoring

- Set up alerts for health score <0.7
- Monitor generation times
- Track costs and token usage
- Review metrics daily

### 3. Error Handling

- Always implement fallback logic
- Monitor circuit breaker state
- Set appropriate timeouts
- Log all failures

### 4. Performance

- Use appropriate generation mode for use case
- Monitor and optimize generation times
- Implement caching where possible
- Use batch operations for populations

### 5. Security

- Keep API keys secure
- Validate all inputs
- Monitor for unusual patterns
- Implement rate limiting

### 6. Maintenance

- Regular system health checks
- Update models and dependencies
- Review and clean up metrics
- Test recovery procedures

## Support & Resources

### Documentation Files

- `LLM_INTEGRATION_SUMMARY.md` - High-level overview
- `LLM_INTEGRATION_PRODUCTION_GUIDE.md` - This file
- `strategy_generation/integration/` - Integration components
- `config/llm_config.py` - Configuration management

### Key Components

- `EnhancedStrategyGenerator` - Main entry point
- `IntelligentRouter` - Component selection
- `MonitoringSystem` - Metrics and health
- `LLMConfig` - Configuration management

### Getting Help

1. Check system status: `generator.get_system_status()`
2. Review monitoring metrics
3. Check logs in `logs/system.log`
4. Export system report for analysis

---

**Version**: 1.0  
**Last Updated**: 2026-01-01  
**Status**: Production Ready