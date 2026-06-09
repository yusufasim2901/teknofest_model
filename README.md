# teknofest_model — Multi-Agent System (MAS) Infrastructure

Event-driven backbone for a **5G-enabled smart road safety platform**.  
Agents process live RTSP video feeds asynchronously, communicate via RabbitMQ topic exchange, and persist state through Redis & PostgreSQL.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                       │
│                                                                │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────┐             │
│  │  RabbitMQ     │  │  Redis   │  │  PostgreSQL  │             │
│  │  (Broker)     │  │  (Cache) │  │  (Store)     │             │
│  └──────┬───────┘  └────┬─────┘  └──────┬───────┘             │
│         │               │               │                      │
│         └───────────┬───┴───────────────┘                      │
│                     │                                          │
│              ┌──────┴──────┐                                   │
│              │  FastAPI    │                                    │
│              │  (EventBus) │                                   │
│              └─────────────┘                                   │
└────────────────────────────────────────────────────────────────┘

Agents:  Perception · Detection · Behavior Analytics · 5G Negotiation · Decision & Security
```

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose v2+
- Python 3.12+ (for running the PoC script locally)

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with production credentials
```

### 2. Launch the Stack

```bash
docker-compose up -d
```

Verify all services are healthy:

```bash
docker-compose ps
```

### 3. Check the API

```bash
curl http://localhost:8000/health
# → {"status": "healthy", "rabbitmq": "connected", "redis": "connected"}
```

### 4. Run the Proof-of-Concept

```bash
# Install dependencies locally (in a venv)
pip install -r requirements.txt

# Run the PoC (connects to RabbitMQ on localhost)
RABBITMQ_HOST=localhost python scripts/poc_hello_world.py
```

Expected output:

```
12:00:00 │ INFO     │ poc │ 🔵 [PerceptionAgent] Published  → perception.hello  (seq=1, ...)
12:00:00 │ INFO     │ poc │ 🟢 [DetectionAgent] Received   ← perception.hello  |  payload=Hello from PerceptionAgent! (seq=1)  (total_rx=1)
```

Press `Ctrl+C` to stop gracefully.

---

## Project Structure

```
├── docker-compose.yml         # Service orchestration
├── Dockerfile                 # FastAPI container image
├── .env.example               # Environment variable template
├── requirements.txt           # Pinned Python dependencies
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings (env management)
│   ├── main.py                # FastAPI app + EventBus class
│   ├── models/
│   │   └── events.py          # AgentEvent schema
│   └── agents/
│       └── base.py            # Abstract BaseAgent class
└── scripts/
    └── poc_hello_world.py     # Standalone PoC demo
```

## API Endpoints

| Method | Path              | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Liveness probe (RabbitMQ + Redis)    |
| POST   | `/events/publish` | Manual event injection (debug)       |

## Management UIs

| Service   | URL                          | Credentials             |
|-----------|------------------------------|-------------------------|
| RabbitMQ  | http://localhost:15672        | See `.env`              |
| API Docs  | http://localhost:8000/docs   | —                       |

## License

See [LICENSE](./LICENSE).