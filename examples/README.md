# agentevals Examples

This directory contains example agents demonstrating how to use agentevals for live streaming evaluation.

## Live Streaming Setup

To stream traces from your agent code to agentevals in real-time, your agent needs:

### 1. OpenTelemetry Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
trace.set_tracer_provider(provider)
```

### 2. Streaming Processor Configuration

```python
from agentevals.streaming.processor import AgentEvalsStreamingProcessor

processor = AgentEvalsStreamingProcessor(
    ws_url="ws://localhost:8001/ws/traces",
    session_id="your-session-id",  # Unique identifier for this run
    trace_id="your-trace-id",       # Unique identifier for this trace
)
```

### 3. Connect and Register

```python
await processor.connect(
    eval_set_id="your_eval_set",    # The eval set to use for evaluation
    metadata={"model": "...", ...}  # Optional metadata
)

provider.add_span_processor(processor)
```

### 4. Run Your Agent

Once configured, run your agent normally. All OpenTelemetry spans created during execution will automatically stream to the agentevals dev server.

### 5. Shutdown

```python
await processor.shutdown_async()
```

## Running the Dev Server

```bash
# Terminal 1: Start agentevals dev server
agentevals serve --dev --port 8001

# Terminal 2: Start UI (optional)
cd ui && npm run dev
```

Navigate to http://localhost:5173 and select "I am developing an agent" to see traces stream in live.

## Examples

- **dice_agent/** - Simple ADK agent demonstrating live streaming with tool usage evaluation

See each example's README for detailed usage instructions.
