# Monitoring Dashboards

This directory contains monitoring dashboard configurations for vLLM, providing
comprehensive observability for your vLLM deployments.

## Dashboard Platforms

We provide dashboards for two popular observability platforms:

- **[Grafana](https://grafana.com)**
- **[Perses](https://perses.dev)**

## Dashboard Format Approach

All dashboards are provided in **native formats** that work across different
deployment methods:

### Grafana (JSON)

- ✅ Works with any Grafana instance (cloud, self-hosted, Docker)
- ✅ Direct import via Grafana UI or API
- ✅ Can be wrapped in Kubernetes operators when needed
- ✅ No vendor lock-in or deployment dependencies

### Perses (YAML)

- ✅ Works with standalone Perses instances
- ✅ Compatible with Perses API and CLI
- ✅ Supports Dashboard-as-Code workflows
- ✅ Can be wrapped in Kubernetes operators when needed

## Dashboard Contents

Both platforms provide equivalent monitoring capabilities:

| Dashboard | Description |
|-----------|-------------|
| **Performance Statistics** | Tracks latency, throughput, and performance metrics |
| **Query Statistics** | Monitors request volume, query performance, and KPIs |

## Quick Start

First, navigate to this example's directory:

```bash
cd examples/online_serving/dashboards
```

### Grafana

Import the JSON directly into the Grafana UI, or use the API:

```bash
curl -X POST http://grafana/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/performance_statistics.json
```

### Perses

Import via the Perses CLI:

```bash
percli apply -f perses/performance_statistics.yaml
```

## Requirements

- **Prometheus** metrics from your vLLM deployment
- **Data source** configured in your monitoring platform
- **vLLM metrics** enabled and accessible

## Metrics vs Logs: Best Practices

All vLLM dashboards use **Prometheus metrics** exclusively rather than log-based queries.
This is the recommended approach for production observability:

### Why Metrics Over Logs

| Aspect | Metrics (Prometheus) | Logs |
|--------|---------------------|------|
| **Performance** | Low overhead, pre-aggregated | High cardinality, expensive queries |
| **Reliability** | Time-series optimized storage | Requires log aggregation infrastructure |
| **Alerting** | Native PromQL support | Complex log parsing required |
| **Cost** | Efficient storage and queries | Higher storage and query costs |
| **Real-time** | Sub-second resolution | Depends on log shipping latency |

### Available Metrics

vLLM exposes comprehensive Prometheus metrics via the `/metrics` endpoint:

**Latency Metrics (Histograms)**
- `vllm:e2e_request_latency_seconds` - End-to-end request latency
- `vllm:time_to_first_token_seconds` - Time to first token (TTFT)
- `vllm:time_per_output_token_seconds` - Inter-token latency (TPOT)
- `vllm:inter_token_latency_seconds` - Inter-token latency
- `vllm:request_queue_time_seconds` - Queue waiting time
- `vllm:request_prefill_time_seconds` - Prefill phase duration
- `vllm:request_decode_time_seconds` - Decode phase duration

**Throughput Metrics (Counters)**
- `vllm:prompt_tokens_total` - Total prompt tokens processed
- `vllm:generation_tokens_total` - Total tokens generated
- `vllm:request_success_total` - Successful requests by finish reason

**Resource Metrics (Gauges)**
- `vllm:num_requests_running` - Currently running requests
- `vllm:num_requests_waiting` - Requests waiting in queue
- `vllm:kv_cache_usage_perc` - KV cache utilization percentage

**Token Distribution (Histograms)**
- `vllm:request_prompt_tokens` - Input prompt token counts
- `vllm:request_generation_tokens` - Output generation token counts

For the complete list of metrics, see the [Production Metrics documentation](../../../docs/usage/metrics.md).

## Platform-Specific Documentation

For detailed deployment instructions and platform-specific options, see:

- **[Grafana Documentation](./grafana)** - JSON dashboards, operator usage, manual import
- **[Perses Documentation](./perses)** - YAML specs, CLI usage, operator wrapping

## Contributing

When adding new dashboards, please:

1. Provide native formats (JSON for Grafana, YAML specs for Perses)
2. Update platform-specific README files
3. Ensure dashboards work across deployment methods
4. Test with the latest platform versions
