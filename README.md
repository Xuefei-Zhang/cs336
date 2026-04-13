# aiinfra-e2e

Minimal scaffold for the AIInfra E2E LLM LoRA/QLoRA project.

## Setup

```bash
make setup
```

## Lint

```bash
make lint
```

## Tests

```bash
make test
```

## Observability

Start the wrapper metrics endpoint first so Prometheus has something to scrape, then bring up the observability stack:

```bash
make obs-up
```

Stop it with:

```bash
make obs-down
```

Ports:

- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default login: `admin` / `admin`)

Prometheus scrapes the wrapper metrics target from `WRAPPER_METRICS_TARGET`, which defaults to `host.docker.internal:9100` to match the serve wrapper's default Prometheus port.

On Linux, Docker may need host-gateway support for `host.docker.internal`; this compose file includes `extra_hosts: host.docker.internal:host-gateway`, but if your Docker setup does not support that alias you can set `WRAPPER_METRICS_TARGET` to another reachable host:port before running `make obs-up`.
