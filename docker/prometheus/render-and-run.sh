#!/bin/sh
set -eu

sed "s|__WRAPPER_METRICS_TARGET__|${WRAPPER_METRICS_TARGET}|g" /etc/prometheus/prometheus.yml > /tmp/prometheus.yml
exec /bin/prometheus --config.file=/tmp/prometheus.yml
