"""
OpenTelemetry SDK setup with Prometheus exporter.
Provides a tracer for creating spans across the application.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


def configure_tracing(service_name: str) -> trace.Tracer:
    """
    Configure the OpenTelemetry TracerProvider.
    For this deployment we use the ConsoleSpanExporter; in production
    swap for OTLPSpanExporter pointing at a Jaeger/Tempo collector.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Console exporter for local visibility — replace with OTLP in production
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def configure_metrics() -> None:
    """Configure OpenTelemetry Prometheus metric reader."""
    reader = PrometheusMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    set_meter_provider(provider)


def instrument_fastapi(app) -> None:
    """Attach OTel auto-instrumentation to a FastAPI application instance."""
    FastAPIInstrumentor.instrument_app(app)
