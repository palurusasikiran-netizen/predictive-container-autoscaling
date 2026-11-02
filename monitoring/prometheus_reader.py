import requests

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"


def query_prometheus(query):
    response = requests.get(
        PROMETHEUS_URL,
        params={"query": query},
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if data["status"] != "success":
        raise RuntimeError(data)

    return data["data"]["result"]


# CPU
cpu_query = """
rate(container_cpu_usage_seconds_total{
    namespace="default",
    pod=~"predictive-app-.*",
    container!="POD",
    container!=""
}[5m])
"""

# Memory in MB
memory_query = """
sum(container_memory_working_set_bytes{
    namespace="default",
    pod=~"predictive-app-.*",
    container!="POD",
    container!=""
}) / 1024 / 1024
"""

# Network in KB/s
network_query = """
(
    sum(rate(container_network_receive_bytes_total{
        namespace="default",
        pod=~"predictive-app-.*"
    }[5m]))
    +
    sum(rate(container_network_transmit_bytes_total{
        namespace="default",
        pod=~"predictive-app-.*"
    }[5m]))
) / 1024
"""


if __name__ == "__main__":
    cpu = query_prometheus(cpu_query)
    memory = query_prometheus(memory_query)
    network = query_prometheus(network_query)

    print("CPU:", cpu)
    print("Memory (MB):", memory)
    print("Network (KB/s):", network)