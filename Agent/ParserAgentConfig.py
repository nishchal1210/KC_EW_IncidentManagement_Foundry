Config = {
    "AgentDefinition": (
        "Extract exactly one incident from raw text and return JSON only."
    ),
    "Query": (
        "Extract incident_number, timestamp, hw_system, sw_system, issue, "
        "source, and severity."
    ),
    "Guidance": (
        "Use keys incident_number, timestamp, hw_system, sw_system, issue, "
        "source, severity. Format timestamps as YYYY-MM-DD HH:mm:ss. Use an "
        "empty string for missing optional values. Do not include prose or "
        "markdown fences."
    ),
}
