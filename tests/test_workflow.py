import os
import sys
import unittest

# Ensure the project root is on sys.path so that package imports resolve
# correctly whether this file is run directly (python tests/test_workflow.py)
# or via the unittest runner from the project root
# (python -m unittest tests.test_workflow).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MCP.client import normalize_incident
from MCP.server import _normalize_status
from Agent.AzureAIInterface import _foundry_agent_api_version, _split_endpoint_query


class IncidentNormalizationTests(unittest.TestCase):
    def test_aliases_are_normalized(self) -> None:
        result = normalize_incident(
            {
                "Incident Number": " INC-1 ",
                "Timestamp": "2026-07-01 10:00:00",
                "Hardware": "Router",
                "Software": "SAP",
                "description": "Login failed",
                "Priority": "High",
            }
        )
        self.assertEqual(result["incident_number"], "INC-1")
        self.assertEqual(result["issue"], "Login failed")
        self.assertEqual(result["sw_system"], "SAP")

    def test_required_fields_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            normalize_incident({"incident_number": "INC-1"})


class StatusNormalizationTests(unittest.TestCase):
    def test_ui_terminal_labels_use_database_canonical_values(self) -> None:
        self.assertEqual(_normalize_status("Close"), ("Closed", "Closed", True))
        self.assertEqual(_normalize_status("Cancel"), ("Cancelled", "Cancelled", True))

    def test_nonterminal_status(self) -> None:
        self.assertEqual(_normalize_status("WIP"), ("WIP", "WIP", False))

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_status("Resolved-ish")


class FoundryEndpointTests(unittest.TestCase):
    def test_responses_endpoint_is_normalized_to_openai_protocol_base(self) -> None:
        base_url, query = _split_endpoint_query(
            "https://example.services.ai.azure.com/api/projects/p/agents/a/"
            "endpoint/protocols/openai/responses?api-version=v1"
        )
        self.assertEqual(
            base_url,
            "https://example.services.ai.azure.com/api/projects/p/agents/a/"
            "endpoint/protocols/openai",
        )
        self.assertEqual(query["api-version"], "v1")

    def test_foundry_agent_api_version_defaults_to_v1(self) -> None:
        self.assertEqual(_foundry_agent_api_version(), "v1")


if __name__ == "__main__":
    unittest.main()
