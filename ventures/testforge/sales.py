from __future__ import annotations

import argparse

from sales.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="TestForge sales department commands")
    parser.add_argument("command", choices=["demo"])
    args = parser.parse_args()

    if args.command == "demo":
        result = run_demo()
        print("TESTFORGE SALES DEMO")
        print(f"Mock leads: {result['mock_leads']}")
        print(f"Leads qualified: {result['leads_qualified']}")
        print(f"Drafts created: {result['drafts_created']}")
        print(result["status"])


if __name__ == "__main__":
    main()
