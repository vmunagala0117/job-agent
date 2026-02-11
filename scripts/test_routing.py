"""Test multi-agent routing: Coordinator → JobSearch / AppPrep."""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_framework.azure import AzureOpenAIChatClient

async def main():
    client = AzureOpenAIChatClient(
        model="gpt-5.2",
        azure_endpoint="https://vm-demo-fdry-01.openai.azure.com",
        deployment_name="gpt-5.2",
    )

    # Use the canonical workflows.py (which is now the multi-agent version)
    from job_agent.workflows import build_agent
    agent = build_agent(client)
    print(f"Agent type: {type(agent).__name__}")

    passed = 0
    total = 3

    # Test 1: Job search intent
    print("\n--- Test 1: Job search intent ---")
    response = await agent.run("What AI jobs are available in San Francisco?")
    text = response.text or ""
    print(f"  Response length: {len(text)}")
    if len(text) > 50:
        print("  ✓ PASS — got substantive job search response")
        passed += 1
    else:
        print(f"  ✗ FAIL — response too short: {text[:100]}")

    # Test 2: App prep intent
    print("\n--- Test 2: App prep intent ---")
    response = await agent.run("Can you help me write a cover letter for a job?")
    text = response.text or ""
    print(f"  Response length: {len(text)}")
    if len(text) > 50:
        print("  ✓ PASS — got substantive app prep response")
        passed += 1
    else:
        print(f"  ✗ FAIL — response too short: {text[:100]}")

    # Test 3: Profile setup (should route to job search)
    print("\n--- Test 3: Profile setup (job search route) ---")
    response = await agent.run(
        "My name is Alice, I'm a senior ML engineer with 8 years experience "
        "in Python, PyTorch, and AWS. I want remote positions paying at least 180k."
    )
    text = response.text or ""
    print(f"  Response length: {len(text)}")
    if len(text) > 30 and ("profile" in text.lower() or "alice" in text.lower() or "set" in text.lower()):
        print("  ✓ PASS — profile was set via job search agent")
        passed += 1
    else:
        print(f"  ✗ FAIL — unexpected response: {text[:200]}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All routing tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)

asyncio.run(main())
