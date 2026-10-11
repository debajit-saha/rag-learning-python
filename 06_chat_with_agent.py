
import os
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv()

PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
AGENT_NAME = os.environ["AGENT_NAME"]

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

client = project.get_openai_client(
    agent_name=AGENT_NAME
)

print(f"Connected to agent: {AGENT_NAME}")
print("Ask questions about the employee handbook.")
print("Type 'exit' to quit.\n")


previous_response_id = None

print("Chat started. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    try:
        request_args = {
            "input": question,
        }

        if previous_response_id is not None:
            request_args["previous_response_id"] = (
                previous_response_id
            )

        response = client.responses.create(
            **request_args
        )

        print(f"\nAgent: {response.output_text}\n")

        previous_response_id = response.id

    except Exception as error:
        print(f"\nRequest failed: {error}\n")
