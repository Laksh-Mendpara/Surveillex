import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from secret_key import API_tog as API_KEY

# Set TogetherAI API key
os.environ["TOGETHER_API_KEY"] = API_KEY


# Function to create a prompt and call the TogetherAI model
def generate_expanded_definition(name, definition):
    prompt = f"{definition} is the definition of the {name}. Please rewrite and expand this definition to make it more detailed and consistent with scientific fact. Briefness is required, using only one paragraph."

    # Initialize the TogetherAI model
    model = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    # Send the prompt to the model
    messages = [
        SystemMessage(content="Please rewrite and expand the given definition."),
        HumanMessage(content=prompt),
    ]

    # Invoke the model and return the response
    response = model.invoke(messages)
    return response



if __name__ == "__main__":
    name = "Abuse"
    definition = "cruel or inhumane treatment"
    output = generate_expanded_definition(name, definition)

    print("Generated expanded definition:")
    print(output)
