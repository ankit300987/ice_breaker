from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from third_parties.linkedin import scrape_linkedin_profile
import time
import tqdm


if __name__ == "__main__":
    load_dotenv()

    print("Setting up contextual information...")

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    print("Setting up models ")
    llm = ChatOllama(model="llama3")
    print("Setting up chains...")
    chain = summary_prompt_template | llm
    print("Fetching linkedin profiles..")
    # Create a progress bar
    pbar = tqdm.tqdm(total=1, desc="Loading...")



    start_time = time.time()
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/",
        mock=True,
    )
    end_time = time.time()
    # Update the progress bar
    pbar.update(1)
    pbar.close()
    print(f"LinkedIn data was fetched in {end_time - start_time:.2f} seconds")
    print("Profiles fetched...")
    print("Querying to model...")
    # Create a spinner
    pbar = tqdm.tqdm(total=1, desc="Loading...")
    start_time = time.time()
    res = chain.invoke(input={"information": linkedin_data})
    end_time = time.time()
    # Update the progress bar
    pbar.update(1)
    pbar.close()
    print("Result:")
    print(res)
    print(f"LLM generated data in {end_time - start_time:.2f} seconds")