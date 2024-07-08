'''Source:https://python.langchain.com/v0.1/docs/use_cases/code_understanding/'''
import os
import dotenv
import asyncio
import glob
import git
import validators
import logging
from typing import List, Dict
from git import Repo
from urllib.parse import urlparse
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

dotenv.load_dotenv()

# Available gemini models:
# gemini-1.5-flash-latest
# gemini-1.5-pro-latest
# gemini-1.5-pro
google_api_key = os.getenv("google_api_key")
llm = GoogleGenerativeAI(
    google_api_key=google_api_key, model="gemini-1.5-pro-latest", temperature=0.7
)

def get_language_info(language: str)->List[str]:
    Language_dict = {
        "cpp": [Language.CPP, ".cpp"],
        "go": [Language.GO, ".go"],
        "java": [Language.JAVA, ".java"],
        "kotlin": [Language.KOTLIN, ".kt"],
        "js": [Language.JS, ".js"],
        "ts": [Language.TS, ".ts"],
        "php": [Language.PHP, ".php"],
        "proto": [Language.PROTO, ".proto"],
        "python": [Language.PYTHON, ".py"],
        "rst": [Language.RST, ".rst"],
        "ruby": [Language.RUBY, ".rb"],
        "rust": [Language.RUST, ".rs"],
        "scala": [Language.SCALA, ".scala"],
        "swift": [Language.SWIFT, ".swift"],
        "markdown": [Language.MARKDOWN, ".md"],
        "latex": [Language.LATEX, ".tex"],
        "html": [Language.HTML, ".html"],
        "sol": [Language.SOL, ".sol"],
        "csharp": [Language.CSHARP, ".cs"],
        "cobol": [Language.COBOL, ".cbl"],
        "c": [Language.C, ".c"],
        "lua": [Language.LUA, ".lua"],
        "perl": [Language.PERL, ".pl"],
        "haskell": [Language.HASKELL, ".hs"]
    }
    try:
        return Language_dict[language]
    except KeyError:
        raise ValueError(f"Language {language} is not supported! Please choose from {list(Language_dict.keys())}")



# e.g. url = "https://github.com/my-org/my-repo" or "https://github.com/owner/repo"
def extract_repo_from_url(url: str) -> str:
    try:
        # Parse the URL
        parsed_url = urlparse(url)
        # Ensure the URL is from GitHub
        if parsed_url.netloc.lower() != "github.com":
            raise ValueError("URL is not a GitHub repository URL")
        # Split the path and strip leading/trailing slashes
        path_parts = parsed_url.path.strip("/").split("/")
        # Ensure the path contains at least two parts: owner and repo
        if len(path_parts) < 2:
            raise ValueError("URL path does not contain a valid repository")
        owner, repo = path_parts[0], path_parts[1]
        # Create a platform-independent path
        return os.path.join(owner, repo)
    except Exception as e:
        return BaseException(str(e))


async def load_codebase_files(
    root_folder: str, suffixes: List[str]
) -> Dict[str, str]:
    """
    Loads code files from a specified directory.

    This function recursively searches the `root_folder` for files with the specified
    `suffixes`. It returns a dictionary where keys are the full file paths and values
    are the file names.

    Args:
        root_folder (str): The root directory to search for code files.
        suffix (List[str], optional): A tuple of file extensions to include.

    Returns:
        Dict: A dictionary mapping file paths to file names.

    Example:
        >>> load_codebase_files(root_folder="my_project", suffixes=[".py",])
        {'/path/to/my_project/main.py': 'main.py',
        '/path/to/my_project/utils/helper.py': 'helper.py'}
    """
    code_files: Dict = {}
    if not os.path.isdir(root_folder):
        raise ValueError(f"Invalid root_folder: '{root_folder}' is not a directory.")
    try:
        for ext in suffixes:
            for file_path in glob.glob(
                os.path.join(root_folder, f"**/*{ext}"), recursive=True
            ):
                root = os.path.dirname(file_path)  # Get the directory of the file
                file = os.path.basename(file_path)
                code_files[os.path.join(root, file)] = file  # Use os.path.join
    except FileNotFoundError:
        raise BaseException(f"Directory not found: {root_folder}")
    except PermissionError:
        raise BaseException(f"Permission denied to access: {root_folder}")
    return code_files


# e.g. url = "https://github.com/my-org/my-repo" or "https://github.com/owner/repo"
async def clone_repo(github_url: str, local_repo_path: str = "repo_path") -> None:
    """
    Clones a GitHub repository to a local directory.

    Args:
        github_url (str): The URL of the GitHub repository.
        local_repo_path (str, optional): The path to the local directory where the repository will be cloned.
        Defaults to "repo_path".
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(local_repo_path):
        os.makedirs(local_repo_path)
    if github_url != "":
        if not validators.url(github_url):
            raise f"Invalid {github_url}. Please enter a valid URL."
        try:
            owner_folder = extract_repo_from_url(github_url)
            repo_path = os.path.join(local_repo_path, owner_folder)
            Repo.clone_from(
                github_url,
                to_path=repo_path,
            )
        except git.exc.GitCommandError as e:
            raise RuntimeError(f"Error cloning repository: {e}")
        except OSError as e:
            raise RuntimeError(f"File system error: {e}")

async def load_documents_from_repo(
    languages: List[str],  # Accept a list of languages
    suffixes: List[str],
    local_repo_path: str = "repo_path",
    specific_file_path: str = "",
    exclude: List[str] = [],
) -> List:
    """
    Loads code files from a cloned repository.

    Args:
        languages (List[Language]): A list of programming languages to load.
        local_repo_path (str, optional): The path to the cloned repository. Defaults to "repo_path".
        specific_file_path (str, optional): A specific path within the repository to load files from.
        Defaults to "", which loads from the root of the repository.
        suffixes (List[str], optional): A list of file suffixes to include. If None, defaults to language-specific suffixes.
        exclude (List[str], optional): A list of file or directory names to exclude.

    Returns:
        List: A list of Document objects representing the loaded code files.
    """
    all_documents = []

    for language in languages:
        # if suffixes is None:
        #     suffixes = get_language_info(language)[1]  # Default to language-specific suffix
        try:
            loader = GenericLoader.from_filesystem(
                local_repo_path + specific_file_path,
                glob="**/*",
                suffixes=suffixes,
                exclude=exclude,
                parser=LanguageParser(language=language.value, parser_threshold=500),
            )
            documents = loader.load()
            all_documents.extend(documents)
        except FileNotFoundError:
            raise BaseException(
                f"Error: Directory not found: {local_repo_path + specific_file_path}"
            )
        except PermissionError:
            raise BaseException(
                f"Error: Permission denied to access: {local_repo_path + specific_file_path}"
            )
        except Exception as e:
            # BaseException
            raise BaseException(
                f"The language {language.value} is not supported. {e}"
            )
    return all_documents


async def split_code_into_chunks(documents: List, languages: List[str]) -> List:
    """
    Splits code documents into smaller chunks for embedding and analysis.

    Args:
        documents (List[Document]): A list of Document objects representing code files.
        languages (List[Language]): A list of programming languages to consider for splitting.

    Returns:
        List: A list of text chunks derived from the code documents.
    """
    all_texts = []
    for language in languages:
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=2000,
                chunk_overlap=200,  # chunk_size=2000
            )
            texts = splitter.split_documents(documents=documents)
            all_texts.extend(texts)
        except:
            raise (
                f"The language {language.value} is not supported."
            )
    return all_texts


async def loadAndRetrieveEmbeddings(texts: List):
    """
    Creates embeddings for a list of text documents and sets up a retriever.

    Args:
        texts (List): A list of text documents to embed.

    Returns:
        Any: A retriever object that can be used to search for relevant documents.
        Returns None if there's an error during embedding creation.
    """
    try:
        db = Chroma.from_documents(
            texts,
            GoogleGenerativeAIEmbeddings(
                google_api_key=google_api_key, model="models/embedding-001"
            ),
        )
        retriever = db.as_retriever(
            search_type="mmr",  # Also test "similarity" mmr = (Maximal Marginal Relevance)
            search_kwargs={"k": 7},
        )
        return retriever
    except (ValueError, Exception) as e:
        if "API key" in str(e):
            raise BaseException(f"Error: Invalid Google API key: {e}")
        elif "model" in str(e):
            raise BaseException(f"Error: Invalid embedding model specified: {e}")
        else:
            raise BaseException(f"Error creating embeddings: {e}")


async def chatWithCode(
    question: str,
    retriever: loadAndRetrieveEmbeddings,
    llm=llm,
    chat_history: List = [],
) -> Dict[str, str]:
    """
    Executes a question-answering session using a retriever and a language model.

    Args:
        question (str): The user's question.
        retriever: The retriever object used to find relevant documents.
        llm: The language model used to generate responses.
        chat_history (List, optional): A list of previous conversation turns. Defaults to [].

    Returns:
        Dict: A dictionary containing the answer generated by the language model.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation and the context of the provided GitHub repository, generate a search query to look up information relevant to the conversation.",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # "Answer the user's questions based on the below context:\n\n{context}",
                "You are an expert programming assistant. Answer the user's questions based on the context of the provided GitHub repository and its codebase:\n\n{context}\n\nDo not provide responses unrelated to the repository or general knowledge not contained in the repository. If the user's question cannot be answered based on the repository, politely inform them that the information is not available.Ensure all code snippets are formatted using the correct language identifier in Markdown code blocks. \nFor example:- C++: `cpp`\n- Python: `python`\n- JavaScript: `javascript`\n- Java: `java`\n- Go: `go`",
                # "You are an expert programming mentor. Your role is to guide the user, providing detailed answers, suggestions, and improvements based on the context of the provided GitHub repository and its codebase. Context:\n\n{context}\n\n Please ensure your responses are strictly based on the repository's content. Offer mentorship, suggestions, and improvements as appropriate. Do not include information unrelated to the repository or general knowledge not contained within it. If the user's question cannot be answered based on the repository, kindly inform them that the information is not available."
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ],
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa = create_retrieval_chain(retriever_chain, document_chain)
    # try:
    result = qa.invoke({"input": question, "chat_history": chat_history})
    return result
    # except:
    #     raise f"Error generating response from LLM:"
    # except Exception as e:
    #     raise BaseException(f"Error generating response from LLM: {e}")


if __name__ == "__main__":
    # Use a library like rich for better UI
    import time
    import textwrap
    from rich import print
    from rich.panel import Panel
    from rich.prompt import Prompt
    from IPython.display import Markdown
    
    def to_markdown(text):
        text = text.replace("•", " *")
        return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

    def main() -> None:
        github_url: str = input("Enter GitHub URL here: ")

        languages = [Language.PYTHON]
        suffixes = [get_language_info(language)[1] for language in languages]
        print(suffixes)
        asyncio.run(clone_repo(github_url=github_url))
        codebase_files = asyncio.run(load_codebase_files(root_folder="repo_path",suffixes=suffixes))
        documents = asyncio.run(load_documents_from_repo(languages=languages,suffixes=suffixes))
        texts = asyncio.run(split_code_into_chunks(documents=documents, languages=languages))
        retriever = asyncio.run(loadAndRetrieveEmbeddings(texts))
        chat_history: List = []  # add in session
        question_number = 0
        print(Panel("Available Code Files", title="CodePilot", expand=False))
        for file_path, file_name in codebase_files.items():
            print(f"- {file_name} ({file_path})")
        print(
            Panel("Welcome to the Codebase Chatbot!", title="CodePilot", expand=False)
        )
        while True:
            question: str = Prompt.ask(
                "[bold blue]Ask question here[/] or type [bold red]exit[/] or [bold red]clear[/] to end chat"
            )
            if question.lower() == "exit" or question.lower() == "clear":
                break
            question_number += 1
            print(
                Panel(
                    f"[bold green]Question>>{question_number}[/]\n{question}",
                    title=":question: User Question",
                    expand=False,
                )
            )
            # Record start time
            start_time = time.time()
            result = chatWithCode(
                question=question, retriever=retriever, chat_history=chat_history
            )

            # Calculate time taken
            end_time = time.time()
            time_taken = end_time - start_time
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=result["answer"]))
            print(
                Panel(
                    f"[bold magenta]Answer[/]\n{result['answer']}\n\n[bold magenta]Time taken:[/] {time_taken:.2f} seconds",
                    title="🤖 Codebase Assistant",
                    expand=False,
                )
            )
            if question_number == 11:
                chat_history = []
                question_number = 0
    main()