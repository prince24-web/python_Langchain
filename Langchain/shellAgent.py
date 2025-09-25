# -*- coding: utf-8 -*-
import os
import getpass
import shutil
import subprocess
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
import argparse

# -----------------------------
# 1. Wrap shell commands
# -----------------------------
def run_shell(command: str) -> str:
    """Run any shell command and return output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)

def make_dir(name: str) -> str:
    """Create a new folder."""
    return run_shell(f"mkdir {name}")

def list_dir(path: str = ".") -> str:
    """List files and folders in the directory."""
    return run_shell(f"dir {path}")

def read_file(path: str) -> str:
    """Read the contents of a file."""
    return run_shell(f"type {path}")

def delete_file(path: str) -> str:
    """Delete a file."""
    return run_shell(f"del {path}")

def delete_folder_recursive(folder_name: str, start_path: str = ".") -> str:
    """
    Search upwards from start_path for a folder with folder_name
    and delete it if found.
    """
    current_path = os.path.abspath(start_path)

    while True:
        target = os.path.join(current_path, folder_name)
        if os.path.exists(target) and os.path.isdir(target):
            try:
                shutil.rmtree(target)  # removes folder + all contents
                return f"Deleted folder: {target}"
            except Exception as e:
                return f"Error deleting {target}: {str(e)}"

        # move up one directory
        parent = os.path.dirname(current_path)
        if parent == current_path:  # reached root
            break
        current_path = parent

    return f"Folder '{folder_name}' not found anywhere upwards from {os.path.abspath(start_path)}"

def gitcommit(message: str) -> str:
    """Check git status, add all changes, commit, and push."""
    return run_shell(
        f"git status && git add . && git commit -m \"{message}\" && git push && git status"
    )

# -----------------------------
# 2. Expose functions as tools
# -----------------------------
@tool("run_shell")
def run_shell_tool(command: str) -> str:
    """Run a Windows shell command."""
    return run_shell(command)

@tool("make_dir")
def make_dir_tool(name: str) -> str:
    """Create a folder."""
    return make_dir(name)

@tool("list_dir")
def list_dir_tool(path: str = ".") -> str:
    """List directory contents."""
    return list_dir(path)

@tool("read_file")
def read_file_tool(path: str) -> str:
    """Read file contents."""
    return read_file(path)

@tool("delete_file")
def delete_file_tool(path: str) -> str:
    """Delete a single FILE in the current directory only (not folders)."""
    return delete_file(path)

@tool("delete_folder_recursive")
def delete_folder_recursive_tool(folder_name: str) -> str:
    """Delete a FOLDER by name. If not in current directory, search parent directories until found."""
    return delete_folder_recursive(folder_name)

@tool("git_commit")
def git_commit_tool(message: str) -> str:
    """Stage all changes, commit with a message, and push to remote repo."""
    return gitcommit(message)
#----------------------
# 3. Load Gemini API Key
# -----------------------------
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# -----------------------------
# 4. Create the Agent
# -----------------------------
tools = [
        run_shell_tool,
        make_dir_tool,
        list_dir_tool,
        read_file_tool,
        delete_file_tool,
        delete_folder_recursive_tool,
        git_commit_tool
        ]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# -----------------------------
# 5. Test the Agent
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='*', help='Prompt to send to Gemini')
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else input("Enter your prompt: ")
    print(agent.run(prompt))