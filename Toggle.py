import os

# Path to the configuration file
CONFIG_FILE = 'llm_config.txt'

# Function to read the current LLM from the config file
def get_current_llm():
    if not os.path.exists(CONFIG_FILE):
        return None
    
    with open(CONFIG_FILE, 'r') as file:
        return file.read().strip()

# Function to write the new LLM to the config file
def set_current_llm(llm_name):
    with open(CONFIG_FILE, 'w') as file:
        file.write(llm_name)
    print(f"Switched to {llm_name}")

# Function to toggle between GPT-4o and Grok 2
def toggle_llm():
    current_llm = get_current_llm()
    if current_llm == 'GPT-4o':
        set_current_llm('Grok 2')
    elif current_llm == 'Grok 2':
        set_current_llm('GPT-4o')
    else:
        # If no LLM is set or an unknown LLM is set, default to GPT-4o
        set_current_llm('GPT-4o')

if __name__ == "__main__":
    toggle_llm()
