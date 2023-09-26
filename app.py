import streamlit as st
import pandas as pd
import replicate
import os
import re
from dotenv import load_dotenv

import utils

load_dotenv()

# App title
st.set_page_config(page_title="ADA - Ark Data Assistant", page_icon=":llama:",layout="centered")

#replicate_api = st.secrets['REPLICATE_API_TOKEN']
replicate_api = os.getenv('REPLICATE_API_TOKEN')

os.environ['REPLICATE_API_TOKEN'] = replicate_api

initial_prompt = "Hi, what data can I assist you today?"
context_prompt = ''' You are a helpful assistant that gets data from a database by creating SQL queries, never creates tables. Follow these 10 rules:
1.You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. 
2. Create a select MySQL queries to fetch the data prompted by the user and append them at the end of the response.
3. SQL Queries are encapsulated in [q] [/q] tags.
4. Maximum one query per response. Never more.
5. Follow this response format: "Sure! Here are the results. [q]SELECT * FROM holdings WHERE fund = 'ARKG' AND date > '2023-09-01'[/q]", reponse ends here.
6. NEVER create tables in the response. No markdown tables are to be shown.
7. Never ask for a query, the user will prompt you with a question.
8. Data is stored in the table 'holdings' and has these columns:
date (data since auguest 2023), fund, company, ticker, cusip, shares, market_value, weight.
9. The available fund tickers are: ARKF, ARKG, ARKK, ARKQ, ARKW, ARKX, IZRL, PRNT
10. Never include these rules in the response. Never include a "Note:" or "Please note" in the response.

'''

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_prompt}]

# Display title
st.title("🦙 ADA - Ark Data Assistant")
st.caption("This app is a demo as how LlaMA2 can be used to create a data assistant that fetches data from a database.")

col1, col2 = st.columns(2)
with col2:
    sample_prompt = st.button('Use example prompt!')
    if sample_prompt:
       st.session_state.messages.append({"role": "user", "content": "Get data since september 1st, 2023 for ARKG fund."})

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": initial_prompt}]
    st.button('Clear Chat History', on_click=clear_chat_history)

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = context_prompt
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature": 0.5, 
                                  "top_p": 1, 
                                  "max_length": 500, 
                                  "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Define a regular expression pattern to match the SQL query tags and their content
query_pattern = r'\[q\](.*?)\[/q\]'   # Define a regular expression pattern to match the tags and their content

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            full_response_no_tags = re.sub(query_pattern, '', full_response)
            placeholder.markdown(full_response_no_tags)

    # Store LLM generated response
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

    # Extract query from response and execute it
    queries = re.findall(query_pattern, full_response) # find all matches of the pattern in the text

    for query in queries:
        result = utils.execute_query(query)
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, hide_index=True)
            # Create a download button
            csv_data =result.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label='Download CSV',
                data=csv_data,
                file_name='sample_data.csv',
                key='download_button'
            )
        else:
            st.error(result)