import os
import time
import shutil
import asyncio
import random
import dotenv
import json
from typing import List, Dict
from django.shortcuts import render, HttpResponse,redirect
from django.views.decorators.csrf import csrf_protect
from .models import CodeAnalysisRequest
from langchain_text_splitters import Language
from enum import Enum 
from .codebaseio.codebase_io import get_language_info,clone_repo, load_codebase_files, load_documents_from_repo,split_code_into_chunks, loadAndRetrieveEmbeddings,chatWithCode
from .codebaseio.result_formator import format_response
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAI
from asgiref.sync import sync_to_async
from django.contrib.sessions.backends.db import SessionStore
# Create your views here.

dotenv.load_dotenv()

# Available gemini models:
# gemini-1.5-flash-latest
# gemini-1.5-pro-latest
google_api_key = os.getenv("google_api_key")
print(google_api_key)
llm = GoogleGenerativeAI(
    google_api_key=google_api_key, model="gemini-1.5-pro-latest", temperature=0.5
)

################ Start Helper functionalities #######################
supported_languages = [
    Language.CPP,
    Language.GO,
    Language.JAVA,
    Language.KOTLIN,
    Language.JS,
    Language.TS,
    Language.PHP,
    Language.PROTO,
    Language.PYTHON,
    Language.RST,
    Language.RUBY,
    Language.RUST,
    Language.SCALA,
    Language.SWIFT,
    Language.MARKDOWN,
    Language.LATEX,
    Language.HTML,
    Language.CSHARP,
    Language.SOL,
    Language.COBOL,
    Language.LUA,
    Language.HASKELL,
]


available_languages = []
for tag in Language:
    if tag.value in supported_languages:
        available_languages.append(tag)

cust_session:Dict[str,any]={}
cust_session["available_languages"] = available_languages
chat_history:List=[]
question_number:int=0

async def get_suffixes_languages(languages:List)->tuple:
    languages_to_show = [get_language_info(language)[0] for language in languages]
    suffixes = [get_language_info(language)[1] for language in languages]
    return (languages_to_show,suffixes)

################ End Helper functionalities #######################

@csrf_protect
async def GetRepoData(request):
    await sync_to_async(request.session.__setitem__)('session_key', random.randbytes(16).hex())
    print(await sync_to_async(request.session.__getitem__)('session_key'))
    if request.method == 'POST':
        languages:str = request.POST.getlist('languages')
        github_url:str = request.POST.get('giturl')
        cust_session['languages'] = languages
        cust_session['github_url'] = github_url
        cust_session['available_languages'] = available_languages
        root_folder = random.randbytes(16).hex()
        root_folder = os.path.join('repo_path',root_folder)
        try:
            start_time = time.time()
            suffixes_languages = await get_suffixes_languages(languages=languages)
            languages_to_show, suffixes = suffixes_languages[0],suffixes_languages[1]
            # Clone the repository
            await clone_repo(github_url=github_url,local_repo_path=root_folder)
            end_time = time.time()
            clone_time_taken = end_time - start_time
            print(f"Time taken to clone repository: {clone_time_taken:.2f} seconds")
            start_time = time.time()
            codebase_files = await load_codebase_files(root_folder=root_folder,suffixes=suffixes)
            document = await load_documents_from_repo(languages=languages_to_show,suffixes=suffixes,local_repo_path=root_folder)
            texts = await split_code_into_chunks(documents=document, languages=languages_to_show)
            istexts = 'No'
            print(type(texts))
            if len(texts) > 0:
                istexts = 'Yes'
            
            print(istexts)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken to create embeddings: {time_taken:.2f} seconds")
            cust_session['texts'] = texts
            context = {'languages':available_languages,'canchat':True}
            return render(request,"chat_with_code/chat.html",context=context)
        except Exception as e:
            print(f"The language {languages} is not allowed, {e}")
        finally:
            # Clean up the cloned repository directory
            if os.path.exists(root_folder):
                shutil.rmtree(root_folder)
                # print(f"Deleted folder: {root_folder}")
    context = {'languages':cust_session["available_languages"]}
    return render(request,"chat_with_code/chat.html",context=context)



@csrf_protect
async def GenerateResponse(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        texts = cust_session['texts']
        start_time = time.time()
        retriever = await loadAndRetrieveEmbeddings(texts)
        end_time = time.time()
        retriever_time_taken = end_time - start_time
        print(f"Time taken to create retriever: {retriever_time_taken:.2f} seconds")
        start_time = time.time()
        result = await chatWithCode(
                question=question, retriever=retriever,llm=llm, chat_history=chat_history
            )
        end_time = time.time()
        result_time_taken = end_time - start_time
        print(f"Time taken to generate response: {result_time_taken:.2f} seconds")
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["answer"]))
        start_time = time.time()
        formated_result = await format_response(result['answer'])
        end_time = time.time()
        formated_time_taken = end_time - start_time
        print(f"Time taken to format response: {formated_time_taken:.2f} seconds")
        context = {'languages':available_languages,'question':question, 'answer':formated_result,'canchat':True}
        return render(request,"chat_with_code/chat.html",context=context)
    context = {'languages':cust_session["available_languages"],'canchat':False}
    return render(request,"chat_with_code/chat.html",context=context)