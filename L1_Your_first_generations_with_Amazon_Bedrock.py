#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - Your first generations with Amazon Bedrock

# Welcome to Lesson 1. 
# 
# You'll start with using Amazon Bedrock to prompt a model and customize how it generates its response.
# 
# **Note:** To access the `requirements.txt` file, go to `File` and click on `Open`. Here, you will also find all helpers functions and datasets used in each lesson.
#  
# I hope you enjoy this course!

# ### Import all needed packages
import boto3
import json
#from IPython.display import Audio

session = boto3.Session(profile_name = "AdministratorAccess-376129873205")

# ### Setup the Bedrock runtime
bedrock_runtime = session.client('bedrock-runtime', region_name='us-east-1')

prompt = "Write a one sentence summary of Las Vegas."

# Es importante asegurarse que el modelo esté disponible en la región elegida.
# También es necesario habilitar manualmente el modelo en la consola de AWS. El Amazon Tital Text Lite G1 parece ser uno de los más económicos.
kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}

# La razón por la que se pone **kwargs con los asteriscos es porque los argumentos de la función "invoke_model" van separados por comas.
# Por lo tanto podríamos poner cada uno de los parámetros (modelId, contentType, etc) en forma individual pero sería difícil de leer.
# En cambio con **kwargs, la función sabe que no se le está pasando un JSON sino que tiene que tomar los elementos que tiene dentro.
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

print(json.dumps(response_body, indent=4))
print(response_body['results'][0]['outputText'])