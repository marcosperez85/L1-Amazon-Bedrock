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

session = boto3.Session(profile_name = "AdministratorAccess-376129873205")

# ### Setup the Bedrock runtime
bedrock_runtime = session.client('bedrock-runtime', region_name='us-east-1')

prompt = "Write a one line description of David Copperfield, the magician."

# Es importante asegurarse que el modelo esté disponible en la región elegida.
# También es necesario habilitar manualmente el modelo en la consola de AWS. El Amazon Titan Text Lite G1 parece ser uno de los más económicos.
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
"""
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())"""

# La variable "response" contiene un JSON con metadatos pero no tiene metadatos pero no tiene el resultado que nos da el LLM.
# Así es como luce: 'body': <botocore.response.StreamingBody at 0x7f282847b3a0>}
# Por ese motivo tenemos que quedarnos con el keys "body" y del cual nos interesa leer (read) el botocore StreamingBody.
# Eso nos da otro JSON con sus propios key:values entre los cuales está el resultado del LLM

# print(json.dumps(response_body, indent=4))
# print(response_body['results'][0]['outputText'])

# Al ejecutar eso vemos que el "response_body" tiene la respuesta del LLM en el key llamado "outputText" por lo que vamos a 
# mostrar en la terminal sólo eso: el outputText del elemento 0 del key "results"
# Veamos ahora cómo podemos experimentar con la creatividad del LLM a través de algunos parámetros.

# ### Generation Configuration

kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}

"""
response2 = bedrock_runtime.invoke_model(**kwargs)
response2_body = json.loads(response2.get('body').read())

# print(json.dumps(response_body, indent=4))
generation = response2_body['results'][0]['outputText']
print(generation)"""

# El parámetro "maxTokenCount" de 100 es para limitar el número de tokens que nos va a devolver el LLM.
# No es el número exacto que vamos a obtener sino el máximo que le permitimos que use en la respuesta.
# Luego "temperatura" regula que tan creativa es la respuesta. El valor de 0.7 es lo típico. Cuanto más bajo, menos creativo es
# y por ese motivo también van a ser más consistentes las respuestas que obtengamos en consultas sucesivas.
# Finalmente "topP" es una forma de restringir el número de opciones que LLM tiene cuando elige el siguiente token. No entra 
# en mas detalles el tutorial acerca de eso así que va a convenir buscarlo en otro lado.

# Nota, si vemos que el key "outputText" quedó cortado en mitad de una oración, probablemente sea porque llegó al límite de
# tokens antes de terminar la generación. Eso se puede ver mostrando en pantalla el response_body completo donde se
# ve el "completionReason". Si llega a decir LENGTH es porque se acabaron los tokens y hace falta aumentarlos o reducir 
# el temperature. Esto último es porque un mayor número de tokens implica un costo mayor por el uso del LLM.


# ### Working with other type of data
# En esta sección el instructor muestra cómo reproducir un archivo de audio pero es sólo eso, muestra cómo reproducirlo 
# sin realizarle ningún procesamiento. Es decir, aun no vamos a ver cómo obtener una transcripción sino que brinda una
# transcripción ya hecha para luego crear un resumen mediante ingeniería de prompt. 



