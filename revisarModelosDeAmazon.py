import boto3
import json

session = boto3.Session(profile_name="AdministratorAccess-376129873205")
bedrock_client = session.client('bedrock', region_name='us-east-1')

# Listar los modelos disponibles
response = bedrock_client.list_foundation_models()
print(json.dumps(response, indent=4))