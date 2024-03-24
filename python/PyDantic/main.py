from pydantic import BaseModel
from typing import Literal
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class WorkflowSchema(BaseModel):
    funcName: str = ''

class FunctionContainerConfigSchema(BaseModel):
    funcName: str
    # one of the providers
    provider: Literal['local', 'aliyun','knative','aws','local-once']
    workflow: WorkflowSchema

def get_function_container_config():
    # Read the env from the environment
    env = os.environ

    config : FunctionContainerConfigSchema = FunctionContainerConfigSchema(
        funcName=env['FAASIT_FUNC_NAME'],
        provider=env['FAASIT_PROVIDER'],
        workflow=WorkflowSchema(
            funcName=env.get('FAASIT_WORKFLOW_FUNC_NAME') or ''
        )
    )
    return config.dict()

config = get_function_container_config()
print(config.dict()['provider'])