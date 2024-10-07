# Databricks notebook source
# MAGIC %pip install --quiet mlflow

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

# COMMAND ----------

endpointname = "marinintel"

# COMMAND ----------



endpoint = client.create_endpoint(
  name=f"{endpointname}",
  config={
    "served_entities": [
        {
            "entity_name": f"genai.rag.marinintel_chatbot_model",
            "entity_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": "marinintel_chatbot_model-1",
                "traffic_percentage": 100
            }
        ]
    }
  }
)
