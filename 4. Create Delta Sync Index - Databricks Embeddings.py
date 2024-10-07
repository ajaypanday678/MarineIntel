# Databricks notebook source
# MAGIC %pip install --quiet databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

client = VectorSearchClient()

index = client.create_delta_sync_index(
  endpoint_name="marineintel_vector_search_endpoint",
  source_table_name="genai.rag.shipping_docs",
  index_name="genai.rag.shipping_docs_index",
  pipeline_type="TRIGGERED",
  primary_key="id",
  embedding_source_column="text",
  embedding_model_endpoint_name="databricks-gte-large-en"
)

