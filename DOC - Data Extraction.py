# Databricks notebook source
# MAGIC %pip install --quiet llama-index docx2txt
# MAGIC %restart_python

# COMMAND ----------

from llama_index.core import SimpleDirectoryReader

# COMMAND ----------

import os
from pyspark.sql.functions import substring_index

# Directory path
pdf_volume_path = "/Volumes/genai/rag/raw_docs/shipping/"

# List files in directory
file_paths = [file.path for file in dbutils.fs.ls(pdf_volume_path)]

# Extract file names from paths
df = spark.createDataFrame(file_paths, "string").select(substring_index("value", "/", -1).alias("file_name"))

# Show dataframe
df.show()

# COMMAND ----------

import os

# Get the list of already processed PDF files from the Delta table
processed_files = spark.sql(f"SELECT DISTINCT file_name FROM genai.rag.shipping_docs_track").collect()
processed_files = set(row["file_name"] for row in processed_files)

# Process only new PDF files
new_files = [file for file in os.listdir(pdf_volume_path) if file not in processed_files]
print(len(new_files))

if len(new_files) == 0:
    dbutils.notebook.exit("No new files to process")

dbutils.fs.mkdirs(f'{pdf_volume_path}temp_dir')


for file_name in new_files:
    # Extract text from the PDF file
    pdf_path = os.path.join(pdf_volume_path, file_name)
    dbutils.fs.cp(pdf_path, f'{pdf_volume_path}/temp_dir',True)

# load documents
documents = SimpleDirectoryReader(f'{pdf_volume_path}/temp_dir').load_data()
print(f"Total documents: {len(documents)}")

chunks =[]

for i in documents:
    chunks.append(i.get_content())

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType
import pandas as pd

@pandas_udf("array<string>")
def get_chunks(dummy):
    return pd.Series([chunks])

# Register the UDF
spark.udf.register("get_chunks_udf", get_chunks)

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into genai.rag.shipping_docs (text)
# MAGIC select explode(get_chunks_udf('dummy')) as text;

# COMMAND ----------

df.createOrReplaceTempView("temp_table")  # Create a temporary table from the DataFrame

# Insert only the rows that do not exist in the target table
spark.sql("""
    INSERT INTO genai.rag.shipping_docs_track
    SELECT * FROM temp_table
    WHERE NOT EXISTS (
        SELECT 1 FROM genai.rag.shipping_docs_track
        WHERE temp_table.file_name = genai.rag.shipping_docs_track.file_name
    )
""")

# COMMAND ----------

dbutils.fs.rm(f'{pdf_volume_path}/temp_dir',True)
