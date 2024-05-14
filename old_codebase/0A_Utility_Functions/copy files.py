# Databricks notebook source
import os

source_path = "file:/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions"
destination_path = f"file:/Workspace{os.path.split(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())[0]}"
for files in ["utils.py","cluster.py","predict.py"]:
    dbutils.fs.cp(f"{source_path}/{files}",f"{destination_path}/{files}")
    dbutils.fs.rm(f"{destination_path}/.{files}.crc")

source_path = "file:/Workspace/Users/davide@baxter.com/Solution"
destination_path = f"file:/Workspace{os.path.split(os.path.split(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())[0])[0]}"
for files in ["config.json","stats.json"]:
    dbutils.fs.cp(f"{source_path}/{files}",f"{destination_path}/{files}")
    dbutils.fs.rm(f"{destination_path}/.{files}.crc")