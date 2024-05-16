# Databricks notebook source
from packages.utils import utils
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import gradio as gr

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]
cutoff_date = config["cutoff_date"]["value"]

stats = utils.read_json("./stats.json")

customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"
before_cutoff_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"
recommendation_data_path = f"DATA_VERSIONS/{data_version}/RECOMMENDATION_DATA"
propensity_data_path = f"DATA_VERSIONS/{data_version}/PROPENSITY_DATA"
prediction_data_path = f"DATA_VERSIONS/{data_version}/PREDICTION_DATA"
after_cutoff_path = f"DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

customers = utils.read_parquet(f"{customer_data_path}/customer_details.parquet")
CPL = utils.read_parquet(f"{before_cutoff_path}/curated_products_data.parquet")
AFTER_CUTOFF_CSD = utils.read_parquet(f'{after_cutoff_path}/customer_sales_data.parquet')

recommendation_model = utils.read_pkl(f"{recommendation_data_path}/recommendation_model.pkl")
propensity_model = utils.read_pkl(f"{propensity_data_path}/propensity_model.pkl")
prediction_model = utils.read_pkl(f"{prediction_data_path}/prediction_model.pkl")

# COMMAND ----------

product_list = sorted(CPL["updated"].unique())
customers_list = sorted(customers["customer_num"].unique())

# COMMAND ----------

def get_name(customer_num):
    if customer_num in list(customers["customer_num"]):
        return customers[customers["customer_num"] == customer_num][
            "customer_name"
        ].values[0]
    else:
        return "No name found in record"


def get_recommendation(customer_num, top, remark, replacement, curated=True):
    name = get_name(customer_num)
    output = recommendation_model.recommend_with_remark(customer_num, top, remark, replacement, curated)
    if not remark:
        output = output.drop(["Remark","History"],axis=1)
    output["Rating"] = (output["Rating"]*10000).astype("int")/10000
    return name, output


def get_propensity(customer_num, srp_2_list):
    name = get_name(customer_num)
    output1 = propensity_model.predict_from_lists_with_remark([customer_num], [srp_2_list])
    remark = output1["Remark"].values[0]
    history = output1["History"].values[0]
    win_prob = int(100*prediction_model.predict_from_lists([customer_num], [srp_2_list],probability=True)['outcome'].values[0])/100
    if win_prob<0.5:
        remark = ""
    return name,remark,history,{"Win Probability" :win_prob}


def location_plot(locations):
    if ("Any" in locations) or (locations==[]):
        data = ALS_test
    else:
        data = ALS_test[ALS_test["location"].isin(locations)]
    data.rename(columns={"reciprocal_rank":"mrr"},inplace=True)
    fig1 = px.histogram(
        data,
        x="rank",
        nbins=len(set(ALS_test["rank"])),
        color="location"
    )
    fig1.update_layout(
        autosize=False,
        width=600,
        height=300,
        xaxis_title="Rank",
        yaxis_title="No. of sales",
        legend_title="Location"
    )
    data2 = data.groupby("location")[["mrr"]].mean().sort_values("mrr",ascending=False).reset_index()
    fig2 = px.bar(data2,x="location",y="mrr",color="location")
    fig2.update_layout(
    autosize=False,
    width=600,
    height=300,
    xaxis_title="Location",
    yaxis_title="MRR",
    legend_title="Location"
    )
    mean = round(data["mrr"].mean(),4)
    return mean,fig1,fig2

# COMMAND ----------

ALS_test = recommendation_model.mrr_score(AFTER_CUTOFF_CSD)

# COMMAND ----------

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

blockPrint()

with gr.Blocks(theme=gr.themes.Default()) as demo:
    with gr.Tab("Recommendation Model"):
        with gr.Row():
            with gr.Column():
                rm_customer_num = gr.Textbox(label="Customer number", value="606053")
                rm_top = gr.Slider(1, 20, step=1, value=10, label="No. of recommendations")
                with gr.Row():
                    rm_remark = gr.Checkbox(label="Remark", value=True)
                    rm_replacement = gr.Checkbox(label="Replacement", value=True)
                    # rm_curated = gr.Checkbox(label="Curated", value=True)
                with gr.Row():
                    recommend_button = gr.Button("Recommend")
                with gr.Row():
                    gr.Markdown()
            with gr.Column():
                rm_customer_name = gr.Textbox(label="Customer name")
                recommend_output = gr.DataFrame(interactive=False, label="Recommendation")
        rm_inputs = [rm_customer_num, rm_top, rm_remark, rm_replacement]
        rm_outputs = [rm_customer_name,recommend_output]
        recommend_button.click(fn=get_recommendation, inputs=rm_inputs, outputs=rm_outputs)
    with gr.Tab("Recommendation Stats"):
        with gr.Row():
            gr.Markdown(f"Train Size : **{stats['prediction train shape']['value'][0]}**")
            gr.Markdown(f"Test Size : **{len(ALS_test)}**")
            gr.Markdown(f"Cut-off Date : **{cutoff_date}**")
        with gr.Row():
            gr.Markdown(f"Recommendable Customers : **{stats['recommendable customers']['value']}**")
            gr.Markdown(f"Recommendable Clusters : **{stats['recommendable clusters']['value']}**")
            gr.Markdown(f"Total Clusters : **{stats['customer clustering n clusters']['value']}**")
        with gr.Row():
            rs_location = gr.Dropdown(
                ["Any"] + sorted(ALS_test["location"].unique()),
                value="Any",
                label="Location",
                multiselect=True,
            )
            rs_button = gr.Button("Get Metrics")
        with gr.Row():
            mrr_location = gr.Textbox(label="Mean Reciprocal Rank (MRR)")
        with gr.Row():
            rank_plot = gr.Plot(label="Rank Histogram",show_label=False)
            mrr_plot = gr.Plot(label="Location wise MRR",show_label=False)
        rs_inputs = [rs_location]
        rs_outputs = [mrr_location,rank_plot,mrr_plot]
        rs_button.click(fn=location_plot, inputs=rs_inputs, outputs=rs_outputs)
    with gr.Tab("Propensity Model"):
        with gr.Row():
            pr_customer_num = gr.Textbox(label="Customer number", value="606053")
            pr_srp_2_desc = gr.Dropdown(product_list, label="Product name", value="PROGRESSA", multiselect=False)
        with gr.Row():
            propensity_button = gr.Button("Predict")
        with gr.Row():
            pm_customer_name = gr.Textbox(label="Customer name")
            remark = gr.Textbox(label="Remark")
            probability = gr.Label(label="Probability")
            history = gr.Textbox(label="History")
        pr_inputs = [pr_customer_num, pr_srp_2_desc]
        pr_outputs = [pm_customer_name,remark,history,probability]
        propensity_button.click(fn=get_propensity, inputs=pr_inputs, outputs=pr_outputs)
    with gr.Tab("Propensity Stats"):
        with gr.Row():
            gr.Markdown(f"Train Size : **{(stats['propensity train shape']['value'][0])}**")
            gr.Markdown(f"Test Size : **{(stats['propensity test shape']['value'][0])}**")
            gr.Markdown(f"Cutoff Date : **{cutoff_date}**")
        with gr.Row():
            gr.Markdown(f"Precision : **{round(stats['propensity report']['value']['0']['precision'],2)}**")
            gr.Markdown(f"Recall : **{round(stats['propensity report']['value']['0']['recall'],2)}**")
            gr.Markdown(f"F1 Score : **{round(stats['propensity report']['value']['0']['f1-score'],2)}**")
        with gr.Row():
            confusion_plot = px.imshow(stats['propensity confusion']["value"],text_auto=True,color_continuous_scale='Viridis')
            confusion_plot.update_layout(autosize=False,width=1220,height=300,xaxis_showticklabels=False,yaxis_showticklabels=False)
            gr.Plot(label="Confusion Matrix",value=confusion_plot,show_label=False)

demo.launch(share=True,height=800)