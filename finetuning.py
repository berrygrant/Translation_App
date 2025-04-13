import os
import time
import io
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI"]["api_key"])

def create_finetune_files(goldens: pd.DataFrame):
    """
    Split goldens into 75/25 train/validation and write JSONL files
    in the Chat format required for supervised fine‑tuning.
    """
    shuffled = goldens.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(shuffled) * 0.75)
    train_df = shuffled.iloc[:split]
    val_df   = shuffled.iloc[split:]

    def write_jsonl(df, filename):
        with open(filename, "w") as f:
            for _, row in df.iterrows():
                record = {
                    "messages": [
                        {"role": "system",    "content": "You are a helpful translation assistant."},
                        {"role": "user",      "content": f"Translate Spanish: {row['Spanish'].strip()}"},
                        {"role": "assistant", "content": f" {row['English'].strip()}"}
                    ]
                }
                f.write(json.dumps(record) + "\n")
        return filename

    train_file = write_jsonl(train_df, "goldens_train.jsonl")
    val_file   = write_jsonl(val_df,   "goldens_val.jsonl")
    return train_file, val_file

def fine_tune_openai(goldens: pd.DataFrame, epochs: int, lr: float, batch_size: int):
    """
    Performs supervised fine‑tuning:
      1. Upload train/val JSONL
      2. Create fine‑tune job
      3. Poll until completion
      4. Retrieve result_files (compiled_results.csv)
      5. Download CSV with client.files.content and parse with pandas
      6. Return model_id and last‐row metrics dict
    """
    # 1) Prepare and upload files
    train_file, val_file = create_finetune_files(goldens)
    st.info("Uploading training data…")
    train_resp = client.files.create(file=open(train_file, "rb"), purpose="fine-tune")
    st.info("Uploading validation data…")
    val_resp   = client.files.create(file=open(val_file,   "rb"), purpose="fine-tune")

    # 2) Start fine‑tuning job
    st.info("Initiating fine‑tuning job…")
    ft = client.fine_tuning.jobs.create(
        training_file=train_resp.id,
        validation_file=val_resp.id,
        model="gpt-4o-mini-2024-07-18",
        method={
            "type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": epochs, "batch_size": batch_size}}
        }
    )
    job_id = ft.id
    st.write(f"Fine‑tuning job started. Job ID: {job_id}")

    # 3) Poll until done
    status_ph = st.empty()
    prog_bar  = st.progress(0)
    status    = ft.status
    progress  = 0
    while status not in ("succeeded", "failed"):
        time.sleep(10)
        ft = client.fine_tuning.jobs.retrieve(job_id)
        if ft.status != status:
            status = ft.status
            status_ph.text(f"Status: {status}")
        progress = min(100, progress + 10)
        prog_bar.progress(progress)

    if status != "succeeded":
        st.error("Fine‑tuning failed")
        return None, None

    # 4) Retrieve result_files (compiled_results.csv)
    ft = client.fine_tuning.jobs.retrieve(job_id)
    files = ft.result_files or []
    if not files:
        st.error("No result file found.")
        return ft.fine_tuned_model, {}

    result_file_id = files[0]

    # 5) Poll until the CSV is processed
    file_meta = client.files.retrieve(result_file_id)
    while file_meta.status != "processed":
        time.sleep(5)
        file_meta = client.files.retrieve(result_file_id)

    # Download and parse the CSV
    content = client.files.content(file_id=result_file_id)
    csv_bytes = content.read()
    df_metrics = pd.read_csv(io.BytesIO(csv_bytes))

        # 6) Download and parse the CSV
    content = client.files.content(file_id=result_file_id)
    csv_bytes = content.read()
    df_metrics = pd.read_csv(io.BytesIO(csv_bytes))

    # If CSV is empty, fall back to list-checkpoints metrics
    if df_metrics.empty:
        st.warning("Metrics CSV empty; falling back to checkpoint list metrics.")
        ckpts_page = client.fine_tuning.jobs.checkpoints.list(job_id)
        ckpts = ckpts_page.data
        if ckpts:
            latest = ckpts[-1]
            m = latest.metrics or type("M", (), {})()
            job_metrics = {
                "step":                            getattr(latest, "step_number", "NA"),
                "train_loss":                      getattr(m, "train_loss", "NA"),
                "train_accuracy":                  getattr(m, "train_mean_token_accuracy", "NA"),
                "valid_loss":                      getattr(m, "valid_loss", "NA"),
                "valid_mean_token_accuracy":       getattr(m, "valid_mean_token_accuracy", "NA"),
                "full_valid_loss":                 getattr(m, "full_valid_loss", "NA"),
                "full_valid_mean_token_accuracy":  getattr(m, "full_valid_mean_token_accuracy", "NA")
            }
        else:
            st.error("No checkpoint metrics available; unable to provide metrics.")
            return ft.fine_tuned_model, {}
    else:
        # Extract the last row of metrics
        last = df_metrics.iloc[-1]
        job_metrics = {
            "step":                            int(last.get("step",  "NA")),
            "train_loss":                      float(last.get("train_loss",  "NA")),
            "train_accuracy":                  float(last.get("train_accuracy",  "NA")),
            "valid_loss":                      float(last.get("valid_loss",  "NA")),
            "valid_mean_token_accuracy":       float(last.get("valid_mean_token_accuracy",  "NA")),
            "full_valid_loss":                 float(last.get("full_valid_loss",  "NA")),
            "full_valid_mean_token_accuracy":  float(last.get("full_valid_mean_token_accuracy",  "NA"))
        }

    st.success(f"Fine‑tuning complete. Model ID: {ft.fine_tuned_model}")
    return ft.fine_tuned_model, job_metrics

    # Extract the last row of metrics
    last = df_metrics.iloc[-1]
    job_metrics = {
        "step":                            int(last.get("step",  "NA")),
        "train_loss":                      float(last.get("train_loss",  "NA")),
        "train_accuracy":                  float(last.get("train_accuracy",  "NA")),
        "valid_loss":                      float(last.get("valid_loss",  "NA")),
        "valid_mean_token_accuracy":       float(last.get("valid_mean_token_accuracy",  "NA")),
        "full_valid_loss":                 float(last.get("full_valid_loss",  "NA")),
        "full_valid_mean_token_accuracy":  float(last.get("full_valid_mean_token_accuracy",  "NA"))
    }

    st.success(f"Fine‑tuning complete. Model ID: {ft.fine_tuned_model}")
    return ft.fine_tuned_model, job_metrics

def get_recent_finetuning_jobs():
    """
    Returns up to 3 most recent succeeded fine‑tuning jobs.
    """
    try:
        jobs = client.fine_tuning.jobs.list().data
        done = [j for j in jobs if j.status == "succeeded"]
        done.sort(key=lambda j: getattr(j, "created_at", 0), reverse=True)
        return done[:3]
    except Exception as e:
        st.error(f"Error retrieving fine‑tuning jobs: {e}")
        return []

