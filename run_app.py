import os
import streamlit as st
import pandas as pd
from openai import OpenAI
import httpx
from evaluation import (
    evaluate_translation_comparison,
    run_optimization
)
from translation import openai_translate
from finetuning import fine_tune_openai, get_recent_finetuning_jobs

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SPA 3200 Translation App", page_icon=":speech_balloon:")

# ----------------------------
# Session State Initialization
# ----------------------------
for key, default in [
    ('goldens', None),
    ('quality_offset', 0.0),
    ('translation_mode', "OpenAI API"),
    ('fine_tuned_model_id', None),
    ('fine_tuning_hyperparams', {}),
    ('selected_job_id', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("SPA 3200 Translation App")
st.caption("(Based on Chat-GPT 4o-mini)")

# ----------------------------
# Upload Goldens
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Goldens (CSV with columns 'Spanish' and 'English')",
    type=["csv", "json", "txt"]
)
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if not {"Spanish","English"}.issubset(df.columns):
            st.error("CSV must have 'Spanish' and 'English' columns")
        else:
            st.session_state.goldens = df
            st.success("Goldens uploaded successfully")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ----------------------------
# Translation Mode
# ----------------------------
mode = st.radio(
    "Select Translation Mode",
    ["OpenAI API", "Fine-tuned Model"],
    index=0
)
st.session_state.translation_mode = mode

# ----------------------------
# Fine-tuned Model Selection & Metrics
# ----------------------------
if mode == "Fine-tuned Model":
    choice = st.radio(
        "Select fine-tuned model source",
        ["Recent fine-tuning runs", "Specify manually"],
        index=0
    )
    if choice == "Recent fine-tuning runs":
        jobs = get_recent_finetuning_jobs()
        if jobs:
            prefix = "ft:gpt-4o-mini-2024-07-18:personal::"
            options = []
            for job in jobs:
                mid = getattr(job, "fine_tuned_model", job.fine_tuned_model)
                stripped = mid[len(prefix):] if mid.startswith(prefix) else mid
                disp = f"{prefix}{stripped}"
                jid  = job.id
                options.append((disp, jid))
            sel_disp, sel_jid = st.selectbox(
                "Select a fine-tuned model",
                options,
                format_func=lambda x: x[0]
            )
            st.session_state.fine_tuned_model_id = sel_disp
            st.session_state.selected_job_id      = sel_jid
        else:
            st.warning("No recent fine-tuning runs found")
    else:
        manual = st.text_input("Enter fine-tuned model ID manually:")
        if manual:
            st.session_state.fine_tuned_model_id = manual
            st.session_state.selected_job_id      = None

    # --- LIST checkpoint retrieval only (no single GET) ---
    if st.session_state.selected_job_id:
        try:
            job_id = st.session_state.selected_job_id
            ckpts_page = client.fine_tuning.jobs.checkpoints.list(job_id)
            ckpts = ckpts_page.data
            if not ckpts:
                st.info("No checkpoints found for this job.")
            else:
                latest = ckpts[-1]
                m = latest.metrics or {}
                st.markdown("#### Selected Fine-tuned Model Metrics:")
                st.write(f"**Step:** {latest.step_number}")
                st.write(f"**Train Loss:** {getattr(m, 'train_loss', 'NA')}")
                st.write(f"**Train Mean Token Accuracy:** {getattr(m, 'train_mean_token_accuracy', 'NA')}")
                st.write(f"**Validation Loss:** {getattr(m, 'valid_loss', 'NA')}")
                st.write(f"**Validation Mean Token Accuracy:** {getattr(m, 'valid_mean_token_accuracy', 'NA')}")
                st.write(f"**Full Validation Loss:** {getattr(m, 'full_valid_loss', 'NA')}")
                st.write(f"**Full Validation Mean Token Accuracy:** {getattr(m, 'full_valid_mean_token_accuracy', 'NA')}")
        except Exception as e:
            st.error(f"Error retrieving checkpoint metrics: {e}")
    # -------------------------------------------------------

# ----------------------------
# Translation Section
# ----------------------------
with st.form("translation_form"):
    text = st.text_input("Enter Spanish text to translate:")
    go   = st.form_submit_button("Translate")
    if go and text:
        model = (
            st.session_state.fine_tuned_model_id
            if mode == "Fine-tuned Model"
            else "gpt-4o-mini-2024-07-18"
        )
        trans = openai_translate(text, model=model)
        st.write("**Translation:**", trans)

# ----------------------------
# Evaluate Goldens
# ----------------------------
if st.button("Evaluate Goldens"):
    if st.session_state.goldens is None:
        st.error("Please upload golden utterances first")
    else:
        use_ft = (mode == "Fine-tuned Model")
        if use_ft:
            base = evaluate_translation_comparison(
                st.session_state.goldens,
                model_to_use="gpt-4o-mini-2024-07-18",
                hyperparams={"batch_size":"NA","n_epochs":"NA"}
            )
            fine = evaluate_translation_comparison(
                st.session_state.goldens,
                model_to_use=st.session_state.fine_tuned_model_id,
                hyperparams=st.session_state.fine_tuning_hyperparams
            )
            combined = []
            for b, f in zip(base, fine):
                combined.append({
                    **f,
                    "Delta Meaning Retention (BLEU)": round(f["Meaning Retention (BLEU)"] - b["Meaning Retention (BLEU)"], 2),
                    "Delta Grammaticality": round(f["Grammaticality"] - b["Grammaticality"], 2),
                    "Delta Naturalness": round(f["Naturalness"] - b["Naturalness"], 2),
                    "Delta Style/Register": round(f["Style/Register"] - b["Style/Register"], 2)
                })
            df_res = pd.DataFrame(combined)
        else:
            df_res = pd.DataFrame(evaluate_translation_comparison(
                st.session_state.goldens,
                model_to_use="gpt-4o-mini-2024-07-18",
                hyperparams={"batch_size":"NA","n_epochs":"NA"}
            ))
        df_res = df_res.drop(columns=["batch_size","n_epochs"], errors="ignore")
        st.dataframe(df_res)

# ----------------------------
# Optimize (Simulated)
# ----------------------------
with st.expander("Optimize (Simulated)"):
    st.write("**Hyperparameters Description:**")
    st.write("- **Learning Rate:** Step size in optimization.")
    st.write("- **Batch Size:** Stability of improvements.")
    st.write("- **Epochs:** Number of passes over goldens.")
    with st.form("optimization_form"):
        sim_lr   = st.slider("Learning Rate", 1e-5, 1e-3, 1e-4, 1e-5)
        sim_bs   = st.selectbox("Batch Size", [8,16,32,64])
        sim_ep   = st.number_input("Epochs", 1, 10, 3)
        run_opt  = st.form_submit_button("Run Optimization")
        if run_opt:
            noffset = run_optimization(
                st.session_state.goldens,
                sim_lr, sim_bs, sim_ep,
                st.session_state.quality_offset
            )
            st.session_state.quality_offset = noffset
            st.success(f"Optimization complete. Offset={noffset:.3f}")

# ----------------------------
# Fine-Tune Section
# ----------------------------
with st.expander("Fine-Tune Model"):
    st.write("Adjust hyperparameters and run fine-tuning:")
    with st.form("fine_tuning_form"):
        ft_lr         = st.slider("Learning Rate", 1e-5, 1e-3, 1e-4, 1e-5)
        ft_batch_size = st.selectbox("Batch Size", [8,16,32,64])
        ft_epochs     = st.number_input("Epochs", 1, 10, 3)
        run_ft        = st.form_submit_button("Run Fine-Tuning")
        if run_ft:
            if st.session_state.goldens is None:
                st.error("Upload goldens first")
            else:
                with st.spinner("Fine-tuning..."):
                    mid, jm = fine_tune_openai(
                        st.session_state.goldens,
                        ft_epochs, ft_lr, ft_batch_size
                    )
                    if mid:
                        st.session_state.fine_tuned_model_id      = mid
                        st.session_state.fine_tuning_hyperparams = jm
                st.success(f"Fine-tuning done. Model ID: {mid}")

# ----------------------------
# Explanatory Notes
# ----------------------------
st.markdown("""
### Validation Metrics Explanation

- **BLEU:** Meaning retention score.
- **Grammaticality:** Heuristic for correct grammar.
- **Naturalness:** Fluency via stopword ratio.
- **Style/Register:** TF–IDF cosine similarity.
- **Token-level Metrics:** Precision, Recall, F1 on tokens.
- **Delta Metrics:** Improvement over base model.
""")
