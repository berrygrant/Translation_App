SPA3200_TranslationAppFinal/finetuning.py                                                           0000664 0001751 0001751 00000016162 14777027346 020426  0                                                                                                    ustar   gberry03                        gberry03                                                                                                                                                                                                               import os
import time
import io
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

                                                                                                                                                                                                                                                                                                                                                                                                              SPA3200_TranslationAppFinal/goldens_train.jsonl                                                     0000664 0001751 0001751 00000050033 14777027443 021576  0                                                                                                    ustar   gberry03                        gberry03                                                                                                                                                                                                               {"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El mercado de artesan\u00edas exhibe talentos locales"}, {"role": "assistant", "content": " The craft market showcases local talents"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La pasi\u00f3n por la vida se refleja en cada acci\u00f3n"}, {"role": "assistant", "content": " The passion for life is reflected in every action"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La tradici\u00f3n se transmite de generaci\u00f3n en generaci\u00f3n"}, {"role": "assistant", "content": " Tradition is passed down from generation to generation"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La luna llena ilumina el sendero nocturno"}, {"role": "assistant", "content": " The full moon lights the night path"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El mar embravecido retumba contra las rocas"}, {"role": "assistant", "content": " The raging sea crashes against the rocks"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La danza tradicional celebra la cultura ancestral"}, {"role": "assistant", "content": " Traditional dance celebrates ancestral culture"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Una mente abierta abraza el cambio"}, {"role": "assistant", "content": " An open mind embraces change"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La conversaci\u00f3n fue tan fluida como el agua"}, {"role": "assistant", "content": " The conversation flowed as smoothly as water"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Una sonrisa puede cambiar el d\u00eda de alguien"}, {"role": "assistant", "content": " A smile can change someone's day"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El sol brilla intensamente en el cielo"}, {"role": "assistant", "content": " The sun shines brightly in the sky"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El campo est\u00e1 lleno de flores en primavera"}, {"role": "assistant", "content": " The countryside is full of flowers in spring"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Las monta\u00f1as se alzan majestuosas en el horizonte"}, {"role": "assistant", "content": " The mountains rise majestically on the horizon"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El r\u00edo cristalino refleja el paisaje monta\u00f1oso"}, {"role": "assistant", "content": " The crystal-clear river reflects the mountainous landscape"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Cada estrella en el cielo cuenta una historia"}, {"role": "assistant", "content": " Every star in the sky tells a story"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La ciudad hist\u00f3rica conserva tradiciones milenarias"}, {"role": "assistant", "content": " The historic city preserves millennia-old traditions"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Mis amigos y yo disfrutamos de una tarde de risas"}, {"role": "assistant", "content": " My friends and I enjoy an afternoon of laughter"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La lluvia de ideas gener\u00f3 soluciones creativas"}, {"role": "assistant", "content": " The brainstorming session generated creative solutions"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El viaje en tren revela paisajes inolvidables"}, {"role": "assistant", "content": " The train journey reveals unforgettable landscapes"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La vida es una aventura llena de sorpresas"}, {"role": "assistant", "content": " Life is an adventure full of surprises"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El atardecer ti\u00f1e el cielo de colores c\u00e1lidos"}, {"role": "assistant", "content": " The sunset dyes the sky with warm colors"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La nieve cubre el paisaje con un manto de pureza"}, {"role": "assistant", "content": " The snow covers the landscape with a mantle of purity"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La brisa fresca invita a pasear por el parque"}, {"role": "assistant", "content": " The cool breeze invites a walk in the park"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La lluvia suave acaricia la tierra sedienta"}, {"role": "assistant", "content": " The gentle rain caresses the thirsty earth"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El circo deslumbra con acrobacias impresionantes"}, {"role": "assistant", "content": " The circus dazzles with impressive acrobatics"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El parque est\u00e1 lleno de \u00e1rboles centenarios"}, {"role": "assistant", "content": " The park is full of centuries-old trees"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La ciudad se ilumina con luces brillantes por la noche"}, {"role": "assistant", "content": " The city lights up with bright lights at night"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El viento susurra secretos en el bosque"}, {"role": "assistant", "content": " The wind whispers secrets in the forest"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La arquitectura moderna se mezcla con la hist\u00f3rica"}, {"role": "assistant", "content": " Modern architecture blends with the historic"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El puente conecta dos mundos distintos"}, {"role": "assistant", "content": " The bridge connects two different worlds"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La reuni\u00f3n familiar estuvo llena de cari\u00f1o y recuerdos"}, {"role": "assistant", "content": " The family gathering was filled with love and memories"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Los p\u00e1jaros cantan al amanecer"}, {"role": "assistant", "content": " Birds sing at dawn"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La paz interior se alcanza con la meditaci\u00f3n"}, {"role": "assistant", "content": " Inner peace is achieved through meditation"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El viento de la monta\u00f1a trae historias antiguas"}, {"role": "assistant", "content": " The mountain wind carries ancient stories"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La biblioteca es un templo del conocimiento"}, {"role": "assistant", "content": " The library is a temple of knowledge"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El r\u00edo serpentea suavemente por el valle verde"}, {"role": "assistant", "content": " The river gently winds through the green valley"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La m\u00fasica cl\u00e1sica inspira creatividad"}, {"role": "assistant", "content": " Classical music inspires creativity"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La m\u00fasica de jazz anima las noches en el club"}, {"role": "assistant", "content": " Jazz music livens up the nights at the club"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El poeta encuentra belleza en lo cotidiano"}, {"role": "assistant", "content": " The poet finds beauty in the everyday"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El chef prepara un banquete exquisito"}, {"role": "assistant", "content": " The chef prepares an exquisite feast"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El amor trasciende las fronteras y las culturas"}, {"role": "assistant", "content": " Love transcends borders and cultures"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El viajero busca experiencias que enriquezcan su alma"}, {"role": "assistant", "content": " The traveler seeks experiences that enrich his soul"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El silencio de la noche invita a la reflexi\u00f3n"}, {"role": "assistant", "content": " The silence of the night invites reflection"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Una taza de caf\u00e9 caliente alegra la ma\u00f1ana"}, {"role": "assistant", "content": " A cup of hot coffee brightens the morning"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El bosque encantado guarda misterios sin resolver"}, {"role": "assistant", "content": " The enchanted forest holds unsolved mysteries"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Los ni\u00f1os juegan alegremente en el parque"}, {"role": "assistant", "content": " Children play happily in the park"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El aroma del pan reci\u00e9n horneado es irresistible"}, {"role": "assistant", "content": " The aroma of freshly baked bread is irresistible"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El escultor utiliza el m\u00e1rmol para expresar emociones"}, {"role": "assistant", "content": " The sculptor uses marble to express emotions"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El oto\u00f1o pinta los \u00e1rboles con tonos dorados"}, {"role": "assistant", "content": " Autumn paints the trees with golden hues"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La carrera de bicicletas une a deportistas de todas partes"}, {"role": "assistant", "content": " The bike race brings together athletes from everywhere"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El arte refleja la esencia de la humanidad"}, {"role": "assistant", "content": " Art reflects the essence of humanity"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El tiempo vuela cuando uno se divierte"}, {"role": "assistant", "content": " Time flies when you're having fun"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Estudiar todos los d\u00edas mejora la memoria"}, {"role": "assistant", "content": " Studying every day improves memory"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Una buena conversaci\u00f3n alimenta el alma"}, {"role": "assistant", "content": " A good conversation nourishes the soul"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El mercado bulle de actividad y color"}, {"role": "assistant", "content": " The market buzzes with activity and color"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El libro que le\u00ed anoche fue fascinante"}, {"role": "assistant", "content": " The book I read last night was fascinating"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La plaza central es el coraz\u00f3n de la ciudad"}, {"role": "assistant", "content": " The central square is the heart of the city"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El r\u00edo serpentea a trav\u00e9s del valle"}, {"role": "assistant", "content": " The river winds through the valley"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La playa se extiende con arena dorada y suave"}, {"role": "assistant", "content": " The beach stretches out with soft, golden sand"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La tecnolog\u00eda transforma nuestras vidas a diario"}, {"role": "assistant", "content": " Technology transforms our lives daily"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El sonido de la cascada es relajante y constante"}, {"role": "assistant", "content": " The sound of the waterfall is soothing and steady"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El festival de luces transforma la ciudad en un espect\u00e1culo"}, {"role": "assistant", "content": " The festival of lights transforms the city into a spectacle"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El legado cultural se celebra con festivales y tradiciones"}, {"role": "assistant", "content": " The cultural legacy is celebrated with festivals and traditions"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El puente colgante desaf\u00eda la gravedad con elegancia"}, {"role": "assistant", "content": " The suspension bridge defies gravity with elegance"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La flor del cerezo es s\u00edmbolo de belleza ef\u00edmera"}, {"role": "assistant", "content": " The cherry blossom is a symbol of fleeting beauty"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El zorro astuto se desliza entre los arbustos"}, {"role": "assistant", "content": " The cunning fox slips through the bushes"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El reloj de la torre marca el ritmo de la ciudad"}, {"role": "assistant", "content": " The clock tower keeps the rhythm of the city"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El escultor da vida a la piedra con su arte"}, {"role": "assistant", "content": " The sculptor brings stone to life with his art"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La catedral g\u00f3tica impresiona por su arquitectura"}, {"role": "assistant", "content": " The Gothic cathedral impresses with its architecture"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El aroma de las especias llena la cocina con calidez"}, {"role": "assistant", "content": " The aroma of spices fills the kitchen with warmth"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El desayuno abundante es el combustible del d\u00eda"}, {"role": "assistant", "content": " A hearty breakfast is the fuel for the day"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El violinista emociona con cada nota que toca"}, {"role": "assistant", "content": " The violinist moves the audience with every note played"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La novela narra una historia de amor y aventura"}, {"role": "assistant", "content": " The novel tells a story of love and adventure"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La sinfon\u00eda de la naturaleza es m\u00fasica para el alma"}, {"role": "assistant", "content": " The symphony of nature is music for the soul"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La fiesta en la plaza une a toda la comunidad"}, {"role": "assistant", "content": " The festival in the square brings the whole community together"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La serenata al atardecer llen\u00f3 el aire de romanticismo"}, {"role": "assistant", "content": " The serenade at sunset filled the air with romance"}]}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     SPA3200_TranslationAppFinal/goldens_val.jsonl                                                       0000664 0001751 0001751 00000015443 14777027443 021251  0                                                                                                    ustar   gberry03                        gberry03                                                                                                                                                                                                               {"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El jard\u00edn bot\u00e1nico alberga especies de todo el mundo"}, {"role": "assistant", "content": " The botanical garden hosts species from all over the world"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El teatro ofrece obras que desaf\u00edan la imaginaci\u00f3n"}, {"role": "assistant", "content": " The theater offers plays that challenge the imagination"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La m\u00fasica de guitarra evoca emociones profundas"}, {"role": "assistant", "content": " The guitar music evokes deep emotions"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La feria anual re\u00fane a personas de todos los rincones"}, {"role": "assistant", "content": " The annual fair brings together people from all corners"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El pintor retrata la vida con pinceladas de pasi\u00f3n"}, {"role": "assistant", "content": " The painter portrays life with strokes of passion"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El palacio antiguo cuenta historias de reyes y reinas"}, {"role": "assistant", "content": " The ancient palace tells tales of kings and queens"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La luna nueva inspira nuevos comienzos"}, {"role": "assistant", "content": " The new moon inspires fresh beginnings"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El pintor plasma sus sue\u00f1os en un lienzo"}, {"role": "assistant", "content": " The painter expresses his dreams on a canvas"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El caf\u00e9 de la ma\u00f1ana despierta los sentidos"}, {"role": "assistant", "content": " Morning coffee awakens the senses"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La brisa marina refresca el ambiente"}, {"role": "assistant", "content": " The sea breeze refreshes the atmosphere"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El faro gu\u00eda a los barcos en la oscuridad"}, {"role": "assistant", "content": " The lighthouse guides ships in the darkness"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Las estrellas titilan en la oscuridad de la noche"}, {"role": "assistant", "content": " The stars twinkle in the darkness of the night"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El perro duerme pl\u00e1cidamente en el jard\u00edn"}, {"role": "assistant", "content": " The dog sleeps peacefully in the garden"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El jard\u00edn florece con colores vibrantes en verano"}, {"role": "assistant", "content": " The garden blooms with vibrant colors in summer"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El festival gastron\u00f3mico celebra la diversidad culinaria"}, {"role": "assistant", "content": " The food festival celebrates culinary diversity"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El campo de lavanda emana un aroma embriagador"}, {"role": "assistant", "content": " The lavender field exudes an intoxicating aroma"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La costumbre de dar la bienvenida se celebra con calidez"}, {"role": "assistant", "content": " The tradition of welcoming is celebrated with warmth"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La vista desde la cima es simplemente espectacular"}, {"role": "assistant", "content": " The view from the top is simply spectacular"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La aurora boreal pinta el cielo de colores m\u00edsticos"}, {"role": "assistant", "content": " The aurora borealis paints the sky with mystical colors"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El mar guarda secretos de tiempos pasados"}, {"role": "assistant", "content": " The sea holds secrets of bygone eras"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La biblioteca digital facilita el acceso al conocimiento"}, {"role": "assistant", "content": " The digital library makes accessing knowledge easier"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: El volc\u00e1n en erupci\u00f3n ilumina el cielo nocturno"}, {"role": "assistant", "content": " The erupting volcano lights up the night sky"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: Cada d\u00eda es una nueva oportunidad para aprender"}, {"role": "assistant", "content": " Every day is a new opportunity to learn"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La experiencia en el museo es educativa y fascinante"}, {"role": "assistant", "content": " The museum experience is educational and fascinating"}]}
{"messages": [{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": "Translate Spanish: La primavera renueva la esperanza en el coraz\u00f3n"}, {"role": "assistant", "content": " Spring renews hope in the heart"}]}
                                                                                                                                                                                                                             SPA3200_TranslationAppFinal/metrics.py                                                              0000664 0001751 0001751 00000004333 14773545470 017722  0                                                                                                    ustar   gberry03                        gberry03                                                                                                                                                                                                               # metrics.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

smooth = SmoothingFunction().method1

def compute_grammaticality(sentence):
    verbs = ["is", "are", "was", "were", "has", "have", "do", "does", "did",
             "go", "went", "see", "saw", "says", "said", "run", "runs"]
    score = 0.0
    if sentence and sentence[0].isupper():
        score += 0.3
    if sentence and sentence[-1] in ".!?":
        score += 0.3
    words = sentence.lower().split()
    if any(verb in words for verb in verbs):
        score += 0.4
    return min(score, 1.0)

def compute_naturalness(sentence):
    stopwords = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "and", "but", "or", "so"}
    words = sentence.lower().split()
    if not words:
        return 0.0
    stopword_count = sum(1 for w in words if w in stopwords)
    ratio = stopword_count / len(words)
    if 0.3 <= ratio <= 0.5:
        return 1.0
    else:
        return max(0.0, 1 - abs(ratio - 0.4) * 2)

def compute_style(translated, expected):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([translated, expected])
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return cosine_sim

def compute_precision(candidate_tokens, reference_tokens):
    candidate_set = set(candidate_tokens)
    reference_set = set(reference_tokens)
    if len(candidate_set) == 0:
        return 0.0
    return len(candidate_set.intersection(reference_set)) / len(candidate_set)

def compute_recall(candidate_tokens, reference_tokens):
    candidate_set = set(candidate_tokens)
    reference_set = set(reference_tokens)
    if len(reference_set) == 0:
        return 0.0
    return len(candidate_set.intersection(reference_set)) / len(reference_set)

def compute_f1(candidate_tokens, reference_tokens):
    prec = compute_precision(candidate_tokens, reference_tokens)
    rec = compute_recall(candidate_tokens, reference_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_exact_accuracy(candidate, reference):
    return 1.0 if candidate.strip() == reference.strip() else 0.0
                                                                                                                                                                                                                                                                                                     SPA3200_TranslationAppFinal/run_app.py                                                              0000664 0001751 0001751 00000021717 14777025535 017724  0                                                                                                    ustar   gberry03                        gberry03                                                                                                                                                                                                               import os
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
                                                 SPA3200_TranslationAppFinal/translation.py                                                          0000664 0001751 0001751 00000001404 14777034223 020577  0                                                                                                    ustar   gberry03                        gberry03                                                                                                                                                                                                               # translation.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def openai_translate(text, model="gpt-4o-mini-2024-07-18"):
    prompt = f"Translate the following Spanish text to English:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        translation = response.choices[0].message.content.strip()
        translation = translation.strip('"').strip("'")
        return translation
    except Exception as e:
        return f"Error in translation: {str(e)}"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            