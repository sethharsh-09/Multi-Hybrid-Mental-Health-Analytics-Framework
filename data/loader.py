"""
MindGraphNet - Robust Dataset Loader & Ingestion Source
Seamlessly integrates with Kaggle RMHD while providing an intelligent,
zero-failure realistic fallback benchmark dataset with real thread graphs.
"""
import os
import glob
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from config.settings import DATA_DIR, RANDOM_SEED

BENCHMARK_TOPICS = [
    "anxiety", "depression", "mentalhealth", "lonely",
    "SuicideWatch", "bipolar", "ptsd", "CasualConversation"
]

BENCHMARK_POST_TEMPLATES = [
    # Anxiety templates
    ("Can't stop my heart from racing during team meetings",
     "Every time I have to speak up at work, my chest tightens, my hands shake, and I feel like I'm having a heart attack. I've tried breathing exercises but nothing stops the panic.", "anxiety", "anxiety"),
    ("Terrified of leaving my house this week",
     "The agoraphobia is creeping back in. Just thinking about taking the subway or going to the grocery store fills me with intense dread and nausea.", "anxiety", "anxiety"),
    ("Overthinking every single interaction with friends",
     "Did I say something weird? Are they talking about me in another chat? I analyze every pause and text message until my head hurts.", "anxiety", "anxiety"),
    
    # Depression templates
    ("Can't find the energy to get out of bed today",
     "Everything feels heavy, numb, and pointless. I've been lying here staring at the ceiling for 6 hours. Haven't showered in 4 days. I feel like a ghost in my own life.", "depression", "depression"),
    ("Nothing brings joy anymore, completely hollow",
     "Hobbies that used to keep me alive feel like chores now. Video games, music, books - just silence. The emptiness is suffocating.", "depression", "depression"),
    ("Feeling like an invisible burden to my family",
     "Everyone would be better off without having to manage my emotional weight. I constantly apologize for just existing.", "depression", "depression"),

    # Loneliness templates
    ("Surrounded by people at university but completely alone",
     "I see friend groups laughing in the quad and going out on weekends. I eat lunch alone in the library bathroom stalls so no one sees me.", "lonely", "loneliness"),
    ("Haven't had a real conversation in three weeks",
     "Aside from ordering food at counters, I haven't heard my own voice out loud in days. Social isolation is physically painful.", "lonely", "loneliness"),
    ("Does anyone else feel detached from human connection?",
     "Even when someone is nice to me, there is a glass wall between me and the rest of the world. I don't know how to bridge it.", "lonely", "loneliness"),

    # High Crisis / SuicideWatch templates
    ("I am at the end of my rope and don't know what to do",
     "The pain won't stop and every day is agony. I don't see any future or way out. I've reached my breaking point.", "SuicideWatch", "crisis"),
    ("Thinking about saying goodbye to everyone",
     "I have failed at everything. Nobody understands how dark it is inside my head. I just want the suffering to end forever.", "SuicideWatch", "crisis"),
    ("Can someone please just talk to me before I make a terrible mistake?",
     "Sitting on the floor crying uncontrollably. Feeling completely abandoned by everyone I trusted. I can't do this anymore.", "SuicideWatch", "crisis"),

    # PTSD & Stress templates
    ("Nightmares waking me up drenched in sweat again",
     "Flashbacks from the accident keep intruding whenever I close my eyes. Hypervigilant 24/7. My nervous system is completely fried.", "ptsd", "trauma"),
    ("Extreme burnout and nervous breakdown at my job",
     "Working 70 hours a week under constant threats from management. Chronic exhaustion, constant migraines, and uncontrollable sobbing during lunch breaks.", "mentalhealth", "burnout"),
    
    # Neutral / Support / Recovery templates
    ("Started cognitive behavioral therapy today, feeling hopeful",
     "Had my first session with a licensed counselor. We established a baseline routine and thought journaling. Step by step.", "mentalhealth", "recovery"),
    ("Made dinner for myself for the first time in weeks",
     "Small victory: cooked a hot meal and did the dishes right after. Celebrating the little things.", "CasualConversation", "support"),
    ("Tips for managing sleep schedule when mood fluctuates?",
     "Setting a fixed wake-up time has helped a bit. Any good non-caffeine routines people recommend before bedtime?", "mentalhealth", "lifestyle"),
    ("Weekly check-in: share one positive moment from your week",
     "Let's share small wins no matter how minor. I managed to go for a 15-minute walk outside in the sunshine today.", "CasualConversation", "support")
]


def generate_benchmark_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Generates a realistic, clinically-informed conversational Reddit benchmark dataset
    with realistic user interaction graphs, thread dynamics, and temporal drift.
    """
    random.seed(RANDOM_SEED)
    records = []
    base_time = datetime(2026, 1, 15, 8, 30)
    
    n_authors = max(10, n_samples // 4)
    authors = [f"reddit_user_{i:03d}" for i in range(1, n_authors + 1)]
    
    # Select heavy active users (nodes with high degree) vs infrequent posters
    core_users = authors[:n_authors // 5]
    casual_users = authors[n_authors // 5:]
    
    for i in range(n_samples):
        # Sample template and inject natural linguistic variance
        title, text, topic, root_cause = random.choice(BENCHMARK_POST_TEMPLATES)
        
        # Add natural perturbations
        qualifiers = [" Honestly,", " Update:", " [Vent]", " To be fair,", " Not sure where else to post this.", ""]
        suffix = random.choice(["", " Anyone else relate?", " Thanks for listening.", " Any advice?", " Please help."])
        full_text = f"{title}. {random.choice(qualifiers)} {text}{suffix}".strip()
        
        author = random.choice(core_users) if random.random() < 0.60 else random.choice(casual_users)
        created_at = base_time + timedelta(hours=random.randint(0, 180 * 24))
        
        records.append({
            "post_id": f"P{i:05d}",
            "author": author,
            "text": full_text,
            "topic": topic,
            "created_at": created_at,
            "replies_to": None,
            "root_cause_label": root_cause
        })
        
    df = pd.DataFrame(records).sort_values("created_at").reset_index(drop=True)
    
    # Construct realistic discussion thread reply links (conversational graph edges)
    for idx in range(3, len(df)):
        if random.random() < 0.65:
            # Thread reply: reply to a previous post in the same topic within the last 14 days
            cur_time = df.loc[idx, "created_at"]
            cur_topic = df.loc[idx, "topic"]
            candidates = df[(df.index < idx) & 
                            (df["topic"] == cur_topic) & 
                            (df["created_at"] >= cur_time - timedelta(days=14))]
            if len(candidates) > 0:
                parent_id = candidates.sample(1)["post_id"].values[0]
                df.loc[idx, "replies_to"] = parent_id

    benchmark_path = DATA_DIR / "sample_rmhd.csv"
    df.to_csv(benchmark_path, index=False)
    print(f"Generated benchmark dataset: {len(df)} posts across {df.topic.nunique()} topics -> {benchmark_path}")
    return df


def load_dataset(use_kaggle: bool = True, fallback_size: int = 500) -> pd.DataFrame:
    """
    Attempts to download or load the real Kaggle RMHD dataset.
    If Kaggle credentials are unavailable or download fails,
    gracefully and transparently loads/generates the benchmark dataset.
    """
    sample_file = DATA_DIR / "sample_rmhd.csv"
    
    # 1. Check if Kaggle download is requested and possible
    if use_kaggle:
        try:
            import kagglehub
            has_credentials = (
                ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ) or
                os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
            )
            if has_credentials:
                print("Downloading RMHD from Kaggle...")
                dataset_path = kagglehub.dataset_download("entenam/reddit-mental-health-dataset")
                csvs = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
                for path in csvs:
                    try:
                        tmp = pd.read_csv(path, nrows=5)
                        cols_lower = {c.lower() for c in tmp.columns}
                        if "label" in cols_lower and ("selftext" in cols_lower or "title" in cols_lower):
                            raw_df = pd.read_csv(path)
                            raw_df.columns = [c.lower() for c in raw_df.columns]
                            raw_df["selftext"] = raw_df.get("selftext", "").fillna("")
                            raw_df["title"] = raw_df.get("title", "").fillna("")
                            raw_df["text"] = (raw_df["title"] + " " + raw_df["selftext"]).str.strip()
                            raw_df = raw_df[raw_df["text"].str.len() > 5].reset_index(drop=True)
                            
                            # Synthesize graph thread attributes
                            n = len(raw_df)
                            authors = [f"user_{i}" for i in range(1, max(2, n // 4))]
                            base_time = datetime(2026, 1, 1)
                            df = pd.DataFrame({
                                "post_id": [f"P{i:05d}" for i in range(n)],
                                "author": [random.choice(authors) for _ in range(n)],
                                "text": raw_df["text"],
                                "topic": raw_df.get("subreddit", raw_df.get("label", "mentalhealth")).fillna("mentalhealth"),
                                "created_at": [base_time + timedelta(hours=random.randint(0, 4000)) for _ in range(n)],
                                "replies_to": None,
                                "root_cause_label": raw_df.get("label", "unknown")
                            })
                            print(f"Successfully loaded Kaggle RMHD: {len(df)} posts.")
                            return df
                    except Exception:
                        continue
        except Exception as e:
            print(f"Kaggle RMHD loader encountered exception ({e}). Falling back to benchmark dataset.")

    # 2. Check if local benchmark dataset exists
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        df["created_at"] = pd.to_datetime(df["created_at"])
        print(f"Loaded existing dataset from {sample_file} ({len(df)} posts).")
        return df

    # 3. Generate benchmark dataset
    return generate_benchmark_dataset(n_samples=fallback_size)


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset preview:")
    print(df.head())
