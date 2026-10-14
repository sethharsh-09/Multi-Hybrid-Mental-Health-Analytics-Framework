"""
MindGraphNet - Social Network & Interaction Feature Extraction
Quantifies network reciprocity, user activity levels, and thread depth.
"""
import pandas as pd
import numpy as np

class InteractionFeatureExtractor:
    @staticmethod
    def extract_graph_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Replies received by each post
        reply_counts = df["replies_to"].value_counts().to_dict()
        df["replies_received"] = df["post_id"].map(reply_counts).fillna(0.0).astype(float)
        
        # 2. Author historical activity
        author_counts = df["author_hash"].value_counts().to_dict()
        df["author_activity"] = df["author_hash"].map(author_counts).fillna(1.0).astype(float)
        
        # 3. Thread depth approximation
        depth_map = {}
        # Simple iterative parent lookup
        id_to_parent = dict(zip(df["post_id"], df["replies_to"]))
        for pid in df["post_id"]:
            depth = 0
            cur = pid
            visited = set()
            while cur in id_to_parent and pd.notna(id_to_parent[cur]) and cur not in visited:
                visited.add(cur)
                cur = id_to_parent[cur]
                depth += 1
                if depth > 10:
                    break
            depth_map[pid] = float(depth)
            
        df["thread_depth"] = df["post_id"].map(depth_map).fillna(0.0)
        return df
