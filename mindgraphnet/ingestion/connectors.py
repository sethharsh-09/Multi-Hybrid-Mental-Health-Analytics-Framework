"""
MindGraphNet - Cloud Ingestion Connectors (PRAW & Streaming Stubs)
"""
from typing import List, Dict, Any, Optional

class RedditAPIConnector:
    """
    Production connector pattern for Reddit OAuth2 API using PRAW.
    Safe fallback when credentials are not configured.
    """
    def __init__(self, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None, 
                 user_agent: str = "mindgraphnet-research/1.0"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        if client_id and client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
            except Exception as e:
                print(f"Failed to initialize PRAW: {e}")

    def fetch_subreddit_posts(self, subreddit_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.reddit:
            print("Reddit connector running in stub/mock mode (no credentials provided).")
            return []
        posts = []
        try:
            sub = self.reddit.subreddit(subreddit_name)
            for p in sub.new(limit=limit):
                full_text = f"{p.title or ''} {p.selftext or ''}".strip()
                posts.append({
                    "post_id": p.id,
                    "author": str(p.author) if p.author else "[deleted]",
                    "text": full_text,
                    "topic": subreddit_name,
                    "created_at": p.created_utc,
                    "score": p.score,
                    "num_comments": p.num_comments
                })
        except Exception as e:
            print(f"Error fetching from r/{subreddit_name}: {e}")
        return posts
