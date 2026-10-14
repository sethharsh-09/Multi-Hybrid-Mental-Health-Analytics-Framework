"""
MindGraphNet - Token Saliency & Attention Highlighter
Renders word-level attention heatmaps and plain-English clinical explanations.
"""
from typing import List, Dict, Tuple, Any
import numpy as np

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

class TokenSaliencyExplainer:
    @staticmethod
    def process_attention(attention_matrix: np.ndarray, tokens: List[str]) -> Tuple[List[str], np.ndarray, str]:
        """
        Collapses multi-head attention matrix into token scores, merges wordpieces,
        and generates an interactive HTML highlighted snippet.
        """
        # Column-wise mean attention received
        if attention_matrix.ndim == 2:
            raw_scores = attention_matrix.mean(axis=0)
        else:
            raw_scores = np.ones(len(tokens))

        words, scores = [], []
        for tok, sc in zip(tokens, raw_scores):
            if tok in SPECIAL_TOKENS:
                continue
            if tok.startswith("##") and words:
                words[-1] += tok[2:]
                scores[-1] = max(scores[-1], sc)
            else:
                words.append(tok)
                scores.append(sc)

        scores_arr = np.array(scores) if scores else np.array([0.0])
        min_s, max_s = scores_arr.min(), scores_arr.max()
        norm_scores = (scores_arr - min_s) / (max_s - min_s + 1e-8) if max_s > min_s else np.zeros_like(scores_arr)

        # Build HTML highlighting
        html_spans = []
        for w, ns in zip(words, norm_scores):
            # Crimson red intensity based on normalized score
            alpha = float(0.12 + 0.70 * ns)
            html_spans.append(
                f'<span style="background-color:rgba(220,38,38,{alpha:.2f}); '
                f'padding:2px 6px; margin:2px; border-radius:4px; font-weight:{"bold" if ns > 0.6 else "normal"}; '
                f'display:inline-block;">{w}</span>'
            )
        html_rendered = " ".join(html_spans)

        return words, norm_scores, html_rendered

    @staticmethod
    def get_top_attended_words(words: List[str], norm_scores: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not words:
            return []
        sorted_indices = np.argsort(norm_scores)[::-1][:top_k]
        return [(words[i], round(float(norm_scores[i]), 3)) for i in sorted_indices]
