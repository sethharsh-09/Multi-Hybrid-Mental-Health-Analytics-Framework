"""
Unit and Integration Tests for MindGraphNet Multi-Hybrid Architecture
"""
import unittest
import numpy as np
import pandas as pd
import torch

from mindgraphnet.ingestion.validator import IngestionValidator
from mindgraphnet.preprocessing.anonymizer import DataAnonymizer
from mindgraphnet.preprocessing.text_cleaner import ClinicalTextCleaner
from mindgraphnet.features.nlp_features import NLPFeatureExtractor
from mindgraphnet.features.psycholinguistics import PsycholinguisticExtractor
from mindgraphnet.engine.fusion_model import MultiHybridFusionNet
from mindgraphnet.analytics.risk_scorer import ClinicalRiskScorer

class TestMindGraphNetPipeline(unittest.TestCase):
    def setUp(self):
        self.anonymizer = DataAnonymizer()
        self.cleaner = ClinicalTextCleaner()
        self.validator = IngestionValidator()
        self.nlp = NLPFeatureExtractor()
        self.psych = PsycholinguisticExtractor()
        self.fusion = MultiHybridFusionNet()
        self.scorer = ClinicalRiskScorer()

    def test_anonymizer_scrubbing(self):
        raw_text = "Reach me at test.user@mentalhealth.org or call 555-123-4567, thanks from @user123."
        scrubbed = self.anonymizer.scrub_pii(raw_text)
        self.assertNotIn("test.user@mentalhealth.org", scrubbed)
        self.assertNotIn("555-123-4567", scrubbed)
        self.assertNotIn("@user123", scrubbed)
        self.assertIn("<REDACTED_EMAIL>", scrubbed)

    def test_author_pseudonymization(self):
        author = "secret_patient_42"
        anon1 = self.anonymizer.pseudonymize_identifier(author)
        anon2 = self.anonymizer.pseudonymize_identifier(author)
        self.assertEqual(anon1, anon2) # Deterministic hashing
        self.assertNotEqual(anon1, author)
        self.assertTrue(anon1.startswith("usr_"))

    def test_text_cleaning_negation_preservation(self):
        text = "I can't feel happy and I am not okay."
        cleaned = self.cleaner.clean_text(text)
        tokens = self.cleaner.tokenize(cleaned)
        nostop = self.cleaner.remove_stopwords(tokens)
        self.assertIn("cannot", nostop) # Contraction expansion & preserved negation
        self.assertIn("not", nostop)

    def test_sentiment_and_emotions(self):
        text = "I feel so sad, hopeless, crying alone in the dark."
        sentiment = self.nlp.get_sentiment(text)
        self.assertLess(sentiment["sentiment_score"], 0)
        self.assertEqual(sentiment["sentiment_label"], "negative")

        tokens = text.split()
        emotions = self.nlp.detect_emotions(tokens)
        self.assertGreater(emotions["emo_sadness"] + emotions["emo_hopelessness"], 0)

    def test_psycholinguistic_rumination(self):
        text = "I feel like my life is falling apart and I cannot help myself."
        cleaned = self.cleaner.clean_text(text)
        tokens = cleaned.split()
        feats = self.psych.extract_features(text, tokens)
        self.assertGreater(feats["first_person_ratio"], 0.15)
        self.assertGreater(feats["distress_index"], 0.20)

    def test_fusion_net_forward(self):
        sem_x = torch.randn(2, 768)
        gr_x = torch.randn(2, 32)
        psy_x = torch.randn(2, 16)
        logits, risk_cont, weights = self.fusion(sem_x, gr_x, psy_x)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(risk_cont.shape, (2, 1))
        self.assertEqual(weights.shape, (2, 3))
        # Check weights sum to 1
        weights_sum = weights.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(weights_sum, np.ones(2), atol=1e-5)

    def test_crisis_scorer(self):
        crisis_text = "I want to kill myself and end it all tonight."
        probs = np.array([0.1, 0.2, 0.7])
        emo_dict = {"emo_sadness": 0.5, "emo_hopelessness": 0.5, "emo_fear": 0.0, "emo_joy": 0.0}
        eval_res = self.scorer.evaluate_post(crisis_text, probs, 0.85, emo_dict, -0.8)
        self.assertEqual(eval_res["risk_level"], "High")
        self.assertTrue(eval_res["is_crisis"])

if __name__ == "__main__":
    unittest.main()
