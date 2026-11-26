import unittest

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from scripts.run_shap_analysis import (
    decode_feature_index,
    map_ranked_bits,
    rank_mean_absolute_shap,
    tree_shap_values,
)


class ShapMappingTests(unittest.TestCase):
    def test_tree_shap_values_are_additive_on_a_real_xgboost_model(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0.0, 1.0, 1.0, 2.0])
        model = XGBRegressor(n_estimators=4, max_depth=2, random_state=42)
        model.fit(X, y)

        values, bias = tree_shap_values(model, X)
        raw_margin = model.predict(X, output_margin=True)

        self.assertEqual(values.shape, X.shape)
        self.assertEqual(bias.shape, (len(X),))
        np.testing.assert_allclose(values.sum(axis=1) + bias, raw_margin, atol=1e-6)

    def test_decode_feature_index_uses_component_local_bit_numbers(self):
        self.assertEqual(decode_feature_index(0, bits=2048), ("substrate1", 0))
        self.assertEqual(decode_feature_index(2048, bits=2048), ("organoboron", 0))
        self.assertEqual(decode_feature_index(4096 + 561, bits=2048), ("product", 561))

    def test_rank_mean_absolute_shap_reports_block_and_local_bit(self):
        values = np.zeros((2, 6), dtype=float)
        values[:, 2] = [2.0, -4.0]
        values[:, 4] = [1.0, -1.0]

        ranked = rank_mean_absolute_shap(values, bits=2, top_n=2)

        self.assertEqual(
            ranked.to_dict("records"),
            [
                {
                    "rank": 1,
                    "global_feature": 2,
                    "feature_block": "organoboron",
                    "bit": 0,
                    "mean_abs_shap": 3.0,
                },
                {
                    "rank": 2,
                    "global_feature": 4,
                    "feature_block": "product",
                    "bit": 0,
                    "mean_abs_shap": 1.0,
                },
            ],
        )

    def test_map_ranked_bits_limits_search_to_the_ranked_feature_block(self):
        data = pd.DataFrame(
            {
                "sub_smiles1": ["CC"],
                "sub_smiles2": ["c1ccccc1B(O)O"],
                "prod_smiles": ["c1ccccc1C"],
            }
        )
        ranked = pd.DataFrame(
            [
                {
                    "task": "classifier",
                    "rank": 1,
                    "global_feature": 0,
                    "feature_block": "organoboron",
                    "bit": 0,
                    "mean_abs_shap": 1.0,
                }
            ]
        )

        mappings = map_ranked_bits(
            data,
            ranked,
            radius=0,
            n_bits=1,
            max_examples=1,
        )

        self.assertEqual(len(mappings), 1)
        row = mappings.iloc[0]
        self.assertEqual(row["feature_block"], "organoboron")
        self.assertEqual(row["source_column"], "sub_smiles2")
        self.assertIn("B", row["parent_smiles"])
        self.assertNotEqual(row["environment_smiles"], row["parent_smiles"])

    def test_map_ranked_bits_distinguishes_unique_molecules_from_reaction_rows(self):
        data = pd.DataFrame(
            {
                "sub_smiles1": ["CC", "CCC"],
                "sub_smiles2": ["c1ccccc1B(O)O", "c1ccccc1B(O)O"],
                "prod_smiles": ["c1ccccc1C", "c1ccccc1CC"],
            }
        )
        ranked = pd.DataFrame(
            [
                {
                    "task": "classifier",
                    "rank": 1,
                    "global_feature": 0,
                    "feature_block": "organoboron",
                    "bit": 0,
                    "mean_abs_shap": 1.0,
                }
            ]
        )

        mappings = map_ranked_bits(
            data,
            ranked,
            radius=0,
            n_bits=1,
            max_examples=1,
        )

        row = mappings.iloc[0]
        self.assertEqual(row["parent_molecule_count"], 1)
        self.assertEqual(row["parent_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
