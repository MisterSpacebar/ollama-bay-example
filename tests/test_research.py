"""Read-only adapter tests with synthetic upstream-format research CSV files."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.research import query_research, research_catalog


def write(path: Path, filename: str, rows: list[dict]):
    target = path / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


class ResearchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def make_warnings(self):
        rows = []
        for model, value in [('markov', .2), ('do_history', .3), ('environmental', .4)]:
            rows.append({
                'node': 'S1', 'issue_date': '2025-07-15',
                'horizon_days': 3, 'origin_state': 'Normal',
                'model': model, 'label': 0, 'probability': value,
                'provenance': 'observed_only',
            })
        write(self.root, 'transition_warning/hindcast_predictions.csv', rows)

    def test_catalog_missing_and_available(self):
        self.assertEqual(research_catalog(self.root)['available'], [])
        self.make_warnings()
        found = {x['artifact'] for x in research_catalog(self.root)['available']}
        self.assertEqual(found, {'warning_hindcasts'})

    def test_hindcast_query_preserves_model_and_outcome(self):
        self.make_warnings()
        r = query_research('warning_hindcasts', station='S1', horizon_days=3,
                           issue_date='2025-07-15', model='do_history', root=self.root)
        self.assertEqual(r['n_matching'], 1)
        self.assertEqual(r['rows'][0]['probability'], .3)
        self.assertEqual(r['rows'][0]['label'], 0)
        self.assertEqual(r['forecast_type'], 'retrospective_out_of_sample_hindcast')
        self.assertEqual(len(r['file_sha256']), 64)

    def test_unknown_station_is_not_zero_risk(self):
        self.make_warnings()
        r = query_research('warning_hindcasts', station='unseen', root=self.root)
        self.assertEqual(r['status'], 'no_matching_rows')
        self.assertEqual(r['rows'], [])

    def test_rejects_unmatched_hindcast_models(self):
        self.make_warnings()
        path = self.root / 'transition_warning/hindcast_predictions.csv'
        df = pd.read_csv(path).iloc[:2]
        df.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'all three'):
            query_research('warning_hindcasts', root=self.root)

    def test_rejects_imputed_provenance(self):
        self.make_warnings()
        path = self.root / 'transition_warning/hindcast_predictions.csv'
        df = pd.read_csv(path)
        df.loc[0, 'provenance'] = 'imputed'
        df.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'non-observed'):
            query_research('warning_hindcasts', root=self.root)

    def test_rejects_future_label_as_probability(self):
        self.make_warnings()
        path = self.root / 'transition_warning/hindcast_predictions.csv'
        df = pd.read_csv(path)
        df.loc[0, 'probability'] = 1.5
        df.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'outside'):
            query_research('warning_hindcasts', root=self.root)

    def test_rejects_unsupported_filter_and_bad_date(self):
        self.make_warnings()
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            query_research('warning_hindcasts', threshold_mg_l=2, root=self.root)
        with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
            query_research('warning_hindcasts', issue_date='2025/07/15', root=self.root)

    def test_gam_scores_are_not_live_forecasts(self):
        write(self.root, 'environmental_gam/paired_scores.csv', [{
            'outcome': 'daily_min_do', 'threshold_mgL': 4,
            'model': 'environmental_gam', 'n_paired': 12,
            'skill_vs_baseline': -.1,
        }])
        r = query_research('gam_scores', root=self.root)
        self.assertEqual(r['forecast_type'], 'historical_research_output')
        self.assertEqual(r['rows'][0]['skill_vs_baseline'], -.1)

    def test_episode_censoring_columns_from_upstream(self):
        write(self.root, 'episode_dynamics/episode_table.csv', [{
            'node': 'S1', 'threshold_mgL': 2,
            'start_date_local': '2025-07-01',
            'last_hypoxic_date_local': '2025-07-03',
            'censored_left': True, 'censored_right': True,
            'provenance': 'observed_only',
        }])
        r = query_research('episode_runs', root=self.root)
        self.assertTrue(r['rows'][0]['censored_right'])

    def test_path_escape_symlink_rejected(self):
        self.make_warnings()
        with tempfile.TemporaryDirectory() as outside:
            bad = self.root / 'environmental_gam'
            bad.symlink_to(Path(outside), target_is_directory=True)
            self.assertNotIn('gam_scores', [x['artifact'] for x in research_catalog(self.root)['available']])
            write(Path(outside), 'paired_scores.csv', [{
                'outcome': 'o', 'threshold_mgL': 2,
                'model': 'baseline', 'n_paired': 2, 'skill_vs_baseline': 0,
            }])
            with self.assertRaisesRegex(ValueError, 'escapes'):
                query_research('gam_scores', root=self.root)


if __name__ == '__main__':
    unittest.main()
