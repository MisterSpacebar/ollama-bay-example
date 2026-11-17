"""Synthetic regression cases; no private bay observations are required."""
import unittest

import pandas as pd

from backend.trustworthy_bay import hypoxia_audit


class HypoxiaAuditTests(unittest.TestCase):
    @staticmethod
    def rows(days=3, station='A'):
        ts = pd.date_range('2026-07-01', periods=288 * days, freq='5min', tz='America/New_York')
        return pd.DataFrame({
            'datetime': ts.tz_convert('UTC').astype(str),
            'odo_mgL': [5.0] * len(ts),
            'odo_mgL_observed': [1] * len(ts),
            'node': station,
        })

    def test_onset_and_recovery_are_observed_only(self):
        frame = self.rows()
        frame.loc[288:575, 'odo_mgL'] = 1.5
        result = hypoxia_audit(frame)
        summary = next(x for x in result['station_summaries'] if x['threshold_mg_l'] == 2)
        self.assertEqual(summary['eligible_days'], 3)
        self.assertEqual(summary['observed_runs'], 1)
        self.assertEqual(summary['identifiable_onsets'], 1)
        self.assertEqual(summary['observed_recoveries'], 1)
        self.assertEqual(summary['hours_below_lower_bound'], 24)
        episode = next(x for x in result['episodes'] if x['threshold_mg_l'] == 2)
        self.assertFalse(episode['censored_left'])
        self.assertFalse(episode['censored_right'])

    def test_reconstructed_zeros_do_not_create_events(self):
        frame = self.rows(days=2)
        frame.loc[:287, 'odo_mgL'] = 0.0
        frame['odo_mgL_observed'] = frame['odo_mgL_observed'].astype(str)
        frame.loc[:287, 'odo_mgL_observed'] = 'false'
        result = hypoxia_audit(frame)
        self.assertEqual(result['provenance'], 'observed_flag_filtered')
        summary = next(x for x in result['station_summaries'] if x['threshold_mg_l'] == 2)
        self.assertEqual(summary['unknown_days'], 1)
        self.assertEqual(summary['observed_low_do_days'], 0)
        self.assertEqual(summary['observed_runs'], 0)

    def test_measured_zero_retained_and_censored(self):
        frame = self.rows(days=1)
        frame.loc[0, 'odo_mgL'] = 0.0
        result = hypoxia_audit(frame)
        severe = next(x for x in result['episodes'] if x['threshold_mg_l'] == 2)
        self.assertEqual(severe['minimum_do_mg_l'], 0.0)
        self.assertTrue(severe['censored_left'])
        self.assertTrue(severe['censored_right'])

    def test_missing_day_breaks_episode_and_does_not_claim_recovery(self):
        frame = self.rows(days=3)
        frame.loc[:287, 'odo_mgL'] = 1.0
        frame.loc[288:575, 'odo_mgL_observed'] = 0
        frame.loc[576:, 'odo_mgL'] = 1.0
        result = hypoxia_audit(frame)
        severe = [x for x in result['episodes'] if x['threshold_mg_l'] == 2]
        self.assertEqual(len(severe), 2)
        self.assertTrue(all(x['censored_left'] and x['censored_right'] for x in severe))

    def test_station_identities_never_merge(self):
        a, b = self.rows(days=1), self.rows(days=1, station='B')
        a['odo_mgL'] = 1.0
        result = hypoxia_audit(pd.concat([a, b], ignore_index=True))
        self.assertEqual(result['stations_analyzed'], 2)
        self.assertEqual(len(result['station_summaries']), 4)
        b_severe = next(x for x in result['station_summaries'] if x['station'] == 'B' and x['threshold_mg_l'] == 2)
        self.assertEqual(b_severe['observed_low_do_days'], 0)

    def test_same_node_name_different_source_files_remain_distinct(self):
        a, b = self.rows(days=1), self.rows(days=1)
        a['source_file'] = 'platform_2025.csv'
        b['source_file'] = 'platform_2026.csv'
        a['odo_mgL'] = 1.0
        result = hypoxia_audit(pd.concat([a, b], ignore_index=True))
        self.assertEqual(result['stations_analyzed'], 2)
        self.assertEqual(result['station_col'], 'source_file+node')
        self.assertEqual(len(result['station_summaries']), 4)

    def test_duplicate_slots_are_rejected(self):
        frame = self.rows(days=1)
        frame.loc[1, 'datetime'] = frame.loc[0, 'datetime']
        with self.assertRaisesRegex(ValueError, 'duplicate time slots'):
            hypoxia_audit(frame)

    def test_ambiguous_flags_rejected(self):
        frame = self.rows(days=1)
        frame['odo_mgL_observed'] = frame['odo_mgL_observed'].astype(str)
        frame.loc[0, 'odo_mgL_observed'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Observation flags'):
            hypoxia_audit(frame)

    def test_exact_threshold_not_below(self):
        frame = self.rows(days=1)
        frame['odo_mgL'] = 2.0
        result = hypoxia_audit(frame)
        self.assertFalse(any(x['threshold_mg_l'] == 2 for x in result['episodes']))
        self.assertTrue(any(x['threshold_mg_l'] == 4 for x in result['episodes']))

    def test_local_dst_day_expected_slots(self):
        ts = pd.date_range('2026-11-01', '2026-11-02', freq='5min', inclusive='left', tz='America/New_York')
        frame = pd.DataFrame({'datetime': ts.tz_convert('UTC').astype(str), 'odo_mgL': 5.0})
        result = hypoxia_audit(frame)
        self.assertEqual(result['latest_daily_records'][0]['expected_slots'], 300)
        self.assertEqual(result['latest_daily_records'][0]['valid_slots'], 300)
        self.assertEqual(result['provenance'], 'unflagged_assumed_raw_unverified')


if __name__ == '__main__':
    unittest.main()
