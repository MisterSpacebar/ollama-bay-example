import unittest

import pandas as pd

from backend.fcm_safe import safe_fcm


class FcmSafeTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'datetime': pd.date_range('2026-07-01', periods=24, freq='h', tz='UTC').astype(str),
            'node': ['A'] * 24,
            'temp_c': [27 + i / 24 for i in range(24)],
            'odo_mgL': [5 + (i % 6) / 10 for i in range(24)],
        })

    def test_sorted_single_station(self):
        df = self.frame().sample(frac=1, random_state=7).reset_index(drop=True)
        result = safe_fcm(df, columns=['temp_c', 'odo_mgL'])
        self.assertEqual(result['sample_interval_minutes'], 60)
        self.assertIn('noncausal', result['interpretation'])
        self.assertIn('top_causal_edges', result)

    def test_mixed_nodes_rejected(self):
        df = self.frame()
        df.loc[8:, 'node'] = 'B'
        with self.assertRaisesRegex(ValueError, 'one station'):
            safe_fcm(df, columns=['temp_c', 'odo_mgL'])

    def test_same_node_label_with_multiple_source_files_rejected(self):
        df = self.frame()
        df['source_file'] = ['2025_platform.csv'] * 12 + ['2026_platform.csv'] * 12
        with self.assertRaisesRegex(ValueError, 'multiple station/source identities'):
            safe_fcm(df, columns=['temp_c', 'odo_mgL'])

    def test_irregular_time_rejected(self):
        df = self.frame().drop(index=8)
        with self.assertRaisesRegex(ValueError, 'regular sampling'):
            safe_fcm(df, columns=['temp_c', 'odo_mgL'])

    def test_no_timestamp_rejected(self):
        df = self.frame().drop(columns='datetime')
        with self.assertRaisesRegex(ValueError, 'timestamp_col'):
            safe_fcm(df, columns=['temp_c', 'odo_mgL'])


if __name__ == '__main__':
    unittest.main()
