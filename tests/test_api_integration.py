"""FastAPI integration smoke tests, no Ollama server or real sensor files required."""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend import main


class APIIntegrationTests(unittest.TestCase):
    def setUp(self):
        main.datasets.clear()
        main.dataset_groups.clear()
        self.client = TestClient(main.app)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"BAY_RESEARCH_OUTPUT_DIR": str(self.dir)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_upload_and_observed_oxygen_endpoint(self):
        timestamps = pd.date_range('2026-07-01', periods=288, freq='5min', tz='UTC')
        frame = pd.DataFrame({
            'datetime': timestamps.astype(str),
            'odo_mgL': [5.] * 288,
            'odo_mgL_observed': [1] * 288,
            'node': ['S1'] * 288,
        })
        frame.loc[10, 'odo_mgL'] = 1.0
        uploaded = self.client.post('/api/upload', files={
            'file': ('synthetic.csv', io.BytesIO(frame.to_csv(index=False).encode()), 'text/csv')
        })
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        data_id = uploaded.json()['dataset_id']
        observed = self.client.post('/api/hypoxia-audit', json={
            'dataset_id': data_id, 'timezone': 'UTC',
            'timestamp_col': 'datetime', 'do_col': 'odo_mgL',
        })
        self.assertEqual(observed.status_code, 200, observed.text)
        report = observed.json()
        self.assertEqual(report['stations_analyzed'], 1)
        severe = [x for x in report['episodes'] if x['threshold_mg_l'] == 2]
        self.assertEqual(len(severe), 1)
        self.assertEqual(severe[0]['minimum_do_mg_l'], 1.0)

    def test_invalid_observation_flag_is_validation_error(self):
        frame = pd.DataFrame({
            'datetime': ['2026-07-01T00:00:00Z'],
            'odo_mgL': [1.],
            'odo_mgL_observed': ['not-verified'],
        })
        main.datasets['fixture'] = frame
        res = self.client.post('/api/hypoxia-audit', json={'dataset_id': 'fixture'})
        self.assertEqual(res.status_code, 422)

    def test_research_catalog_and_query(self):
        folder = self.dir / 'environmental_gam'
        folder.mkdir()
        pd.DataFrame([{
            'outcome': 'daily_min_do', 'threshold_mgL': 4, 'model': 'baseline',
            'n_paired': 12, 'skill_vs_baseline': 0,
        }]).to_csv(folder / 'paired_scores.csv', index=False)
        catalog = self.client.get('/api/research/catalog')
        self.assertEqual(catalog.status_code, 200)
        self.assertIn('gam_scores', [x['artifact'] for x in catalog.json()['available']])
        result = self.client.post('/api/research/query', json={'artifact': 'gam_scores'})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['rows'][0]['n_paired'], 12)
        wrong = self.client.post('/api/research/query', json={
            'artifact': 'gam_scores', 'horizon_days': 7,
        })
        self.assertEqual(wrong.status_code, 422)

    def test_chat_research_mode_does_not_require_dataset(self):
        # Stub the LLM completion to validate route wiring without needing Ollama.
        class Msg:
            tool_calls = []
            content = 'The research result is retrospective, not a live warning.'
        class Response:
            message = Msg()
        async def fake_chat(*args, **kwargs):
            self.assertEqual([t['function']['name'] for t in kwargs['tools']],
                             ['query_research_output'])
            return Response()
        with patch.object(main.client, 'chat', side_effect=fake_chat):
            resp = self.client.post('/api/chat', json={
                'research_only': True, 'question': 'What does the warning model mean?'
            })
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn('retrospective', resp.text)

    def _fake_response(self, text='Answer text.'):
        class Msg:
            tool_calls = []
            content = text
        class Response:
            message = Msg()
        return Response()

    def test_chat_streams_status_and_keepalives_while_the_model_is_busy(self):
        import asyncio

        async def slow_chat(*args, **kwargs):
            await asyncio.sleep(0.4)
            return self._fake_response('Slow but answered.')
        with patch.object(main.client, 'chat', side_effect=slow_chat), \
                patch.object(main, 'CHAT_HEARTBEAT_SECONDS', 0.05):
            resp = self.client.post('/api/chat', json={'research_only': True, 'question': 'q'})
        self.assertEqual(resp.status_code, 200, resp.text)
        text = resp.text
        # Progress and keep-alives arrive before the answer, so clients never sit silent.
        self.assertIn('"status": "Asking the model', text)
        self.assertIn(': keep-alive', text)
        self.assertLess(text.index(': keep-alive'), text.index('"token"'))
        self.assertIn('Slow but answered.', text)
        self.assertTrue(text.rstrip().endswith('data: [DONE]'))

    def test_chat_pipeline_error_is_reported_as_an_event_not_a_dropped_stream(self):
        async def failing_chat(*args, **kwargs):
            raise RuntimeError('ollama unreachable')
        with patch.object(main.client, 'chat', side_effect=failing_chat):
            resp = self.client.post('/api/chat', json={'research_only': True, 'question': 'q'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('"error": "ollama unreachable"', resp.text)
        self.assertTrue(resp.text.rstrip().endswith('data: [DONE]'))

    def test_client_disconnect_cancels_the_in_flight_model_request(self):
        import asyncio

        async def scenario():
            started, cancelled = asyncio.Event(), asyncio.Event()

            async def hanging_chat(*args, **kwargs):
                started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            with patch.object(main.client, 'chat', side_effect=hanging_chat), \
                    patch.object(main, 'CHAT_HEARTBEAT_SECONDS', 0.05):
                response = await main.chat(main.ChatRequest(research_only=True, question='q'))
                stream = response.body_iterator
                await stream.__anext__()  # first status event
                await asyncio.wait_for(started.wait(), 2)
                await stream.aclose()  # what the server does when the client goes away
                await asyncio.wait_for(cancelled.wait(), 2)
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
