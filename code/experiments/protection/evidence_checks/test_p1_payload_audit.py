import json
import unittest
from unittest.mock import patch
from mas_faults.llm_client import OpenAICompatibleHTTPClient


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(dict(model='deepseek-flash', id='offline',
            choices=[dict(message=dict(content='{}'))],
            usage=dict(prompt_tokens=1, completion_tokens=1))).encode()


class PayloadAuditTest(unittest.TestCase):
    def test_actual_parameters_match_transport_without_credentials_or_messages(self):
        client = OpenAICompatibleHTTPClient(api_key='offline-secret',
            base_url='https://api.deepseek.com', model='deepseek-flash', provider='deepseek')
        sent = []
        def transport(request, timeout):
            sent.append((json.loads(request.data), timeout))
            return Response()
        with patch.dict('os.environ', {'LLM_MAX_TOKENS': '2048',
                'LLM_TOTAL_REQUEST_TIMEOUT_SECONDS': '120'}), \
                patch('mas_faults.llm_client.ensure_deepseek_offpeak'), \
                patch('mas_faults.llm_client.urllib.request.urlopen', transport):
            client.complete('private task JSON', json_mode=True)
        record = client.request_log[0]
        expected = {k: v for k, v in sent[0][0].items() if k != 'messages'}
        self.assertEqual(record['request_parameters'], expected)
        self.assertEqual(record['request_timeout_seconds'], sent[0][1])
        self.assertEqual(record['total_timeout_seconds'], 120)
        self.assertNotIn('offline-secret', json.dumps(record))
        self.assertNotIn('private task', json.dumps(record))

    def test_transport_error_still_has_parameters(self):
        client = OpenAICompatibleHTTPClient(api_key='offline-secret',
            base_url='https://api.deepseek.com', model='deepseek-flash', provider='deepseek')
        with patch.dict('os.environ', {'LLM_MAX_TOKENS': '2048',
                'LLM_TOTAL_REQUEST_TIMEOUT_SECONDS': '120'}), \
                patch('mas_faults.llm_client.ensure_deepseek_offpeak'), \
                patch('mas_faults.llm_client.urllib.request.urlopen', side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, 'TimeoutError'):
                client.complete('private task')
        self.assertEqual(client.request_log[0]['request_parameters']['max_tokens'], 2048)
        self.assertEqual(client.request_log[0]['total_timeout_seconds'], 120)


if __name__ == '__main__':
    unittest.main()
