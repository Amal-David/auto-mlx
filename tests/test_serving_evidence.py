"""No engine claims may be inferred from unmatched contexts or HTTP concurrency."""
from __future__ import annotations
import contextlib
import copy
from dataclasses import fields, replace
import http.client
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from auto_mlx import cli
from auto_mlx.errors import AutoMLXError
from auto_mlx.serving_evidence import CacheNamespace, ServingComparisonContext, compare_contexts, native_capabilities, METRICS
from auto_mlx.inference_server import LocalInferenceServer
from auto_mlx.schemas import schema_text
try:
    import jsonschema
except ImportError:
    jsonschema=None


def context():
    values={f.name:'a'*64 for f in fields(ServingComparisonContext) if f.name.endswith('_sha256')}
    values.update(engine='reference',engine_revision='b'*40,cache_state='cold',cache_precision='bf16-kv-v1',model_load_state='resident',concurrency=1,input_tokens=2048,max_output_tokens=128,metric='request_first_content_ns',unit='ns',quality_passed=True)
    return ServingComparisonContext(**values)


class ServingEvidenceTests(unittest.TestCase):
    def test_cache_namespace_round_trip_is_immutable(self):
        value=CacheNamespace(**{f.name:('cache-v1' if f.name=='cache_abi' else 'a'*64) for f in fields(CacheNamespace)})
        self.assertEqual(value,CacheNamespace.from_dict(value.to_dict()))
        with self.assertRaises(AttributeError):value.cache_abi='changed'
        self.assertRegex(value.identity,r'^[0-9a-f]{64}$')

    def test_every_cache_dimension_changes_identity(self):
        original=CacheNamespace(**{f.name:('cache-v1' if f.name=='cache_abi' else 'a'*64) for f in fields(CacheNamespace)})
        for f in fields(original):
            with self.subTest(field=f.name):
                changed=replace(original,**{f.name:'cache-v2' if f.name=='cache_abi' else 'b'*64})
                self.assertNotEqual(original.identity,changed.identity)

    def test_cache_missing_or_unknown_fields_fail(self):
        value={f.name:('cache-v1' if f.name=='cache_abi' else 'a'*64) for f in fields(CacheNamespace)}
        for mutate in (lambda d:d.update(command='execute'),lambda d:d.pop('tenant_namespace_sha256'),lambda d:d.update(model_sha256='bad')):
            data=copy.deepcopy(value);mutate(data)
            with self.assertRaises(AutoMLXError):CacheNamespace.from_dict(data)

    def test_equal_contexts_allow_comparison_not_promotion(self):
        result=compare_contexts(context(),context())
        self.assertTrue(result['comparable'])
        self.assertFalse(result['promotion_allowed'])
        self.assertIsNone(result['speedup'])

    def test_engine_and_revision_may_differ(self):
        other=replace(context(),engine='candidate',engine_revision='c'*40)
        self.assertTrue(compare_contexts(context(),other)['comparable'])

    def test_each_identity_mismatch_blocks_comparison(self):
        for f in fields(ServingComparisonContext):
            if f.name.endswith('_sha256'):
                with self.subTest(field=f.name):
                    result=compare_contexts(context(),replace(context(),**{f.name:'c'*64}))
                    self.assertFalse(result['comparable'])
                    self.assertTrue(any(f.name in text for text in result['blockers']))

    def test_cold_warm_and_load_regimes_must_match(self):
        for changes in ({'cache_state':'warm'},{'cache_state':'partial'},{'model_load_state':'cold'},{'concurrency':4},{'cache_precision':'q8-with-bf16-shadow'},{'input_tokens':4096},{'max_output_tokens':256}):
            self.assertFalse(compare_contexts(context(),replace(context(),**changes))['comparable'])

    def test_physical_and_logical_prefill_are_not_comparable(self):
        a=replace(context(),metric='physical_prefill_tokens_per_second',unit='tokens_per_second')
        b=replace(a,metric='logical_prefill_tokens_per_second')
        self.assertFalse(compare_contexts(a,b)['comparable'])

    def test_quality_failure_never_allows_comparison(self):
        self.assertFalse(compare_contexts(context(),replace(context(),quality_passed=False))['comparable'])

    def test_units_and_integer_bounds_are_strict(self):
        for changes in ({'unit':'tokens_per_second'},{'concurrency':True},{'concurrency':0},{'input_tokens':1.5},{'engine_revision':'main'},{'quality_passed':'yes'},{'metric':[]},{'cache_state':[]}):
            with self.subTest(changes=changes),self.assertRaises(AutoMLXError):replace(context(),**changes)

    def test_unknown_context_field_is_rejected(self):
        data=context().to_dict();data['speedup']=100
        with self.assertRaises(AutoMLXError):ServingComparisonContext.from_dict(data)

    def test_python_contract_and_schema_fields_align(self):
        for cls,name in ((CacheNamespace,'cache_namespace.json'),(ServingComparisonContext,'serving_comparison_context.json')):
            schema=json.loads(schema_text(name))
            self.assertEqual(set(schema['properties']),{f.name for f in fields(cls)})
            self.assertEqual(set(schema['required']),set(schema['properties']))
            self.assertFalse(schema['additionalProperties'])

    @unittest.skipIf(jsonschema is None,'optional jsonschema is unavailable')
    def test_live_schema_accepts_contract_and_rejects_wrong_unit(self):
        validator=jsonschema.Draft202012Validator(json.loads(schema_text('serving_comparison_context.json')))
        validator.validate(context().to_dict())
        data=context().to_dict();data['unit']='bytes'
        self.assertTrue(list(validator.iter_errors(data)))

    def test_native_capabilities_do_not_invent_batching_or_cache(self):
        for device in ('cpu','gpu','unknown'):
            caps=native_capabilities(device)
            self.assertEqual(caps['max_compute_batch_size'],1)
            self.assertFalse(caps['continuous_batching'])
            self.assertEqual(caps['prefix_cache'],'none')
            self.assertFalse(caps['device_execution_verified_by_this_report'])
        with self.assertRaises(AutoMLXError):native_capabilities([])

    def test_comparison_cli_is_offline_and_does_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();a=base/'a.json';b=base/'b.json'
            a.write_text(json.dumps(context().to_dict()));b.write_text(json.dumps(replace(context(),cache_state='warm').to_dict()))
            output=io.StringIO()
            with contextlib.redirect_stdout(output):code=cli.main(['compare-serving','--baseline',str(a),'--candidate',str(b)])
            self.assertEqual(code,0)
            result=json.loads(output.getvalue())
            self.assertFalse(result['comparable']);self.assertFalse(result['promotion_allowed'])


class ServingAPIMetadataTests(unittest.TestCase):
    def setUp(self):
        class Session:
            bundle=SimpleNamespace(bundle_id='a'*64,device='cpu')
            empty=False
            def stream(self,request,**kwargs):
                if not self.empty:
                    yield {'event':'delta','text':'','token':1}
                    yield {'event':'delta','text':'answer','token':2}
                yield {'event':'done','text':'' if self.empty else 'answer','finish_reason':'stop','prompt_tokens':3,'completion_tokens':0 if self.empty else 2,'time_to_first_token_ns':1,'generation_elapsed_ns':2,'peak_memory_bytes':3}
        self.session=Session()
        self.server=LocalInferenceServer(0,self.session,lambda request:{'mode':'native_fallback','prefill_step_size':2048})
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown();self.server.server_close();self.thread.join(5)

    def call(self,method='POST',path='/v1/completions',stream=False):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        try:
            body=json.dumps({'model':'a'*64,'prompt':'test','max_tokens':2,'stream':stream}) if method=='POST' else None
            conn.request(method,path,body,{'Content-Type':'application/json'})
            response=conn.getresponse();return response.status,response.read()
        finally:conn.close()

    def test_health_reports_compute_capabilities_not_http_slots(self):
        status,body=self.call('GET','/health')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['capabilities']['max_compute_batch_size'],1)

    def test_completion_reports_parent_boundary_and_worker_metrics(self):
        status,body=self.call();self.assertEqual(status,200)
        data=json.loads(body)['auto_mlx'];m=data['metrics']
        self.assertGreaterEqual(m['request_first_content_ns'],0)
        self.assertGreaterEqual(m['request_end_to_end_ns'],m['request_first_content_ns'])
        self.assertEqual(m['time_to_first_token_ns'],1)
        self.assertIn('before-final-response-write',m['measurement_boundary'])

    def test_sse_reports_same_metric_semantics(self):
        status,body=self.call(stream=True);self.assertEqual(status,200)
        objects=[json.loads(line[6:]) for line in body.splitlines() if line.startswith(b'data: {')]
        final=next(row for row in objects if 'auto_mlx' in row)
        self.assertIn('request_first_content_ns',final['auto_mlx']['metrics'])
        self.assertFalse(final['auto_mlx']['capabilities']['continuous_batching'])

    def test_empty_completion_does_not_invent_first_content(self):
        self.session.empty=True
        status,body=self.call();self.assertEqual(status,200)
        self.assertIsNone(json.loads(body)['auto_mlx']['metrics']['request_first_content_ns'])

    def test_http_overload_returns_explicit_retryable_error(self):
        acquired=[self.server._slots.acquire(blocking=False) for _ in range(4)]
        self.assertTrue(all(acquired))
        try:
            status,body=self.call('GET','/health')
            self.assertEqual(status,503)
            self.assertIn('admission',json.loads(body)['error']['message'])
        finally:
            for _ in range(4):self.server._slots.release()


if __name__=='__main__':unittest.main()
