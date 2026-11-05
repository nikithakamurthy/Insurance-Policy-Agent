"""Deterministic regressions for the app's transport and access rules. Model responses are controlled fixtures."""
import asyncio, base64, json, sys, time, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_demo import server as s
from fastapi.testclient import TestClient


class SessionTests(unittest.TestCase):
    def test_agent_reply_does_not_count_as_a_claimant_revision(self):
        session=s.IntakeSession('revision')
        s.append_turn(session,'Claimant','I want to add my books','u1')
        revision=session.revision
        s.append_turn(session,'Agent','Which country do you live in?','a1')
        self.assertEqual(session.revision,revision)

    def test_turn_retry_deduplicates_only_identifier(self):
        session=s.IntakeSession('test')
        s.append_turn(session,'Claimant','No injuries','1');s.append_turn(session,'Claimant','No','2');s.append_turn(session,'Claimant','No','2')
        self.assertEqual(len(session.transcript),2)

    def test_ui_state_is_what_the_page_shows(self):
        session=s.IntakeSession('ui');s.set_country(session,'India')
        state=s._ui_state(session)
        self.assertEqual(state['locale'],{'country':'India','currency':'INR'})
        self.assertEqual(sorted(state),['items','live_model','locale','revision','session_id','tool_activity','transcript'])


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_state_turns_inform_model_without_claimant_facts(self):
        session=s.IntakeSession('camera-wire',owner='owner');s.sessions[session.session_id]=session
        messages=[dict(type='camera_state',enabled=True),dict(type='camera_state',enabled=False),dict(type='close')]
        delivered=[]
        class WS:
            headers={'host':'127.0.0.1:4188','origin':'http://127.0.0.1:4188'}
            client=NS(host='127.0.0.1');url=NS(scheme='ws');cookies={'intake_owner':'owner'};query_params={'session_id':'camera-wire'}
            async def accept(self):pass
            async def close(self,**kw):pass
            async def send_json(self,payload):pass
            async def receive_text(self):return json.dumps(messages.pop(0))
        class Live:
            async def send_client_content(self,**kw):delivered.append(kw)
            async def receive(self):
                await asyncio.Event().wait()
                yield None
        class CM:
            async def __aenter__(self):return Live()
            async def __aexit__(self,*args):pass
        fake=NS(aio=NS(live=NS(connect=lambda **kw:CM())))
        with patch.object(s,'_has_api_key',return_value=True),patch.object(s,'_client',return_value=fake):
            await asyncio.wait_for(s.live_voice(WS()),1)
        self.assertEqual([x['turn_complete'] for x in delivered],[False,False])
        self.assertIn('STATE: ON',delivered[0]['turns'].parts[0].text)
        self.assertIn('STATE: OFF',delivered[1]['turns'].parts[0].text)
        self.assertEqual(session.transcript,[])
        s.sessions.pop(session.session_id)

    async def test_cleanup_deletes_frames_and_cancels_jobs(self):
        session=s.IntakeSession('expired',updated_at=time.monotonic()-s.SESSION_TTL-1,last_frame=b'image')
        job=asyncio.create_task(asyncio.sleep(100));session.tasks.add(job);s.sessions[session.session_id]=session
        await s.cleanup_sessions()
        self.assertNotIn('expired',s.sessions);self.assertTrue(job.cancelled());self.assertIsNone(session.last_frame)

    async def test_typed_turn_does_not_block_media(self):
        session=s.IntakeSession('transport',owner='owner');s.sessions[session.session_id]=session
        media=[]
        class WS:
            headers={'host':'127.0.0.1:4188','origin':'http://127.0.0.1:4188'}
            client=NS(host='127.0.0.1');url=NS(scheme='ws');cookies={'intake_owner':'owner'};query_params={'session_id':'transport'}
            def __init__(self):self.i=0;self.sent=[]
            async def accept(self):pass
            async def close(self,**kw):pass
            async def send_json(self,payload):self.sent.append(payload)
            async def receive_text(self):
                messages=[dict(type='text',text='Test claim',id='1'),dict(type='video',data=base64.b64encode(b'\xff\xd8\xffTEST').decode()),dict(type='audio',data='AAA='),dict(type='close')]
                value=messages[self.i];self.i+=1;return json.dumps(value)
        class Live:
            async def send_client_content(self,**kw):pass
            async def send_realtime_input(self,**kw):media.extend(kw)
            async def receive(self):
                await asyncio.Event().wait()
                yield None
        class CM:
            async def __aenter__(self):return Live()
            async def __aexit__(self,*args):pass
        fake=NS(aio=NS(live=NS(connect=lambda **kw:CM())))
        with patch.object(s,'_has_api_key',return_value=True),patch.object(s,'_client',return_value=fake),patch.object(s,'VISION_URL',''):
            ws=WS();await asyncio.wait_for(s.live_voice(ws),1)
        self.assertEqual(media,['video','audio']);self.assertFalse(session.camera_enabled);self.assertIsNone(session.live_socket);self.assertEqual(session.transcript[0]['id'],'1')
        s.sessions.pop('transport')


class AccessTests(unittest.TestCase):
    def setUp(self):
        s.sessions.clear();self.client=TestClient(s.app,base_url='http://127.0.0.1:4188');self.headers={'Origin':'http://127.0.0.1:4188'}
    def tearDown(self):self.client.close();s.sessions.clear()
    def create(self):return self.client.post('/api/sessions',headers=self.headers).json()['session_id']
    def test_owner_cookie_required(self):
        sid=self.create();self.assertEqual(self.client.get('/api/sessions/'+sid).status_code,200)
        with TestClient(s.app,base_url='http://127.0.0.1:4188') as other:
            self.assertEqual(other.get('/api/sessions/'+sid).status_code,404)
    def test_origin_host_and_missing_origin_rejected(self):
        for headers in [{},{'Origin':'https://evil.example'},{'Origin':'http://evil.example','Host':'evil.example'}]:
            self.assertEqual(self.client.post('/api/sessions',headers=headers).status_code,403)
    def test_websocket_rejects_origin_and_wrong_owner(self):
        from starlette.websockets import WebSocketDisconnect
        sid=self.create()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/live?session_id='+sid,headers={'Origin':'https://evil.example'}):pass
        self.client.cookies.clear()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/live?session_id='+sid,headers=self.headers):pass
    def test_reset_deletes_old_claim(self):
        old=self.create();s.append_turn(s.sessions[old],'Claimant','Old confidential claim')
        self.assertEqual(self.client.delete('/api/sessions/'+old,headers=self.headers).status_code,200)
        new=self.create();self.assertNotEqual(old,new)
        self.assertEqual(self.client.get('/api/sessions/'+old).status_code,404)
        self.assertEqual(len(self.client.get('/api/sessions/'+new).json()['state']['transcript']),1)
    def test_session_limit_and_source_not_served(self):
        for _ in range(4):self.create()
        self.assertEqual(self.client.post('/api/sessions',headers=self.headers).status_code,429)
        self.assertEqual(self.client.get('/server.py').status_code,404)


if __name__=='__main__':unittest.main(verbosity=2)
