"""Finite, dependency-injected candidate playback recorder; no simulator import at module load."""
from __future__ import annotations
import json, time
class BoundedRecorder:
    def __init__(self, env, policy, output, episodes=1, horizon_s=3.0, seed=8):
        if episodes<1 or horizon_s<=0: raise ValueError('finite positive episode budget required')
        self.env,self.policy,self.output,self.episodes,self.horizon,self.seed=env,policy,output,episodes,horizon_s,seed
    def run(self):
        records=[]; outcomes=[]
        try:
            obs,_=self.env.reset(seed=self.seed)
            for ep in range(self.episodes):
                start=time.monotonic(); initial=self.env.snapshot(); records.append({'kind':'initial','episode':ep+1,'state':initial})
                done=False; steps=0
                while not done and time.monotonic()-start<self.horizon:
                    action=self.policy(obs)
                    obs,reward,terminated,truncated,info=self.env.step(action)
                    snap=self.env.snapshot(); steps+=1
                    records.append({'kind':'step','episode':ep+1,'step':steps,'state':snap,'reward_proxy':float(reward),'info':info})
                    done=bool(terminated or truncated)
                terminal=self.env.snapshot(); records.append({'kind':'terminal_pre_reset','episode':ep+1,'state':terminal})
                outcomes.append({'episode':ep+1,'status':'completed' if done else 'incomplete','steps':steps,'error':None})
                if ep+1<self.episodes: obs,_=self.env.reset()
        except Exception as exc:
            outcomes.append({'episode':len(outcomes)+1,'status':'error','error':f'{type(exc).__name__}: {exc}'})
        finally:
            self.output.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in records))
        return outcomes
