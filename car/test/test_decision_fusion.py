"""Stable local/cloud interface and independent output-contract checks; no devices."""
from dataclasses import replace
import unittest
from car.test.test_cloud_recovery import evidence, STOP, FORWARD
from car.test.test_cloud_recovery_contract import recovery
from autodrive.control.cloud_recovery import RecoveryConfig
from autodrive.control.decision_fusion import (
    FusionInput, FusionDecision, CloudAdvice, DecisionFusionAdapter, RuleBasedRecoveryPolicy)


class FusionTests(unittest.TestCase):
    def adapter(self):
        return DecisionFusionAdapter(RuleBasedRecoveryPolicy(RecoveryConfig(enabled=True),'run'))

    def test_local_proposal_keeps_identity_and_expiry(self):
        adapter=self.adapter()
        decision=adapter.decide(FusionInput(FORWARD,evidence(1.,1),1.,1.15))
        self.assertEqual(decision.source,'local');self.assertEqual(decision.command,FORWARD)
        self.assertEqual(decision.deadline,1.15);self.assertEqual(decision.sequence,1)
        self.assertEqual(adapter.get_state()['fusion']['policy_id'],'rule-recovery-v1')

    def test_cloud_and_local_share_one_output_contract(self):
        adapter=self.adapter()
        for seq,now in enumerate((1.,1.5,2.1),1):
            decision=adapter.decide(FusionInput(STOP,evidence(now,seq),now,None))
        self.assertEqual(decision.source,'hold')
        event=adapter.pending_request['context']['event_id']
        adapter.receive(CloudAdvice(event,recovery(),2.2))
        decision=adapter.decide(FusionInput(STOP,evidence(2.3,4),2.3,None))
        self.assertEqual(decision.source,'cloud_recovery')
        self.assertEqual(decision.candidate_id,'forward-white');self.assertEqual(decision.event_id,event)
        self.assertLessEqual(decision.deadline,2.45)
        revoked=adapter.veto('operator cancelled',2.31)
        self.assertEqual(revoked.command.action,'stop');self.assertIsNone(revoked.deadline)

    def test_expired_local_deadline_and_hard_veto_stop(self):
        for deadline,hard in ((None,True),(.9,True),(1.15,False)):
            adapter=self.adapter();e=evidence(1.,1);e['hard_safe']=hard
            self.assertEqual(adapter.decide(FusionInput(FORWARD,e,1.,deadline)).command.action,'stop')

    def test_alternative_policy_must_obey_output_contract(self):
        class Alternative(RuleBasedRecoveryPolicy):
            policy_id='test-policy'
            def evaluate(self,inputs):
                return FusionDecision(FORWARD,'local','alternative',inputs.now+.15,None,None,
                                      inputs.evidence['observation']['sequence'],self.policy_id)
        adapter=DecisionFusionAdapter(Alternative(RecoveryConfig(enabled=True),'run'))
        # A policy may not invent a moving local proposal when the input is stop.
        rejected=adapter.decide(FusionInput(STOP,evidence(1.,1),1.,1.15))
        self.assertEqual(rejected.command.action,'stop')
        accepted=adapter.decide(FusionInput(FORWARD,evidence(1.2,2),1.2,1.35))
        self.assertEqual(accepted.policy_id,'test-policy');self.assertEqual(accepted.command,FORWARD)

    def test_policy_failure_and_unbacked_cloud_motion_fail_closed(self):
        class Broken(RuleBasedRecoveryPolicy):
            def evaluate(self,inputs):raise RuntimeError('not a user-visible diagnostic')
        adapter=DecisionFusionAdapter(Broken(RecoveryConfig(enabled=True),'run'))
        self.assertEqual(adapter.decide(FusionInput(FORWARD,evidence(1.,1),1.,1.15)).source,'hold')
        class Unbacked(RuleBasedRecoveryPolicy):
            def evaluate(self,inputs):
                return FusionDecision(FORWARD,'cloud_recovery','invented',inputs.now+.15,
                                      'fake','forward-white',1,self.policy_id)
        adapter=DecisionFusionAdapter(Unbacked(RecoveryConfig(enabled=True),'run'))
        self.assertEqual(adapter.decide(FusionInput(STOP,evidence(1.,1),1.,None)).source,'hold')

    def test_replacement_policy_cannot_renew_a_cloud_step_forever(self):
        class Renewable(RuleBasedRecoveryPolicy):
            def evaluate(self,inputs):
                if self.engine.phase in ('REVALIDATE','PROBE_STEP'):
                    event=self.pending_request['context']['event_id']
                    return FusionDecision(FORWARD,'cloud_recovery','renewed',inputs.now+.15,
                        event,'forward-white',inputs.evidence['observation']['sequence'],self.policy_id)
                return super().evaluate(inputs)
        adapter=DecisionFusionAdapter(Renewable(RecoveryConfig(enabled=True),'run'))
        for seq,now in enumerate((1.,1.5,2.1),1):adapter.decide(FusionInput(STOP,evidence(now,seq),now,None))
        event=adapter.pending_request['context']['event_id'];adapter.receive(CloudAdvice(event,recovery(),2.2))
        first=adapter.decide(FusionInput(STOP,evidence(2.3,4),2.3,None))
        second=adapter.decide(FusionInput(STOP,evidence(2.4,5),2.4,None))
        self.assertEqual(first.deadline,second.deadline)
        self.assertEqual(adapter.decide(FusionInput(STOP,evidence(2.46,6),2.46,None)).source,'hold')

    def test_policy_can_be_replaced_without_inheriting_recovery_engine(self):
        class HoldPolicy:
            policy_id='independent-hold-policy'
            pending_request=None
            def receive(self,advice):pass
            def evaluate(self,inputs):
                return FusionDecision(STOP,'hold','test hold',None,None,None,
                                      inputs.evidence['observation']['sequence'],self.policy_id)
            def veto(self,reason,now):pass
            def get_state(self):return dict(phase='HOLD',event_id=None,reason='test hold')
            def drain_transitions(self):return []
        adapter=DecisionFusionAdapter(HoldPolicy())
        decision=adapter.decide(FusionInput(FORWARD,evidence(1.,1),1.,1.15))
        self.assertEqual(decision.policy_id,'independent-hold-policy')
        self.assertEqual(decision.command.action,'stop')
        self.assertEqual(adapter.get_state()['phase'],'HOLD')


if __name__=='__main__':unittest.main()
