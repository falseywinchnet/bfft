"""Fit all available surface observations, retaining separate prediction evidence.

An all-view fit is an estimate, not a new heldout evaluation. Previously blocked
predictions are attached only after optimization and cannot select its inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from .spatial_surface_study import initialize,FOLDS
from .jet_bundle import refine_jets
from .scene_promotion import audit


def attach_predictions(candidate,folder):
    predictions=defaultdict(list)
    for fold in range(FOLDS):
        for row in json.loads((folder/f'{fold}-coherent.predictions.json').read_text()):predictions[row['track']].append(row)
    for j in candidate['surface_jets']:
        rows=predictions[j['track']];expected={o['capture'] for o in j['observations']}
        if len(rows)!=len(expected) or {r['capture'] for r in rows}!=expected:raise ValueError('Missing or duplicated blocked prediction evidence')
        passed=len(rows)>=3 and all(r['resolved'] and r['passed'] for r in rows)
        j['prediction_passed']=bool(passed);j['accepted']=j['training_supported'] and passed
        j['blocked_prediction']=dict(observations=len(rows),resolved=sum(r['resolved'] for r in rows),passed=sum(r.get('passed',False) for r in rows))
    return candidate


def main():
    p=argparse.ArgumentParser()
    for n in ('evidence','pose','jets','study','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--joint-cameras',action='store_true');p.add_argument('--max-evaluations',type=int,default=500);a=p.parse_args()
    e=json.loads(a.evidence.read_text());pose=json.loads(a.pose.read_text());packet=json.loads(a.jets.read_text());marked=copy.deepcopy([j for j in packet['results'] if 'observations' in j])
    source_receipt=json.loads((a.study/'0-coherent.receipt.json').read_text())
    if bool(source_receipt.get('joint_cameras',False))!=a.joint_cameras:raise ValueError('Prediction study has a different camera policy')
    expected_inputs=set(source_receipt['inputs_sha256'].values())
    if any(hashlib.sha256(p.read_bytes()).hexdigest() not in expected_inputs for p in (a.evidence,a.pose,a.jets)):raise ValueError('Prediction study has different inputs')
    for j in marked:
        for o in j['observations']:o.update(training=True,validation=False,guarded=False)
    cameras=pose.get('camera_models',e['cameras']);prepared,initialization=initialize(marked,pose,cameras)
    _,receipt,candidate=refine_jets(prepared,pose,cameras,max_evaluations=a.max_evaluations,coherent_surfaces=True,freeze_cameras=not a.joint_cameras)
    candidate=attach_predictions(candidate,a.study);candidate=audit(e,candidate)
    candidate['full_field']=dict(initialization=initialization,training_supported=sum(j['training_supported'] for j in candidate['surface_jets']),prediction_passed=sum(j['accepted'] for j in candidate['surface_jets']),
        policy='All observations fitted once. Prediction evidence comes from the matched four-fold guarded study and is attached after fitting; it never selects optimization inputs. Stored camera initialization and discovered maps are conditional. This is an experimental field, not an accepted complete scene.',
        sources_sha256={n:hashlib.sha256(Path(__file__).with_name(n+'.py').read_bytes()).hexdigest() for n in ('full_surface_field','spatial_surface_study','surface_jets','jet_bundle','surface_coherence','scene_promotion')},promoted=False,study_summary_sha256=hashlib.sha256((a.study/'summary.json').read_bytes()).hexdigest(),parent_pose_sha256=hashlib.sha256(a.pose.read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(candidate));print(json.dumps(dict(cameras=len(candidate['poses']),points=len(candidate['points']),training_supported=candidate['full_field']['training_supported'],prediction_passed=candidate['full_field']['prediction_passed'],evaluations=receipt['evaluations'],status=receipt['status'],seconds=receipt['seconds'])))


if __name__=='__main__':main()
