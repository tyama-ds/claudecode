"""Operator checks connecting the frozen claim guard and paired-evaluation records."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import replication as rep
import paired_trial as paired
import claim_guard as guard_cli


def runtime_check(root):
    protocol, hashes = paired.validate_preregistration(root)
    amendment = rep._read_json(root / 'code_alignment_amendment.json')
    checked = {}
    for name in ['factcheck/claim_guard.py','claim_guard.py','factcheck/repair.py','replication.py','paired_trial.py','paired_review.py']:
        snapshot = root / ('amendments/01/' + name if name == amendment['runtime_path'] else 'code_snapshots/' + name)
        actual = rep._sha(Path(name))
        if actual != rep._sha(snapshot):
            raise ValueError('Runtime differs from recorded snapshot: ' + name)
        checked[name] = actual
    if checked[amendment['runtime_path']] != amendment['new_sha256']:
        raise ValueError('Amended implementation hash mismatch')
    return protocol, checked


def doc_info(root, identity):
    protocol, hashes = runtime_check(root)
    doc = next(d for d in protocol['documents'] if d['id'] == identity)
    mapping = paired._arm_mapping(protocol, doc)
    improved = next(a for a,v in mapping.items() if v == 'improved')
    return doc, root / doc['path'], improved, hashes


def execution_budget(record, limit, label):
    rep._execution(record, label)
    count = record['execution']['web_calls']
    if count > limit:
        raise ValueError(f'{label} exceeds frozen Web call cap: {count}>{limit}')
    return count


def guard_args(folder, arm):
    base = folder / 'arms' / arm
    return ['--original',str(folder/'input/original_report.json'),
            '--candidate',str(base/'candidate.raw.json'),'--ledger',str(base/'ledger.raw.json')]


def prepare_guard(root, identity):
    doc, folder, arm, _ = doc_info(root, identity)
    base = folder/'arms'/arm
    execution_budget(rep._read_json(base/'candidate.raw.json'),18,'candidate')
    return guard_cli.main(['prepare',*guard_args(folder,arm),'--output-dir',str(base/'guard_input')])


def apply_guard(root, identity):
    doc, folder, arm, hashes = doc_info(root, identity)
    base = folder/'arms'/arm
    candidate_calls=execution_budget(rep._read_json(base/'candidate.raw.json'),18,'candidate')
    guard_calls=execution_budget(rep._read_json(base/'guard.raw.json'),7,'guard')
    code=guard_cli.main(['apply',*guard_args(folder,arm),'--guard-packet',str(base/'guard_input/guard_packet.json'),
                         '--guard',str(base/'guard.raw.json'),'--output-dir',str(base/'gated')])
    if code:
        raise ValueError('Gate failed')
    effective=base/'gated/repair.effective.json'
    linkage_path=base/'gate_linkage.json'
    recorded=(rep._read_json(linkage_path)['recorded_at_utc'] if linkage_path.exists()
              else datetime.now(timezone.utc).isoformat())
    rep._write_immutable({base/'repair.raw.json':effective.read_bytes(),
                         linkage_path:rep._json_bytes({'recorded_at_utc':recorded,
                          'effective_repair_sha256':rep._sha(effective),'scored_repair_sha256':rep._sha(effective),
                          'candidate_web_calls':candidate_calls,'guard_web_calls':guard_calls,'runtime_sha256':hashes})})
    return {'document':identity,'arm':arm,'candidate_calls':candidate_calls,'guard_calls':guard_calls}


def freeze_baseline(root, identity):
    _,folder,_,_=doc_info(root,identity)
    path=folder/'evaluation/baseline.raw.json'
    baseline=rep._read_json(path)
    rep.validate_baseline(rep._read_json(folder/'input/original_report.json'),baseline)
    execution_budget(baseline,25,'baseline')
    target=folder/'evaluation/baseline.freeze.json'
    if not target.exists():
        rep._write_immutable({target:rep._json_bytes({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'sha256':rep._sha(path)})})
    rep._baseline_freeze(folder)
    return {'document':identity,'baseline_sha256':rep._sha(path)}


def verify_gate(folder, arm):
    base=folder/'arms'/arm
    if (base/'repair.raw.json').read_bytes() != (base/'gated/repair.effective.json').read_bytes():
        raise ValueError('Scored repair differs from gated output')
    execution_budget(rep._read_json(base/'candidate.raw.json'),18,'candidate')
    execution_budget(rep._read_json(base/'guard.raw.json'),7,'guard')
    # Recompute semantics-free selection and hashes; no model judgment is rewritten.
    from factcheck.claim_guard import gate_repair
    inputs=[rep._read_json(p) for p in [folder/'input/original_report.json',base/'candidate.raw.json',base/'ledger.raw.json',base/'guard.raw.json',base/'guard_input/guard_packet.json']]
    recalculated=gate_repair(*inputs[:3],inputs[3],guard_packet=inputs[4])
    if recalculated != rep._read_json(base/'gated/gate.decisions.json'):
        raise ValueError('Gate decision replay mismatch')
    return recalculated['summary']


def prepare_post(root, identity):
    doc,folder,improved,_=doc_info(root,identity)
    verify_gate(folder,improved)
    freeze_baseline(root,identity)
    mapping=paired._arm_mapping(rep._read_json(root/'protocol.json'),doc)
    for arm in doc['arms']:
        if mapping[arm]=='current':
            execution_budget(rep._read_json(folder/'arms'/arm/'repair.raw.json'),25,'current repair')
        paired.prepare(root,identity,arm)
    return paired.prepare_post(root,identity)


def manifest(root):
    protocol,hashes=runtime_check(root)
    docs=[]
    for doc in protocol['documents']:
        identity=doc['id']
        _,folder,arm,_=doc_info(root,identity)
        gate=verify_gate(folder,arm)
        # Strictly require joint evidence, even though legacy reporting supports solo audits.
        for name in ['paired_post.raw.json','paired_post.freeze.json','paired_post.ingest.json']:
            if not (folder/'evaluation'/name).exists():
                raise ValueError(f'{identity}: missing joint audit {name}')
        paired.ingest_post(root,identity)
        execution_budget(rep._read_json(folder/'evaluation/paired_post.raw.json'),25,'joint post')
        row={'id':identity,'gate_summary':gate,'original_sha256':rep._sha(folder/'input/original_report.json'),
             'baseline_sha256':rep._sha(folder/'evaluation/baseline.raw.json'),
             'joint_response_sha256':rep._sha(folder/'evaluation/paired_post.raw.json'),
             'joint_ingest_sha256':rep._sha(folder/'evaluation/paired_post.ingest.json'),
             'effective_equals_scored':True,'joint_audit_verified':True}
        originals=rep._read_json(folder/'input/original_report.json')
        exports={folder/'input/original_report.txt':originals['text'].encode('utf-8')}
        for arm_name in doc['arms']:
            revised=rep._read_json(folder/'arms'/arm_name/'revised_report.json')
            exports[folder/'arms'/arm_name/'revised_report.txt']=revised['text'].encode('utf-8')
        rep._write_immutable(exports)
        docs.append(row)
    existing=root/'operator_manifest.json'
    created=(rep._read_json(existing)['created_at_utc'] if existing.exists()
             else datetime.now(timezone.utc).isoformat())
    value={'created_at_utc':created,'runtime_sha256':hashes,
           'operator_helper_sha256':rep._sha(Path(__file__)),
           'preregistration_sha256':rep._sha(root/'preregistration.json'),
           'amendment_sha256':rep._sha(root/'code_alignment_amendment.json'),'documents':docs,
           'limits':['Hashes establish content identity, not factual correctness or actual isolation.','Web call counts are worker execution attestations; parent tool logs provide additional evidence.']}
    rep._write_immutable({root/'operator_manifest.json':rep._json_bytes(value)})
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare-guard','apply-guard','freeze-baseline','prepare-post','manifest'])
    parser.add_argument('--root',type=Path,default=Path('pilot/claim_guard_20261102'))
    parser.add_argument('--doc')
    args=parser.parse_args()
    functions={'prepare-guard':prepare_guard,'apply-guard':apply_guard,'freeze-baseline':freeze_baseline,'prepare-post':prepare_post}
    value=manifest(args.root) if args.command=='manifest' else functions[args.command](args.root,args.doc)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
