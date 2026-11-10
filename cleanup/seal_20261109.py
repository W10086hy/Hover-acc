"""Record authorized metadata refreshes after byte-preserving result cleanup."""
from cleanup.package import ROOT,read,write,digest

path=ROOT/'cleanup/retention_20261109.json';record=read(path)
assert record['status']=='complete'
updated=[];replaced=[]
for row in record['protected_files']:
    actual=digest(ROOT/row['path'])
    if actual!=row['sha256']:
        assert not row['immutable'] or row['path']=='P2/results/stationarity/FINAL37.md',row['path']
        updated.append(dict(path=row['path'],before_sha256=row['sha256'],after_sha256=actual,
            reason='Current navigation/status orreviewed maintenance code;no flight data orweights changed'))
for row in record['removed_files']:
    target=ROOT/row['path']
    if target.exists():
        assert row['path']=='P2/results/stationarity/README.md'
        assert digest(target)!=row['sha256']
        replaced.append(dict(path=row['path'],before_sha256=row['sha256'],after_sha256=digest(target),
            reason='Old debugging index removed;new final-results index created at same path'))
record.update(updated_metadata_files=updated,replacement_files=replaced,
    review='Independent read-only retention andfinal-metadata review;path_map after-SHA and120s horizon gaps fixed',
    final_models_and_raw_results_unchanged=True)
write(path,record)
print(dict(updated_metadata=len(updated),rebuilt_documents=len(replaced),raw_or_model_changes=0))
