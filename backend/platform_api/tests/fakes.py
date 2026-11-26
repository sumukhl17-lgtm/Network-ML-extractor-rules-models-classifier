"""A fake MLflow client with the same method names and return shapes the real MlflowClient uses
(search_registered_models, search_model_versions, get_model_version_by_alias,
set_registered_model_alias, get_run, get_experiment_by_name, search_runs)."""
from types import SimpleNamespace as NS


class FakeMlflowClient:
    def __init__(self):
        self.versions = {}   # name -> {version(str): NS}
        self.aliases = {}    # name -> {alias: version}
        self.run_metrics = {}
        self.experiments = {}   # name -> (NS experiment, [runs])
        self.alias_calls = []

    def add_version(self, name, version, metrics=None, created_ms=1_700_000_000_000):
        run_id = "run-{}-{}".format(name, version)
        self.versions.setdefault(name, {})[str(version)] = NS(
            version=str(version), run_id=run_id, creation_timestamp=created_ms + int(version), status="READY")
        self.run_metrics[run_id] = metrics or {}
        return run_id

    def search_registered_models(self, *a, **k):
        return [NS(name=n) for n in self.versions]

    def search_model_versions(self, filter_string=None, *a, **k):
        name = filter_string.split("'")[1]
        return list(self.versions.get(name, {}).values())

    def get_model_version_by_alias(self, name, alias):
        version = self.aliases.get(name, {}).get(alias)
        if version is None:
            raise Exception("RESOURCE_DOES_NOT_EXIST: alias {} not found".format(alias))
        return self.versions[name][version]

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, str(version)))
        self.aliases.setdefault(name, {})[alias] = str(version)

    def get_run(self, run_id):
        if run_id not in self.run_metrics:
            raise Exception("run not found")
        return NS(data=NS(metrics=self.run_metrics[run_id]))

    def add_experiment(self, name, runs):
        self.experiments[name] = (NS(experiment_id="1"), runs)

    def get_experiment_by_name(self, name):
        return self.experiments.get(name, (None, []))[0]

    def search_runs(self, experiment_ids, order_by=None, max_results=50, **k):
        for exp, runs in self.experiments.values():
            if exp.experiment_id in experiment_ids:
                return runs[:max_results]
        return []


def make_run(run_id, name, metrics, params=None, started=1_700_000_000_000):
    return NS(info=NS(run_id=run_id, run_name=name, start_time=started, status="FINISHED"),
              data=NS(metrics=metrics, params=params or {}, tags={}))
