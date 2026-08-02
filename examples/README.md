# Examples

These examples are data only. They do not download a model, execute a candidate, or require MLX.

```bash
PYTHONPATH=src python3 -m auto_mlx validate workload --input examples/workload.json
PYTHONPATH=src python3 -m auto_mlx inspect provider --input examples/provider.json
```

The provider configurations are proposals for the declared knobs. A future evaluator would derive candidate IDs only after binding one configuration to the workload; no candidate ID is stored in the provider document.
