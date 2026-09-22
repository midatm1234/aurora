"""Human CLI; scientific execution shares the exact approved MCP backend."""
import argparse
import json
from pathlib import Path
import sys
from .common import REPO, atomic_json, read_json, redact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="workflow.local.json")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure")
    setup.add_argument("--workspace", required=True)
    commands.add_parser("doctor").add_argument("--check-torch",action="store_true")
    commands.add_parser("inspect")
    p = commands.add_parser("plan")
    p.add_argument("--request")
    p.add_argument("--recipe",default="cpu-smoke-v1")
    p.add_argument("--head")
    p.add_argument("--mode")
    p.add_argument("--case-id",default="reproduction")
    commands.add_parser("approve").add_argument("plan_id")
    p = commands.add_parser("execute")
    p.add_argument("plan_id")
    p.add_argument("--stage",default="workflow")
    for action in ("status","logs","cancel","recover"):
        commands.add_parser(action).add_argument("job_id")
    commands.add_parser("knowledge-validate")
    args = parser.parse_args()
    try:
        if args.command == "configure":
            workspace = Path(args.workspace).expanduser().resolve()
            config = Path(args.config)
            if config.exists():
                raise ValueError("Local config exists; inspect/edit it without overwriting")
            from .settings import Settings
            settings = Settings(data_root=workspace/"data",cache_root=workspace/"cache",
                                output_root=workspace/"runs",state_root=workspace/"state")
            atomic_json(config, settings.model_dump(mode="json"))
            result = {"state":"configured","config":str(config.resolve())}
        elif args.command == "knowledge-validate":
            from .knowledge import validate_bundle
            result = validate_bundle(REPO/"knowledge")
        else:
            from .jobs import JobStore
            store = JobStore(args.config)
            if args.command == "doctor":
                from .backend import doctor
                result = doctor(store.settings,check_torch=args.check_torch)
            elif args.command == "inspect":
                from .assets import inspect_assets
                result = {**inspect_assets(store.settings.assets),**store.list_jobs()}
            elif args.command == "plan":
                from .planning import PlanRequest, make_plan
                request = read_json(args.request) if args.request else {
                    "recipe":args.recipe,"head":args.head,"mode":args.mode,"case_id":args.case_id}
                result = store.save_plan(make_plan(PlanRequest.model_validate(request),store.settings))
            elif args.command == "approve":
                result = store.approve_in_terminal(args.plan_id)
            elif args.command == "execute":
                result = store.submit(args.plan_id,args.stage)
            else:
                result = getattr(store,args.command)(args.job_id)
        print(json.dumps(redact(result),indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps({"state":"failed","errors":[{"type":type(exc).__name__,"message":redact(str(exc))}]}),file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
