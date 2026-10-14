"""Trigger or stop a scenario on the running simulator via the simulator.control topic.

    python -m event_simulator.trigger <scenario> <camera_id> [--time-scale N]
    python -m event_simulator.trigger --list
    python -m event_simulator.trigger --stop <scenario_id>

Inside Docker: docker compose exec event_simulator python -m event_simulator.trigger ...
"""

import argparse
import sys
import uuid

from event_simulator.config import SimulatorConfig
from event_simulator.scenarios import load_scenarios
from event_simulator.simulator import load_cameras
from traffic_common import get_settings
from traffic_common.db import create_db_engine
from traffic_common.messaging import TypedProducer
from traffic_common.schemas import SimulatorCommand
from traffic_common.topics import Topic


def send_command(command: SimulatorCommand, bootstrap_servers: str) -> None:
    with TypedProducer(
        bootstrap_servers, client_id="simulator-trigger", register_shutdown=False
    ) as p:
        p.send(Topic.SIMULATOR_CONTROL, command)
        if p.flush(10):
            raise RuntimeError(
                f"command not delivered to {Topic.SIMULATOR_CONTROL} within 10s; is Kafka up and "
                "does the topic exist? (.\\scripts\\create_topics.ps1)"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m event_simulator.trigger", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    parser.add_argument("scenario", nargs="?")
    parser.add_argument("camera_id", nargs="?")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="run the timeline N times faster (default 1 = real time)",
    )
    parser.add_argument("--list", action="store_true", help="list scenarios and cameras")
    parser.add_argument("--stop", metavar="SCENARIO_ID", type=uuid.UUID)
    args = parser.parse_args(argv)

    settings = get_settings("event_simulator.trigger")
    scenarios = load_scenarios(SimulatorConfig.from_env().scenarios_dir)
    try:
        cameras = {c.id: c for c in load_cameras(create_db_engine(settings.postgres_url))}
    except Exception as exc:
        cameras = None
        print(f"warning: could not read cameras from Postgres ({exc}); not validating camera_id")

    if args.list:
        print("Scenarios:")
        for s in scenarios.values():
            print(
                f"  {s.name:<26} {s.incident_type.value:<15} peak {s.peak_severity.value:<9}"
                f"{s.duration / 60:5.1f} min  {len(s.phases)} phases"
            )
        if cameras is not None:
            print("Cameras:")
            for c in cameras.values():
                print(f"  {c.id:<26} {c.location_name}")
        return 0

    if args.stop:
        send_command(
            SimulatorCommand(action="stop_scenario", scenario_id=args.stop),
            settings.kafka_bootstrap_servers,
        )
        print(f"stop requested for scenario {args.stop}")
        return 0

    if not args.scenario or not args.camera_id:
        parser.error("give <scenario> <camera_id>, or use --list / --stop")
    if args.scenario not in scenarios:
        parser.error(
            f"unknown scenario {args.scenario!r}; available: {', '.join(sorted(scenarios))}"
        )
    if cameras is not None and args.camera_id not in cameras:
        parser.error(f"unknown camera {args.camera_id!r}; available: {', '.join(sorted(cameras))}")

    scenario = scenarios[args.scenario]
    command = SimulatorCommand(
        action="start_scenario", scenario=args.scenario, camera_id=args.camera_id,
        time_scale=args.time_scale, requested_by="cli",
    )  # fmt: skip
    send_command(command, settings.kafka_bootstrap_servers)
    print(f"scenario_id: {command.scenario_id}")
    print(
        f"{scenario.name} on {args.camera_id}: {len(scenario.phases)} phases, "
        f"{scenario.duration / args.time_scale / 60:.1f} min at time scale {args.time_scale:g}"
    )
    print("Follow it:  docker compose logs -f event_simulator")
    print(f"Stop it:    python -m event_simulator.trigger --stop {command.scenario_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
