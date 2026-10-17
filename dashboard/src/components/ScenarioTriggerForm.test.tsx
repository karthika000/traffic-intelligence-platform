import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { ScenarioInfo } from "../api/types";
import { ScenarioTriggerForm } from "./ScenarioTriggerForm";

const scenarios: ScenarioInfo[] = [
  {
    name: "broken_down_lorry",
    description: "Broken-down lorry blocking a lane.",
    incident_type: "breakdown",
    duration_seconds: 960,
    phases: [{ name: "breakdown", duration_seconds: 180, severity: "medium" }],
  },
  {
    name: "multi_vehicle_accident",
    description: "Multi-vehicle collision.",
    incident_type: "accident",
    duration_seconds: 1080,
    phases: [
      { name: "first_reports", duration_seconds: 120, severity: "high" },
      { name: "escalation", duration_seconds: 240, severity: "critical" },
    ],
  },
];
const cameras = [
  { id: "cam-elephant-castle", name: "Elephant & Castle" },
  { id: "cam-hyde-park-corner", name: "Hyde Park Corner" },
];
const ok = (scenario: string, camera_id: string, time_scale: number) => ({
  scenario_id: "6f1c2a9e-0000-4000-8000-000000000000",
  scenario,
  camera_id,
  time_scale,
  status: "accepted",
});

describe("ScenarioTriggerForm", () => {
  it("defaults to the first scenario and camera", async () => {
    const onTrigger = vi.fn(async (i) => ok(i.scenario, i.camera_id, i.time_scale));
    render(<ScenarioTriggerForm scenarios={scenarios} cameras={cameras} onTrigger={onTrigger} />);
    await userEvent.click(screen.getByRole("button", { name: "Trigger scenario" }));
    expect(onTrigger).toHaveBeenCalledWith({ scenario: "broken_down_lorry", camera_id: "cam-elephant-castle", time_scale: 6 });
  });

  it("triggers the chosen scenario on the chosen camera and time scale", async () => {
    const onTrigger = vi.fn(async (i) => ok(i.scenario, i.camera_id, i.time_scale));
    render(<ScenarioTriggerForm scenarios={scenarios} cameras={cameras} onTrigger={onTrigger} />);
    await userEvent.selectOptions(screen.getByLabelText("Scenario"), "multi_vehicle_accident");
    await userEvent.selectOptions(screen.getByLabelText("Camera"), "cam-hyde-park-corner");
    await userEvent.selectOptions(screen.getByLabelText("Time scale"), "10");
    // Phase durations follow the time scale: 120 s / 10 = 12 s.
    expect(screen.getByText("first_reports 12 s")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Trigger scenario" }));
    expect(onTrigger).toHaveBeenCalledTimes(1);
    expect(onTrigger).toHaveBeenCalledWith({ scenario: "multi_vehicle_accident", camera_id: "cam-hyde-park-corner", time_scale: 10 });
    expect(await screen.findByRole("status")).toHaveTextContent("Started 6f1c2a9e");
  });

  it("disables the button while the request is in flight", async () => {
    let resolve!: (v: ReturnType<typeof ok>) => void;
    const onTrigger = vi.fn(() => new Promise<ReturnType<typeof ok>>((r) => (resolve = r)));
    render(<ScenarioTriggerForm scenarios={scenarios} cameras={cameras} onTrigger={onTrigger} />);
    await userEvent.click(screen.getByRole("button", { name: "Trigger scenario" }));
    const busy = screen.getByRole("button", { name: "Triggering…" });
    expect(busy).toBeDisabled();
    await userEvent.click(busy);
    expect(onTrigger).toHaveBeenCalledTimes(1);
    resolve(ok("broken_down_lorry", "cam-elephant-castle", 6));
    await waitFor(() => expect(screen.getByRole("button", { name: "Trigger scenario" })).toBeEnabled());
  });

  it("shows the API's error message", async () => {
    const onTrigger = vi.fn(async () => {
      throw new ApiError("camera cam-x not found", 404, "not_found");
    });
    render(<ScenarioTriggerForm scenarios={scenarios} cameras={cameras} onTrigger={onTrigger} />);
    await userEvent.click(screen.getByRole("button", { name: "Trigger scenario" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("camera cam-x not found");
  });

  it("can't be submitted before scenarios and cameras load", () => {
    render(<ScenarioTriggerForm scenarios={[]} cameras={[]} onTrigger={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Trigger scenario" })).toBeDisabled();
  });

  it("uses the given defaults", async () => {
    const onTrigger = vi.fn(async (i) => ok(i.scenario, i.camera_id, i.time_scale));
    render(
      <ScenarioTriggerForm
        scenarios={scenarios}
        cameras={cameras}
        onTrigger={onTrigger}
        defaults={{ scenario: "multi_vehicle_accident", camera_id: "cam-hyde-park-corner", time_scale: 6 }}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Trigger scenario" }));
    expect(onTrigger).toHaveBeenCalledWith({ scenario: "multi_vehicle_accident", camera_id: "cam-hyde-park-corner", time_scale: 6 });
  });
});
