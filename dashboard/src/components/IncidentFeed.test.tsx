import { act, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { feed } from "../lib/incidents";
import { createLiveStore } from "../store/live";
import { env, incident, summary } from "../test/fixtures";
import { IncidentFeed } from "./IncidentFeed";

function renderFeed(store = createLiveStore()) {
  const view = () => (
    <MemoryRouter>
      <IncidentFeed
        incidents={feed(store.getState().incidents, 40)}
        summaries={store.getState().summaries}
        cameraNames={{ "cam-hyde-park-corner": "Hyde Park Corner" }}
      />
    </MemoryRouter>
  );
  const utils = render(view());
  return { store, rerender: () => utils.rerender(view()) };
}

describe("IncidentFeed", () => {
  it("shows an empty state", () => {
    renderFeed();
    expect(screen.getByText(/No incidents yet/)).toBeInTheDocument();
  });

  it("updates a row in place when a newer version arrives", () => {
    const { store, rerender } = renderFeed();
    act(() => store.getState().applyBatch([env("incidents", incident({ version: 1, severity: "medium" }))]));
    rerender();
    const before = screen.getAllByTestId("incident-row");
    expect(before).toHaveLength(1);
    expect(within(before[0]).getByText("Suspected")).toBeInTheDocument();
    expect(within(before[0]).getByText("Medium")).toBeInTheDocument();
    // First render does not flash.
    expect(before[0].querySelector(".animate-flash")).toBeNull();

    act(() =>
      store.getState().applyBatch([
        env("incidents", incident({ version: 2, status: "confirmed", severity: "critical" })),
        env("summaries", summary({ incident_version: 2, headline: "Critical accident at Hyde Park Corner, confirmed" })),
      ]),
    );
    rerender();
    const after = screen.getAllByTestId("incident-row");
    expect(after).toHaveLength(1);
    expect(after[0]).toHaveAttribute("data-version", "2");
    expect(within(after[0]).getByText("Confirmed")).toBeInTheDocument();
    expect(within(after[0]).getByText("Critical")).toBeInTheDocument();
    expect(within(after[0]).getByTestId("incident-headline")).toHaveTextContent("Critical accident at Hyde Park Corner, confirmed");
    expect(after[0].querySelector(".animate-flash")).not.toBeNull();
  });

  it("ignores an out-of-order older version", () => {
    const { store, rerender } = renderFeed();
    act(() => store.getState().applyBatch([env("incidents", incident({ version: 3, status: "confirmed" }))]));
    act(() => store.getState().applyBatch([env("incidents", incident({ version: 2, status: "suspected" }))]));
    rerender();
    const row = screen.getByTestId("incident-row");
    expect(row).toHaveAttribute("data-version", "3");
    expect(within(row).getByText("Confirmed")).toBeInTheDocument();
  });

  it("puts new incidents on top and keeps existing rows", () => {
    const { store, rerender } = renderFeed();
    act(() => store.getState().applyBatch([env("incidents", incident({ incident_id: "a", suspected_at: "2026-10-17T09:00:00Z" }))]));
    rerender();
    act(() =>
      store.getState().applyBatch([env("incidents", incident({ incident_id: "b", suspected_at: "2026-10-17T09:05:00Z", title: "Breakdown on Old Street" }))]),
    );
    rerender();
    const rows = screen.getAllByTestId("incident-row");
    expect(rows.map((r) => r.getAttribute("data-incident"))).toEqual(["b", "a"]);
  });

  it("highlights the incident produced by a demo run", () => {
    const run = { scenarioId: "s1", scenario: "multi_vehicle_accident", cameraId: "cam-hyde-park-corner", timeScale: 6, triggeredAt: 0, incidentId: "inc-1", linked: true };
    render(
      <MemoryRouter>
        <IncidentFeed incidents={[incident()]} summaries={{}} cameraNames={{}} demoByIncident={new Map([["inc-1", run]])} />
      </MemoryRouter>,
    );
    expect(screen.getByText(/demo: multi_vehicle_accident/)).toBeInTheDocument();
    expect(screen.getByTestId("incident-row").className).toContain("ring-accent");
  });
});
