# traffic-common

Shared Python package used by every service in `services/`.

| Module | Purpose |
| --- | --- |
| `traffic_common.config` | `Settings` loaded from environment variables |
| `traffic_common.logging` | Consistent log format across services |
| `traffic_common.lifecycle` | `run_forever()` keep-alive loop with graceful SIGTERM/SIGINT shutdown |

Install for local development (from the repo root):

```powershell
pip install -e "shared[dev]"
```
