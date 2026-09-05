//! Minimal QMP client used to hot-unplug vfio-pci devices before shutdown.
//!
//! Tearing down a TDX guest serialises the GPU release behind the TD's private
//! memory reclaim, which takes minutes on a large guest. Asking QEMU to detach
//! the vfio devices first hands the cards back to the host while that reclaim
//! continues in the dying process.
//!
//! The unplug is guest-cooperative: `device_del` raises an ACPI eject request
//! and the guest decides when to release the device. Every call here is
//! therefore bounded and best-effort — a failure falls back to the plain
//! SIGTERM path, which is exactly what happened before this module existed.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{unix::OwnedWriteHalf, UnixStream};
use tokio::time::timeout;
use tracing::{debug, warn};

/// Device id prefix stamped onto every vfio-pci device in `configure_gpus`.
const VFIO_ID_PREFIX: &str = "vfio_";

type Reader = BufReader<tokio::net::unix::OwnedReadHalf>;

/// Detach every vfio-pci device from the running QEMU and wait for the guest
/// to acknowledge. Returns the number of devices confirmed removed.
pub(crate) async fn detach_vfio_devices(socket: &Path, budget: Duration) -> Result<usize> {
    timeout(budget, detach_inner(socket))
        .await
        .map_err(|_| anyhow::anyhow!("timed out after {budget:?}"))?
}

async fn detach_inner(socket: &Path) -> Result<usize> {
    let stream = UnixStream::connect(socket)
        .await
        .with_context(|| format!("failed to connect to QMP socket {}", socket.display()))?;
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    // QEMU speaks first with its greeting banner.
    let greeting = read_message(&mut reader).await?;
    if greeting.get("QMP").is_none() {
        bail!("unexpected QMP greeting: {greeting}");
    }
    execute(&mut writer, &mut reader, "qmp_capabilities", Value::Null).await?;

    let pci = execute(&mut writer, &mut reader, "query-pci", Value::Null).await?;
    let ids = collect_vfio_ids(&pci);
    if ids.is_empty() {
        debug!("no vfio-pci devices to detach");
        return Ok(0);
    }

    let mut requested = BTreeSet::new();
    for id in &ids {
        match execute(&mut writer, &mut reader, "device_del", json!({ "id": id })).await {
            Ok(_) => {
                requested.insert(id.clone());
            }
            // A device that is already gone is not an error worth aborting on.
            Err(error) => warn!(%id, %error, "device_del failed"),
        }
    }
    if requested.is_empty() {
        bail!("no device_del command was accepted");
    }
    wait_for_deleted(&mut reader, requested).await
}

/// `query-pci` nests devices under bridges, so walk the tree.
fn collect_vfio_ids(response: &Value) -> Vec<String> {
    fn walk(node: &Value, out: &mut Vec<String>) {
        if let Some(id) = node.get("qdev_id").and_then(Value::as_str) {
            if id.starts_with(VFIO_ID_PREFIX) {
                out.push(id.to_string());
            }
        }
        for key in ["devices", "pci_bridge", "bus"] {
            match node.get(key) {
                Some(Value::Array(children)) => children.iter().for_each(|c| walk(c, out)),
                Some(child @ Value::Object(_)) => walk(child, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    if let Some(buses) = response.as_array() {
        buses.iter().for_each(|bus| walk(bus, &mut out));
    }
    out.sort();
    out.dedup();
    out
}

/// Drain events until every requested device reports DEVICE_DELETED.
async fn wait_for_deleted(reader: &mut Reader, mut pending: BTreeSet<String>) -> Result<usize> {
    let total = pending.len();
    while !pending.is_empty() {
        let message = read_message(reader).await?;
        if message.get("event").and_then(Value::as_str) != Some("DEVICE_DELETED") {
            continue;
        }
        // QEMU emits DEVICE_DELETED twice: once for the device, once for its
        // qom path with no `device` field. Only the former interests us.
        if let Some(device) = message.pointer("/data/device").and_then(Value::as_str) {
            pending.remove(device);
        }
    }
    Ok(total)
}

async fn execute(
    writer: &mut OwnedWriteHalf,
    reader: &mut Reader,
    command: &str,
    arguments: Value,
) -> Result<Value> {
    let mut request = json!({ "execute": command });
    if !arguments.is_null() {
        request["arguments"] = arguments;
    }
    let mut line = serde_json::to_vec(&request)?;
    line.push(b'\n');
    writer
        .write_all(&line)
        .await
        .with_context(|| format!("failed to send QMP command {command}"))?;
    writer
        .flush()
        .await
        .context("failed to flush QMP command")?;

    loop {
        let message = read_message(reader).await?;
        if let Some(error) = message.get("error") {
            bail!("QMP command {command} failed: {error}");
        }
        if let Some(value) = message.get("return") {
            return Ok(value.clone());
        }
        // Anything else is an asynchronous event; keep reading for the reply.
    }
}

async fn read_message(reader: &mut Reader) -> Result<Value> {
    let mut line = String::new();
    let read = reader
        .read_line(&mut line)
        .await
        .context("failed to read from QMP socket")?;
    if read == 0 {
        bail!("QMP socket closed");
    }
    serde_json::from_str(&line).with_context(|| format!("invalid QMP message: {}", line.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_nested_vfio_ids_and_ignores_others() {
        let pci = json!([{
            "bus": 0,
            "devices": [
                { "qdev_id": "virtio-blk0" },
                {
                    "qdev_id": "pci.1",
                    "pci_bridge": {
                        "devices": [{ "qdev_id": "vfio_gpu1" }]
                    }
                },
                { "qdev_id": "vfio_gpu2" }
            ]
        }]);
        assert_eq!(collect_vfio_ids(&pci), vec!["vfio_gpu1", "vfio_gpu2"]);
    }

    #[test]
    fn no_vfio_devices_yields_empty() {
        let pci = json!([{ "bus": 0, "devices": [{ "qdev_id": "virtio-net0" }] }]);
        assert!(collect_vfio_ids(&pci).is_empty());
    }
}
