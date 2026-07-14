import { useEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Radio, RefreshCw } from "lucide-react"
import { toast } from "sonner"

import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { api, errorMessage } from "@/lib/api"

interface MonitorStatus {
  schema_version: "monitor_webrtc.v1"
  transport: "webrtc"
  status: string
  signaling_ready: boolean
  peer_count: number
  frame_count: number
  selected_node: { path?: string } | null
  error: string | null
}

interface MonitorPayload {
  job: { id: string; status: string; message?: string | null } | null
  webrtc_status: MonitorStatus | null
}

interface SessionDescriptionPayload {
  type: RTCSdpType
  sdp: string
}

const TERMINAL_JOB_STATUSES = new Set(["failed", "canceled", "succeeded"])

function waitForIceGatheringComplete(peer: RTCPeerConnection): Promise<void> {
  if (peer.iceGatheringState === "complete") return Promise.resolve()
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", changed)
      reject(new Error("Timed out while gathering WebRTC host candidates"))
    }, 10_000)
    const changed = () => {
      if (peer.iceGatheringState !== "complete") return
      window.clearTimeout(timeout)
      peer.removeEventListener("icegatheringstatechange", changed)
      resolve()
    }
    peer.addEventListener("icegatheringstatechange", changed)
  })
}

function stopPeer(peer: RTCPeerConnection | null, video: HTMLVideoElement | null) {
  peer?.close()
  const stream = video?.srcObject
  if (stream instanceof MediaStream) stream.getTracks().forEach((track) => track.stop())
  if (video) video.srcObject = null
}

function preferBrowserVp8(transceiver: RTCRtpTransceiver) {
  const codecs = RTCRtpSender.getCapabilities("video")?.codecs
  if (!codecs) return
  transceiver.setCodecPreferences([
    ...codecs.filter((codec) => codec.mimeType.toLowerCase() === "video/vp8"),
    ...codecs.filter((codec) => codec.mimeType.toLowerCase() !== "video/vp8"),
  ])
}

export function RoomMonitor() {
  const queryClient = useQueryClient()
  const videoRef = useRef<HTMLVideoElement>(null)
  const peerRef = useRef<RTCPeerConnection | null>(null)
  const monitorStartAttempted = useRef(false)
  const automaticRenegotiationUsed = useRef(false)
  const previousJobId = useRef<string | null>(null)
  const negotiationSequence = useRef(0)
  const [connection, setConnection] = useState({ jobId: null as string | null, status: "waiting" })
  const [negotiationVersion, setNegotiationVersion] = useState(0)

  const monitor = useQuery({
    queryKey: ["monitor"],
    queryFn: () => api<MonitorPayload>("/monitoring/webcam"),
    refetchInterval: (query) => {
      const currentJob = query.state.data?.job
      const connected = connection.jobId === (currentJob?.id ?? null)
        && connection.status === "connected"
        && currentJob?.status === "running"
      return connected ? 5_000 : 1_000
    },
  })
  const startMonitor = useMutation({
    mutationFn: () => api<MonitorPayload>("/monitoring/webcam", { method: "POST", body: "{}" }),
    onSuccess: (data) => queryClient.setQueryData(["monitor"], data),
    onError: (error) => toast.error("Room monitor could not start", { description: errorMessage(error) }),
  })

  const jobId = monitor.data?.job?.id ?? null
  const jobStatus = monitor.data?.job?.status ?? null
  const signalingReady = monitor.data?.webrtc_status?.signaling_ready === true
  const connectionStatus = connection.jobId === jobId ? connection.status : "waiting"

  useEffect(() => {
    if (!monitor.isSuccess || monitorStartAttempted.current || startMonitor.isPending) return
    if (!jobStatus || TERMINAL_JOB_STATUSES.has(jobStatus)) {
      monitorStartAttempted.current = true
      startMonitor.mutate()
    }
  }, [monitor.isSuccess, jobStatus, startMonitor])

  useEffect(() => {
    if (previousJobId.current === jobId) return
    previousJobId.current = jobId
    automaticRenegotiationUsed.current = false
  }, [jobId])

  useEffect(() => {
    if (!jobId || jobStatus !== "running" || !signalingReady) return
    const sequence = ++negotiationSequence.current
    let disposed = false
    const video = videoRef.current
    stopPeer(peerRef.current, video)

    const peer = new RTCPeerConnection({ iceServers: [] })
    peerRef.current = peer
    const transceiver = peer.addTransceiver("video", { direction: "recvonly" })
    preferBrowserVp8(transceiver)

    const retryFailedConnection = () => {
      if (disposed || sequence !== negotiationSequence.current) return
      setConnection({ jobId, status: "failed" })
      stopPeer(peer, video)
      if (!automaticRenegotiationUsed.current) {
        automaticRenegotiationUsed.current = true
        setNegotiationVersion((version) => version + 1)
      }
    }

    peer.ontrack = (event) => {
      const stream = event.streams[0] ?? new MediaStream([event.track])
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        void videoRef.current.play().catch(retryFailedConnection)
      }
    }
    peer.onconnectionstatechange = () => {
      if (peer.connectionState === "connected") setConnection({ jobId, status: "connected" })
      if (["failed", "disconnected"].includes(peer.connectionState)) retryFailedConnection()
    }

    void (async () => {
      try {
        setConnection({ jobId, status: "connecting" })
        const offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        await waitForIceGatheringComplete(peer)
        if (!peer.localDescription) throw new Error("WebRTC offer did not produce a local description")
        const answer = await api<SessionDescriptionPayload>(`/monitoring/webcam/${jobId}/webrtc/offer`, {
          method: "POST",
          body: JSON.stringify({ type: peer.localDescription.type, sdp: peer.localDescription.sdp }),
        })
        if (disposed) return
        await peer.setRemoteDescription(answer)
      } catch {
        retryFailedConnection()
      }
    })()

    return () => {
      disposed = true
      if (peerRef.current === peer) peerRef.current = null
      stopPeer(peer, video)
    }
  }, [jobId, jobStatus, signalingReady, negotiationVersion])

  useEffect(() => {
    const video = videoRef.current
    return () => stopPeer(peerRef.current, video)
  }, [])

  const retry = () => {
    if (jobId && !TERMINAL_JOB_STATUSES.has(jobStatus ?? "") && signalingReady) {
      setNegotiationVersion((version) => version + 1)
      return
    }
    startMonitor.mutate()
  }
  const displayStatus = connectionStatus === "connected" || connectionStatus === "connecting" || connectionStatus === "failed"
    ? connectionStatus
    : monitor.data?.webrtc_status?.status ?? jobStatus ?? "waiting"
  const message = startMonitor.isPending
    ? "Starting camera…"
    : monitor.data?.webrtc_status?.error ?? (connectionStatus === "failed" ? "WebRTC connection failed" : "Waiting for room camera…")

  return (
    <Card className="col-span-4 overflow-hidden">
      <CardHeader><CardTitle className="flex items-center gap-2"><Radio className="size-4 text-primary-strong" />Room monitor</CardTitle><CardDescription>UGREEN safety overview · WebRTC video</CardDescription></CardHeader>
      <CardContent>
        <div className="surface-grid relative aspect-video overflow-hidden rounded-lg bg-muted">
          <video
            ref={videoRef}
            data-testid="room-monitor-video"
            data-connection-state={connectionStatus}
            className="size-full object-cover"
            muted
            autoPlay
            playsInline
            aria-label="Live room monitor"
          />
          {connectionStatus !== "connected" && <div data-testid="room-monitor-message" className="absolute inset-0 grid place-items-center bg-muted/80 text-xs text-muted-foreground">{message}</div>}
        </div>
        <div className="mt-3 flex items-center justify-between"><StatusBadge status={displayStatus} /><Button size="sm" variant="ghost" onClick={retry} disabled={startMonitor.isPending}><RefreshCw />Retry</Button></div>
      </CardContent>
    </Card>
  )
}
