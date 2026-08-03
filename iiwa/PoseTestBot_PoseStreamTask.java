package application;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.Charset;
import java.util.concurrent.TimeUnit;

import com.kuka.roboticsAPI.applicationModel.tasks.CycleBehavior;
import com.kuka.roboticsAPI.applicationModel.tasks.RoboticsAPICyclicBackgroundTask;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.task.properties.TaskFunctionProvider;

import org.json.simple.JSONObject;

/**
 * Automatic read-only 10 ms best-effort pose-stream task.
 *
 * Create this source as an automatic cyclic background task in the exact
 * Sunrise.Workbench project. The motion applications control only whether
 * sampling is active; runCyclic() never commands or modifies robot motion.
 * Every runtime exception is contained so an observable stream fault does not
 * silently terminate the automatic task.
 */
public class PoseTestBot_PoseStreamTask
		extends RoboticsAPICyclicBackgroundTask
		implements PoseTestBotPoseStreamFunction {
	private static final String POSE_PACKET_SCHEMA_VERSION = "robot_pose.v1";
	private static final Charset UTF_8 = Charset.forName("UTF-8");
	private static final int TARGET_PERIOD_MS = 10;
	private static final int END_PACKET_COUNT = 3;
	private static final int END_PACKET_INTERVAL_MS = 50;

	private LBR robot;
	private DatagramSocket socket;
	private InetAddress receiverAddress;
	private int receiverPort;
	private String runId;
	private String referenceFramePath;
	private ObjectFrame referenceFrame;
	private String motionName;
	private boolean configured;
	private boolean streaming;
	private long nextSequence;
	private long segmentStartSentPoseCount;
	private long previousPoseStartedNs;
	private long sentPoseCount;
	private long sendFailureCount;
	private long fatalFailureCount;
	private long maximumPoseDeltaNs;
	private long maximumPoseQueryDurationNs;
	private String lastError = "";

	@Override
	public void initialize() {
		robot = getContext().getDeviceFromType(LBR.class);
		initializeCyclic(0, TARGET_PERIOD_MS, TimeUnit.MILLISECONDS,
				CycleBehavior.BestEffort);
		getLogger().info("PoseTestBot pose stream task initialized at a "
				+ TARGET_PERIOD_MS + " ms best-effort target period");
	}

	@TaskFunctionProvider
	public PoseTestBotPoseStreamFunction providePoseStreamFunction() {
		return this;
	}

	@Override
	public synchronized void runCyclic() {
		if (!streaming) {
			return;
		}
		try {
			sendPose(motionName);
		} catch (IOException e) {
			recordSendFailure("cyclic pose send", e);
		} catch (RuntimeException e) {
			/*
			 * Do not let an unhandled exception terminate the automatic
			 * background task. Stop this segment and expose a fatal fault to
			 * the requesting application instead.
			 */
			streaming = false;
			recordFatalFailure("cyclic pose acquisition", e);
		}
	}

	@Override
	public synchronized void configure(String requestedReceiverIp,
			int requestedReceiverPort, String requestedRunId,
			String requestedReferenceFramePath) {
		if (requestedReceiverIp == null
				|| requestedReceiverIp.trim().length() == 0) {
			throw new IllegalArgumentException(
					"receiverIp must be a non-empty address");
		}
		if (requestedReceiverPort < 1 || requestedReceiverPort > 65535) {
			throw new IllegalArgumentException(
					"receiverPort must be between 1 and 65535");
		}
		if (requestedRunId == null || requestedRunId.trim().length() == 0) {
			throw new IllegalArgumentException("runId must be non-empty");
		}
		if (requestedReferenceFramePath == null
				|| !requestedReferenceFramePath.startsWith("/")
				|| requestedReferenceFramePath.endsWith("/")) {
			throw new IllegalArgumentException(
					"referenceFramePath must be an absolute Application Data path");
		}

		ObjectFrame requestedReferenceFrame = getApplicationData().getFrame(
				requestedReferenceFramePath);
		if (requestedReferenceFrame == null) {
			throw new IllegalStateException(
					"Pose reference frame is missing: "
					+ requestedReferenceFramePath);
		}

		DatagramSocket requestedSocket = null;
		try {
			requestedSocket = new DatagramSocket();
			receiverAddress = InetAddress.getByName(
					requestedReceiverIp.trim());
		} catch (IOException e) {
			if (requestedSocket != null) {
				requestedSocket.close();
			}
			throw new IllegalStateException(
					"Unable to configure pose-stream UDP target", e);
		}

		closeSocket();
		socket = requestedSocket;
		receiverPort = requestedReceiverPort;
		runId = requestedRunId.trim();
		referenceFramePath = requestedReferenceFramePath;
		referenceFrame = requestedReferenceFrame;
		motionName = "";
		configured = true;
		streaming = false;
		nextSequence = 0L;
		segmentStartSentPoseCount = 0L;
		previousPoseStartedNs = 0L;
		sentPoseCount = 0L;
		sendFailureCount = 0L;
		fatalFailureCount = 0L;
		maximumPoseDeltaNs = 0L;
		maximumPoseQueryDurationNs = 0L;
		lastError = "";
		getLogger().info("Configured pose stream for run " + runId
				+ " to " + receiverAddress.getHostAddress() + ":"
				+ receiverPort + " in " + referenceFramePath);
	}

	@Override
	public synchronized void startMotion(String requestedMotionName) {
		requireConfigured();
		if (fatalFailureCount > 0) {
			throw new IllegalStateException(
					"Pose stream has a fatal fault: " + lastError);
		}
		if (streaming) {
			throw new IllegalStateException(
					"Pose stream is already active for " + motionName);
		}
		motionName = requiredMotionName(requestedMotionName);
		segmentStartSentPoseCount = sentPoseCount;
		previousPoseStartedNs = 0L;
		streaming = true;
	}

	@Override
	public synchronized long stopMotion() {
		streaming = false;
		motionName = "";
		previousPoseStartedNs = 0L;
		return sentPoseCount - segmentStartSentPoseCount;
	}

	@Override
	public synchronized boolean sendCurrentPose(
			String requestedMotionName) {
		requireConfigured();
		if (fatalFailureCount > 0) {
			throw new IllegalStateException(
					"Pose stream has a fatal fault: " + lastError);
		}
		try {
			sendPose(requiredMotionName(requestedMotionName));
			return true;
		} catch (IOException e) {
			recordSendFailure("single pose send", e);
			return false;
		} catch (RuntimeException e) {
			recordFatalFailure("single pose acquisition", e);
			return false;
		}
	}

	@Override
	public synchronized void finishCapture() {
		requireConfigured();
		streaming = false;
		motionName = "";
		previousPoseStartedNs = 0L;

		long endSequence = nextSequence++;
		JSONObject endObject = packetEnvelope(
				"end", "end", endSequence, System.nanoTime());
		byte[] data = endObject.toJSONString().getBytes(UTF_8);
		int successfulEndSends = 0;
		boolean interrupted = false;
		try {
			for (int i = 0; i < END_PACKET_COUNT; i++) {
				try {
					sendPayload(data);
					successfulEndSends++;
				} catch (IOException e) {
					recordSendFailure(
							"end marker " + (i + 1), e);
				}
				if (i + 1 < END_PACKET_COUNT) {
					try {
						Thread.sleep(END_PACKET_INTERVAL_MS);
					} catch (InterruptedException e) {
						interrupted = true;
						lastError = "end-marker retry interval interrupted: "
								+ e;
						getLogger().error(lastError);
					}
				}
			}
		} finally {
			configured = false;
			closeSocket();
			if (interrupted) {
				Thread.currentThread().interrupt();
			}
		}
		if (successfulEndSends == 0) {
			throw new IllegalStateException(
					"All end-marker transmissions failed: " + lastError);
		}
	}

	@SuppressWarnings("unchecked")
	private void sendPose(String requestedMotionName) throws IOException {
		long poseStartedNs = System.nanoTime();
		Frame currentPose = robot.getCurrentCartesianPosition(
				robot.getFlange(), referenceFrame);
		long poseQueryDurationNs = System.nanoTime() - poseStartedNs;
		long poseDeltaNs = previousPoseStartedNs == 0L
				? 0L : poseStartedNs - previousPoseStartedNs;
		previousPoseStartedNs = poseStartedNs;
		maximumPoseDeltaNs = Math.max(maximumPoseDeltaNs, poseDeltaNs);
		maximumPoseQueryDurationNs = Math.max(
				maximumPoseQueryDurationNs, poseQueryDurationNs);

		JSONObject jsonObject = packetEnvelope(
				"pose", requestedMotionName, nextSequence++, poseStartedNs);
		jsonObject.put("sender_target_period_ms",
				Long.valueOf(TARGET_PERIOD_MS));
		jsonObject.put("sender_previous_pose_delta_ns",
				Long.valueOf(poseDeltaNs));
		jsonObject.put("sender_pose_query_duration_ns",
				Long.valueOf(poseQueryDurationNs));
		jsonObject.put("X", currentPose.getX());
		jsonObject.put("Y", currentPose.getY());
		jsonObject.put("Z", currentPose.getZ());
		jsonObject.put("A", currentPose.getAlphaRad());
		jsonObject.put("B", currentPose.getBetaRad());
		jsonObject.put("C", currentPose.getGammaRad());
		sendPayload(jsonObject.toJSONString().getBytes(UTF_8));
		sentPoseCount++;
	}

	@SuppressWarnings("unchecked")
	private JSONObject packetEnvelope(String packetKind,
			String requestedMotionName, long sequence,
			long senderMonotonicNs) {
		JSONObject jsonObject = new JSONObject();
		jsonObject.put("schema_version", POSE_PACKET_SCHEMA_VERSION);
		jsonObject.put("packet_kind", packetKind);
		jsonObject.put("sequence", Long.valueOf(sequence));
		jsonObject.put("sender_monotonic_ns",
				Long.valueOf(senderMonotonicNs));
		jsonObject.put("sender_wall_timestamp_ms",
				Long.valueOf(System.currentTimeMillis()));
		jsonObject.put("run_id", runId);
		jsonObject.put("motion", requestedMotionName);
		jsonObject.put("from_frame", "robot_flange");
		jsonObject.put("to_frame", "template_base");
		jsonObject.put("sunrise_reference_frame_path", referenceFramePath);
		return jsonObject;
	}

	private void sendPayload(byte[] data) throws IOException {
		DatagramPacket packet = new DatagramPacket(
				data, data.length, receiverAddress, receiverPort);
		socket.send(packet);
	}

	private String requiredMotionName(String requestedMotionName) {
		if (requestedMotionName == null
				|| requestedMotionName.trim().length() == 0
				|| "end".equals(requestedMotionName.trim())) {
			throw new IllegalArgumentException(
					"motionName must be non-empty and cannot be end");
		}
		return requestedMotionName.trim();
	}

	private void requireConfigured() {
		if (!configured || socket == null || socket.isClosed()) {
			throw new IllegalStateException("Pose stream is not configured");
		}
	}

	private void recordSendFailure(String context, Exception error) {
		sendFailureCount++;
		lastError = context + " failed: " + error;
		if (sendFailureCount == 1L || sendFailureCount % 100L == 0L) {
			getLogger().error(lastError + " (send failure "
					+ sendFailureCount + ")");
		}
	}

	private void recordFatalFailure(String context, RuntimeException error) {
		fatalFailureCount++;
		lastError = context + " failed: " + error;
		getLogger().error(lastError
				+ "; cyclic sampling stopped for this capture");
	}

	private void closeSocket() {
		if (socket != null) {
			socket.close();
			socket = null;
		}
	}

	@Override
	public synchronized int getTargetPeriodMs() {
		return TARGET_PERIOD_MS;
	}

	@Override
	public synchronized long getSentPoseCount() {
		return sentPoseCount;
	}

	@Override
	public synchronized long getSendFailureCount() {
		return sendFailureCount;
	}

	@Override
	public synchronized long getFatalFailureCount() {
		return fatalFailureCount;
	}

	@Override
	public synchronized long getMaximumPoseDeltaNs() {
		return maximumPoseDeltaNs;
	}

	@Override
	public synchronized long getMaximumPoseQueryDurationNs() {
		return maximumPoseQueryDurationNs;
	}

	@Override
	public synchronized String getLastError() {
		return lastError;
	}

	@Override
	public synchronized void dispose() {
		streaming = false;
		configured = false;
		closeSocket();
		super.dispose();
	}
}
